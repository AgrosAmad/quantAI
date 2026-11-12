"""Explicit gate preparation, joint finite shots and density-matrix noise audit.

Noise is synthetic: uniform nonidentity Pauli channel after each primitive gate,
plus independent classical bit flips at readout. No hardware calibration claim.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
from experiments.qvf_project_a.interface_study import load_interface
from experiments.qvf_project_a.interface_qvf import angle_state
from experiments.qvf_project_a.models import pixel_coordinates
from experiments.qvf_project_a.train import REPO, configure_runtime, sha256, write_json, save_checkpoint
from experiments.qvf_project_a.verify_token_study import metrics_numpy
from experiments.qvf_project_a.verify_edge_study import masked_edge_numpy
from .models import LocalQuantum


def observable_rgb():
    bits = (np.arange(32)[:, None] >> np.arange(4, -1, -1)) & 1
    z = 1. - 2. * bits
    return np.stack((z[:, 0] + z[:, 3], z[:, 1] + z[:, 4], z[:, 2]), -1)


@torch.no_grad()
def inputs_and_probabilities(model, codes, coordinates):
    local, position = model.retrieve(codes, coordinates)
    x = torch.cat((*local, position[None].expand(len(codes), -1, -1)), -1).flatten(0, 1)
    encoded = model.amplitudes(x)
    if hasattr(model, 'position_weights'):
        offsets = torch.einsum('pc,lqc->plq', coordinates, model.position_weights)
        data = (torch.pi * encoded.tanh().reshape(len(codes), len(coordinates), 5, 5) + offsets[None]).flatten(0, 1)
        state = angle_state(data, model.quantum)
        kind = 'angle'
    else:
        data = encoded
        state = data @ model.quantum.circuit_matrix()
        kind = 'amplitude'
    p = state.square().double()
    p /= p.sum(-1, keepdim=True)
    return kind, data.detach().cpu().double().numpy(), p.cpu().numpy()


def primitive_ops(kind, data, quantum):
    """RY/CNOT list; vector angles allow batch preparation without free state load."""
    ops = []
    if kind == 'amplitude':
        a = np.asarray(data, dtype=np.float64)
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        if np.any(a < 0):
            raise ValueError('This preparation audit supports positive real QVF amplitudes')
        reverse = list(range(5))[::-1]
        for k in range(5, 0, -1):
            squares = a.reshape(len(a), 2 ** (5 - k), 2 ** k) ** 2
            total = squares.sum(-1)
            ratio = np.divide(squares[..., 2 ** (k - 1):].sum(-1), total,
                              out=np.zeros_like(total), where=total > 0)
            theta = compute_theta(2 * np.arcsin(np.sqrt(np.clip(ratio, 0, 1))))
            controls, target = reverse[k:], reverse[k - 1]
            if not controls:
                ops.append(qml.RY(theta[:, 0], wires=target))
            else:
                code = gray_code(len(controls))
                for i in range(len(code)):
                    ops.append(qml.RY(theta[:, i], wires=target))
                    bit = (int(code[i], 2) ^ int(code[(i + 1) % len(code)], 2)).bit_length() - 1
                    ops.append(qml.CNOT(wires=[controls[bit], target]))
    weights = quantum.weights.detach().cpu().double().numpy()
    for layer in range(5):
        if kind == 'angle':
            ops.extend(qml.RY(data[:, layer, wire], wires=wire) for wire in range(5))
        if isinstance(quantum, LocalQuantum):
            filters = quantum.filters.detach().cpu().double().numpy()
            for index, (a, b) in enumerate(quantum.pairs):
                x = filters[layer, 0 if quantum.shared else index]
                ops.extend((qml.RY(x[0], wires=a), qml.RY(x[1], wires=b)))
                ops.extend(qml.CRY(x[2], wires=[a, b]).decomposition())
                ops.extend(qml.CRY(x[3], wires=[b, a]).decomposition())
        else:
            cond = quantum.cond.detach().cpu().double().numpy()
            for a in range(5):
                for b in range(5):
                    if a != b:
                        ops.extend(qml.CRY(cond[a, b, layer], wires=[a, b]).decomposition())
        ops.extend(qml.RY(weights[wire, layer], wires=wire) for wire in range(5))
    return [(op.name, tuple(int(w) for w in op.wires), np.asarray(op.data[0]) if op.data else None) for op in ops]


def gate_resources(ops):
    def depth(sequence):
        available = [0] * 5
        for _, wires, _ in sequence:
            end = max(available[w] for w in wires) + 1
            for w in wires:
                available[w] = end
        return max(available)
    routed = []
    for name, wires, angle in ops:
        if name != 'CNOT' or abs(wires[0] - wires[1]) == 1:
            routed.append((name, wires, angle))
            continue
        a, b = wires
        direction = 1 if b > a else -1
        swaps = [(x, x + direction) for x in range(a, b - direction, direction)]
        for x, y in swaps:
            routed.extend([('CNOT', (x, y), None), ('CNOT', (y, x), None), ('CNOT', (x, y), None)])
        routed.append(('CNOT', (b - direction, b), None))
        for x, y in reversed(swaps):
            routed.extend([('CNOT', (x, y), None), ('CNOT', (y, x), None), ('CNOT', (x, y), None)])
    return {'all_to_all_primitives': dict(Counter(o[0] for o in ops)),
            'all_to_all_depth': depth(ops),
            'line_restore_after_each_gate_primitives': dict(Counter(o[0] for o in routed)),
            'line_restore_after_each_gate_depth': depth(routed),
            'routing': 'Explicit nonoptimized SWAP route; restore logical order after each nonlocal CNOT. All-to-all noise run excludes this routing.'}


def depolarize(rho, wires, probability):
    if probability == 0:
        return rho
    wires = list(wires)
    rest = [w for w in range(5) if w not in wires]
    order = wires + rest
    permutation = [0] + [w + 1 for w in order] + [w + 6 for w in order]
    inverse = np.argsort(permutation).tolist()
    d, r = 2 ** len(wires), 2 ** len(rest)
    matrix = rho.reshape(-1, *([2] * 10)).permute(permutation).reshape(-1, d, r, d, r)
    reduced = torch.einsum('baiaj->bij', matrix)
    mixed = torch.eye(d, device=rho.device, dtype=rho.dtype)[None, :, None, :, None] * reduced[:, None, :, None, :] / d
    mixed = mixed.reshape(-1, *([2] * 10)).permute(inverse).reshape_as(rho)
    alpha = probability * d * d / (d * d - 1)
    return (1 - alpha) * rho + alpha * mixed


@torch.no_grad()
def simulate_density(ops, batch, device='cpu', p1=0., p2=0., readout=0.):
    rho = torch.zeros(batch, 32, 32, dtype=torch.float64, device=device)
    rho[:, 0, 0] = 1.
    indices = torch.arange(32, device=device)
    for name, wires, angle in ops:
        if name == 'RY':
            angle = torch.as_tensor(np.array(angle, copy=True), dtype=rho.dtype, device=device)
            if angle.ndim == 0:
                angle = angle.expand(batch)
            c, s = (angle / 2).cos().reshape(batch, *([1] * 9)), (angle / 2).sin().reshape(batch, *([1] * 9))
            state = rho.reshape(batch, *([2] * 10))
            for axis in (wires[0] + 1, wires[0] + 6):
                a, b = state.unbind(axis)
                state = torch.stack((c * a - s * b, s * a + c * b), axis)
            rho = state.reshape(batch, 32, 32)
            rho = depolarize(rho, wires, p1)
        elif name == 'CNOT':
            a, b = wires
            permutation = indices ^ (((indices >> (4 - a)) & 1) << (4 - b))
            rho = rho[:, permutation][:, :, permutation]
            rho = depolarize(rho, wires, p2)
        else:
            raise ValueError(name)
    assert (rho.diagonal(dim1=1, dim2=2).sum(-1) - 1).abs().max() < 1e-9
    p = rho.diagonal(dim1=1, dim2=2).clone()
    for wire in range(5):
        p = (1 - readout) * p + readout * p[:, indices ^ (1 << (4 - wire))]
    assert p.min() > -1e-10
    p = p.clamp_min(0)
    return (p / p.sum(-1, keepdim=True)).cpu().numpy()


def shot_rgb(probabilities, shots, seed):
    """Exact multinomial via conditional binomials; all Z channels share counts."""
    p = np.asarray(probabilities, dtype=np.float64)
    rng = np.random.default_rng(seed)
    remaining = np.full(len(p), shots, dtype=np.int64)
    mass = np.ones(len(p))
    rgb = np.zeros((len(p), 3))
    observable = observable_rgb()
    for column in range(31):
        fraction = np.divide(p[:, column], mass, out=np.zeros_like(mass), where=mass > 1e-15)
        count = rng.binomial(remaining, np.clip(fraction, 0, 1))
        rgb += count[:, None] * observable[column]
        remaining -= count
        mass -= p[:, column]
    rgb += remaining[:, None] * observable[31]
    return rgb / shots


def image_tensor(flat, count):
    return torch.from_numpy(flat.reshape(count, 32, 32, 3).transpose(0, 3, 1, 2).copy())


def run(root, study_root=None):
    device = configure_runtime('cuda:0', 370)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_root = Path(study_root).resolve() if study_root else REPO / 'experiments/qvf_project_a/results/20261109_v11_interfaces'
    protocol = json.loads((source_root / 'protocol.json').read_text())
    if study_root:
        from .study import load_model
        loader = load_model
        names = list(dict.fromkeys(x['name'] for x in protocol['runs']))
    else:
        loader, names = load_interface, ['attention_amp', 'attention_angle']
    records, checks = [], []
    noise_levels = [(0., 0., 0.), (.0001, .001, .005), (.001, .01, .02)]
    metadata = {'project': 'Z', 'source_protocol': str(source_root / 'protocol.json'),
        'source_protocol_sha256': sha256(source_root / 'protocol.json'),
        'sources': {str(Path(__file__).relative_to(REPO)): sha256(Path(__file__))},
        'shots': [256, 1024, 4096], 'measurement_seeds': list(range(5)),
        'noise_levels_p1_p2_readout': noise_levels,
        'noise': 'Uniform nonidentity Pauli error with probability p1 after RY, p2 after CNOT on both wires jointly; independent bit flips at measurement. Synthetic all-to-all connectivity, includes explicit state preparation. Not calibrated hardware.',
        'selection': 'All 32 completion images for shots, all three training seeds. Gate-noise diagnostic: first four excluded images, seed0, all1024 coordinates. Codes are frozen after noiseless partial-image fitting; evaluates inference robustness, not noisy adaptation.',
        'limits': 'No hardware execution or hardware training demonstration; no image encoder or diffusion. Line routing counts are an explicit unoptimized construction, not a lower bound.'}
    write_json(root / 'protocol.json', metadata)
    for name in names:
        for seed in protocol['seeds']:
            directory = source_root / f'{name}_seed{seed}'
            model, _, _ = loader(directory / 'fit/checkpoints/last.pt', device)
            model.requires_grad_(False)
            saved = torch.load(directory / 'final/completion.pt', weights_only=False)
            codes = saved['codes'].to(device)
            originals = saved['originals'].double()
            coords = pixel_coordinates(32, 32, device=device)
            parts = [inputs_and_probabilities(model, batch, coords) for batch in codes.split(16)]
            kind = parts[0][0]
            data, p = np.concatenate([x[1] for x in parts]), np.concatenate([x[2] for x in parts])
            exact = image_tensor(p @ observable_rgb(), len(codes))
            max_error = (exact - saved['reconstructions']).abs().max().item()
            assert max_error < 2e-6, (name, max_error)
            pixels = protocol['completion']['unobserved_pixels']
            exact_metrics = metrics_numpy(exact, originals, pixels)
            sampled = []
            for shots in metadata['shots']:
                for trial in metadata['measurement_seeds']:
                    prediction = image_tensor(shot_rgb(p, shots, 20000 + seed * 100 + trial), len(codes))
                    sampled.append({'shots': shots, 'measurement_seed': trial,
                        **metrics_numpy(prediction, originals, pixels),
                        'gradient_mse': masked_edge_numpy(prediction, originals, pixels)['gradient_mse']})
            ops = primitive_ops(kind, data[:8], model.quantum)
            rebuilt = simulate_density(ops, 8, device)
            error = float(np.max(np.abs(rebuilt - p[:8])))
            assert error < 2e-6, error
            record = {'name': name, 'seed': seed, 'kind': kind, 'checkpoint_sha256': sha256(directory / 'fit/checkpoints/last.pt'),
                'completion_sha256': sha256(directory / 'final/completion.pt'),
                'resources': gate_resources(ops), 'exact_completion': exact_metrics, 'finite_shots': sampled}
            checks.append({'name': name, 'seed': seed, 'maximum_saved_rgb_error': max_error, 'maximum_gate_probability_error': error})
            if seed == 0:
                noisy = []
                for p1, p2, readout in noise_levels:
                    output = []
                    for offset in range(0, 4 * 1024, 128):
                        compiled = primitive_ops(kind, data[offset:offset + 128], model.quantum)
                        output.append(simulate_density(compiled, 128, device, p1, p2, readout))
                    prob = np.concatenate(output)
                    predicted = image_tensor(prob @ observable_rgb(), 4)
                    finite = image_tensor(shot_rgb(prob, 1024, 30101), 4)
                    noisy.append({'p1': p1, 'p2': p2, 'readout': readout,
                        'expectation': metrics_numpy(predicted, originals[:4], pixels),
                        'shots1024': metrics_numpy(finite, originals[:4], pixels)})
                    save_checkpoint(root / f'{name}_noise_{p2:g}.pt', {'probabilities': prob, 'originals': originals[:4], 'expectation': predicted, 'shots1024': finite})
                    print(f'Noise audited {name}: p2={p2:g}', flush=True)
                record['noise_seed0_first_four_images'] = noisy
            records.append(record)
            write_json(root / 'results.json', {'runs': records})
            print(f'Audited {name} seed{seed}: finite shots and explicit gates', flush=True)
    write_json(root / 'verification.json', {'checks': checks, 'all_passed': True,
        'finite_shots_joint_multinomial': True, 'noise_includes_preparation': True})
    write_json(root / 'COMPLETE.json', {'runs': len(records), 'protocol_sha256': sha256(root / 'protocol.json')})
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--study-root', type=Path)
    args = parser.parse_args()
    print(run(args.root, args.study_root))
