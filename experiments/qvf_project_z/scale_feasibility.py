"""Z5 explicit preparation, joint finite shots and bounded synthetic noise."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.qvf_project_a.models import pixel_coordinates
from experiments.qvf_project_a.train import REPO, configure_runtime, sha256, write_json, save_checkpoint
from experiments.qvf_project_a.verify_token_study import metrics_numpy
from experiments.qvf_project_a.verify_edge_study import masked_edge_numpy
from .feasibility import image_tensor, shot_rgb
from .spatial_gates import inputs_and_probabilities, observable_rgb, primitive_ops, simulate_density, gate_resources, marginal_readout
from .scale_study import resources, load_model, snapshot_sources


def run(root, study_root=None):
    device = configure_runtime('cuda:0', 370)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_root = Path(study_root).resolve() if study_root else REPO / 'experiments/qvf_project_a/results/20261109_v11_interfaces'
    protocol = json.loads((source_root / 'protocol.json').read_text())
    loader = load_model
    names = list(dict.fromkeys(x['name'] for x in protocol['runs']))
    snapshot_sources(root)
    records, checks = [], []
    noise_levels = [(0., 0., 0.), (.0001, .001, .005), (.001, .01, .02)]
    metadata = {'project': 'Z5', 'source_protocol': str(source_root / 'protocol.json'),
        'source_protocol_sha256': sha256(source_root / 'protocol.json'),
        'sources': json.loads((root / 'source_manifest.json').read_text()),
        'shots': [256, 1024, 4096], 'measurement_seeds': list(range(5)),
        'noise_levels_p1_p2_readout': noise_levels,
        'noise': 'Uniform nonidentity Pauli error with probability p1 after RY, p2 after CNOT on both wires jointly; independent bit flips at measurement. Synthetic all-to-all connectivity, includes explicit state preparation. Not calibrated hardware.',
        'selection': 'All 32 completion images for shots, all three training seeds. Gate-noise diagnostic: first four excluded images, seed0, all1024 coordinates. Codes are frozen after noiseless partial-image fitting; evaluates inference robustness, not noisy adaptation.',
        'resource_counting': 'Stored quantum entries45/55, output-relevant45/53 at5/7qubits. Explicit execution includes two removable terminal work-wire RYs at7qubits; architecture is unchanged from Z4.',
        'limits': 'No hardware execution or hardware training demonstration; no image encoder or diffusion. Line routing counts are an explicit unoptimized construction, not a lower bound.'}
    write_json(root / 'protocol.json', metadata)
    for name in names:
        for seed in protocol['seeds']:
            directory = source_root / f'{name}_seed{seed}'
            model, _, _ = loader(directory / 'fit/checkpoints/last.pt', device)
            model.requires_grad_(False)
            q = model.spatial_cfg.qubits
            saved = torch.load(directory / 'final/completion.pt', weights_only=False)
            codes = saved['codes'].to(device)
            originals = saved['originals'].double()
            coords = pixel_coordinates(32, 32, device=device)
            parts = [inputs_and_probabilities(model, batch, coords) for batch in codes.split(16)]
            kind = parts[0][0]
            data, p = np.concatenate([x[1] for x in parts]), np.concatenate([x[2] for x in parts])
            exact = image_tensor(p @ observable_rgb(q), len(codes))
            max_error = (exact - saved['reconstructions']).abs().max().item()
            assert max_error < 5e-6, (name, max_error)
            pixels = protocol['completion']['unobserved_pixels']
            exact_metrics = metrics_numpy(exact, originals, pixels)
            sampled = []
            for shots in metadata['shots']:
                for trial in metadata['measurement_seeds']:
                    prediction = image_tensor(shot_rgb(marginal_readout(p), shots, 20000 + seed * 100 + trial), len(codes))
                    sampled.append({'shots': shots, 'measurement_seed': trial,
                        **metrics_numpy(prediction, originals, pixels),
                        'gradient_mse': masked_edge_numpy(prediction, originals, pixels)['gradient_mse']})
            ops = primitive_ops(kind, data[:8], model.quantum)
            rebuilt = simulate_density(ops, 8, q, device)
            error = float(np.max(np.abs(rebuilt - p[:8])))
            assert error < 5e-6, error
            record = {'name': name, 'seed': seed, 'kind': kind, 'checkpoint_sha256': sha256(directory / 'fit/checkpoints/last.pt'),
                'completion_sha256': sha256(directory / 'final/completion.pt'),
                'qubits': q, 'resources': gate_resources(ops, q), 'exact_completion': exact_metrics, 'finite_shots': sampled}
            for gate in ('RY', 'CNOT'):
                assert record['resources']['all_to_all_primitives'][gate] == resources(model)[f'primitive_{gate.lower()}_per_query']
            checks.append({'name': name, 'seed': seed, 'maximum_saved_rgb_error': max_error, 'maximum_gate_probability_error': error})
            if seed == 0:
                noisy = []
                for p1, p2, readout in noise_levels:
                    output = []
                    for offset in range(0, 4 * 1024, 64):
                        compiled = primitive_ops(kind, data[offset:offset + 64], model.quantum)
                        output.append(simulate_density(compiled, 64, q, device, p1, p2, readout))
                    prob = np.concatenate(output)
                    predicted = image_tensor(prob @ observable_rgb(q), 4)
                    finite = image_tensor(shot_rgb(marginal_readout(prob), 1024, 30101), 4)
                    noisy.append({'p1': p1, 'p2': p2, 'readout': readout,
                        'expectation': metrics_numpy(predicted, originals[:4], pixels),
                        'shots1024': metrics_numpy(finite, originals[:4], pixels)})
                    save_checkpoint(root / f'{name}_noise_{p2:g}.pt', {'probabilities': prob, 'originals': originals[:4], 'expectation': predicted, 'shots1024': finite})
                    print(f'Noise audited {name}: p2={p2:g}', flush=True)
                record['noise_seed0_first_four_images'] = noisy
            records.append(record)
            write_json(root / 'results.json', {'runs': records})
            print(f'Audited {name} seed{seed}: finite shots and explicit gates', flush=True)
    for name, digest in metadata['sources'].items():
        assert sha256(REPO / name) == digest
        assert sha256(root / 'source_snapshot' / name) == digest
    write_json(root / 'verification.json', {'checks': checks, 'all_passed': True,
        'finite_shots_joint_multinomial': True, 'noise_includes_preparation': True})
    write_json(root / 'COMPLETE.json', {'runs': len(records), 'protocol_sha256': sha256(root / 'protocol.json')})
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--study-root', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.root, args.study_root))
