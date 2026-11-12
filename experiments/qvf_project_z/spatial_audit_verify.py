"""Independent expected-shot-error check, full-circuit shifts and audit figures."""
import argparse
import json
import statistics
from pathlib import Path
import numpy as np
import torch
import pennylane as qml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.qvf_project_a.interface_study import load_interface
from experiments.qvf_project_a.models import pixel_coordinates
from experiments.qvf_project_a.train import REPO, configure_runtime, sha256, write_json
from experiments.qvf_project_a.verify_token_study import metrics_numpy, check_metrics
from .spatial_gates import inputs_and_probabilities, observable_rgb, primitive_ops, simulate_density, shot_mse_moments
from .spatial_study import load_model


def report(root):
    root = Path(root)
    info = json.loads((root / 'protocol.json').read_text())
    source = Path(info['source_protocol']).parent
    protocol = json.loads((source / 'protocol.json').read_text())
    assert sha256(source / 'protocol.json') == info['source_protocol_sha256']
    for name, digest in info['sources'].items():
        assert sha256(REPO / name) == digest
    loader = load_model
    device = configure_runtime('cuda:0', 201)
    rows = json.loads((root / 'results.json').read_text())['runs']
    checks = []
    for row in rows:
        name, seed = row['name'], row['seed']
        directory = source / f'{name}_seed{seed}'
        assert sha256(directory / 'fit/checkpoints/last.pt') == row['checkpoint_sha256']
        assert sha256(directory / 'final/completion.pt') == row['completion_sha256']
        model, _, _ = loader(directory / 'fit/checkpoints/last.pt', device)
        model.requires_grad_(False)
        q = model.spatial_cfg.qubits
        saved = torch.load(directory / 'final/completion.pt', weights_only=False)
        codes = saved['codes'].to(device)
        coords = pixel_coordinates(32, 32, device=device)
        parts = [inputs_and_probabilities(model, x, coords) for x in codes.split(16)]
        kind = parts[0][0]
        data, probs = np.concatenate([x[1] for x in parts]), np.concatenate([x[2] for x in parts])
        pixels = protocol['completion']['unobserved_pixels']
        obs = observable_rgb(q)
        mean = probs @ obs
        variance = probs @ (obs ** 2) - mean ** 2
        target = saved['originals'].permute(0, 2, 3, 1).double().numpy().reshape(-1, 3)
        def selected(x):
            return x.reshape(32, 1024, 3)[:, pixels]
        noiseless_mse = float(np.mean(selected((mean - target) ** 2)))
        check = {'name': name, 'seed': seed, 'expected_shot_mse': {}}
        for shots in info['shots']:
            expected = noiseless_mse + float(selected(variance).mean()) / shots
            values = [r['mse'] for r in row['finite_shots'] if r['shots'] == shots]
            measured = statistics.mean(values)
            sem = statistics.stdev(values) / len(values) ** .5
            exact_mean, exact_variance = shot_mse_moments(probs, target, shots)
            selected_mean = exact_mean.reshape(32, 1024)[:, pixels]
            selected_variance = exact_variance.reshape(32, 1024)[:, pixels]
            assert abs(float(selected_mean.mean()) - expected) < 1e-12
            analytic_sem = float(np.sqrt(selected_variance.sum() / selected_variance.size**2 / len(values)))
            assert abs(expected - measured) < 6 * analytic_sem + 1e-7, (name, shots, expected, measured, analytic_sem)
            check['expected_shot_mse'][shots] = {'analytic': expected, 'sample_mean': measured,
                'empirical_monte_carlo_sem': sem, 'analytic_monte_carlo_sem': analytic_sem}
        if seed == 0:
            ops = primitive_ops(kind, data[:1], model.quantum)
            errors = []
            for index in (5 if kind == 'slot' else 0, len(ops) - 1 - (q - 5)):
                assert ops[index][0] == 'RY'
                angle = float(np.asarray(ops[index][2]).reshape(-1)[0])
                @qml.qnode(qml.device('default.qubit', wires=q), interface='torch', diff_method='backprop')
                def expectation(theta):
                    for j, (gate, wires, value) in enumerate(ops):
                        if gate == 'CNOT':
                            qml.CNOT(wires=wires)
                        else:
                            qml.RY(theta if j == index else float(np.asarray(value).reshape(-1)[0]), wires=wires)
                    return tuple(qml.expval(qml.PauliZ(w)) for w in range(5))
                theta = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
                value = torch.stack(expectation(theta)).sum()
                exact = torch.autograd.grad(value, theta)[0].item()
                values = []
                signs = 1. - 2. * ((np.arange(2**q)[:, None] >> np.arange(q-1, q-6, -1)) & 1)
                for delta in (np.pi / 2, -np.pi / 2):
                    altered = list(ops)
                    altered[index] = ('RY', ops[index][1], np.array(angle + delta))
                    values.append((simulate_density(altered, 1, q) @ signs).sum())
                error = abs(exact - (values[0] - values[1]) / 2)
                assert error < 1e-10, error
                errors.append({'primitive_index': index, 'absolute_gradient_error': error})
            check['trained_circuit_primitive_shift_tests'] = errors
            for noise in row['noise_seed0_first_four_images']:
                payload = torch.load(root / f"{name}_noise_{noise['p2']:g}.pt", weights_only=False)
                for key in ('expectation', 'shots1024'):
                    check_metrics(metrics_numpy(payload[key], payload['originals'], pixels), noise[key])
        checks.append(check)
    write_json(root / 'independent_verification.json', {'all_passed': True, 'checks': checks,
        'scope': 'Analytical shot-MSE expectation and exact joint fourth-moment Monte Carlo standard error vs five independent measurement seeds; full trained-circuit primitive shifts at data-dependent input and final measured-wire rotation vs PennyLane autograd; saved noisy-image metrics recomputed. Not a full hardware-training experiment.'})
    names = list(dict.fromkeys(row['name'] for row in rows))
    summary = {}
    for name in names:
        selected_rows = [r for r in rows if r['name'] == name]
        summary[name] = {'exact_completion_psnr': statistics.mean(r['exact_completion']['psnr_db'] for r in selected_rows),
            'shot_psnr': {str(s): statistics.mean(x['psnr_db'] for r in selected_rows for x in r['finite_shots'] if x['shots'] == s) for s in info['shots']},
            'resources': selected_rows[0]['resources'],
            'gatewise_gradient_recipe': {'shifted_evaluations_per_query': 2 * (selected_rows[0]['resources']['all_to_all_primitives']['RY'] - (5 if name.startswith(('spatial', 'center')) else 0)),
                'description': 'Two shifts per data-dependent or trained primitive RY occurrence, including amplitude preparation but excluding five fixed slot-preparation gates; chain-rule aggregation for shared/source parameters. Conservative executable recipe, not a lower bound; excludes forward, optimizer and classical work.'}}
    write_json(root / 'summary.json', summary)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    from .spatial_report import LABELS as labels
    for name in names:
        s = summary[name]
        line, = axes[0].plot(info['shots'], [s['shot_psnr'][str(x)] for x in info['shots']], marker='o', label=labels[name])
        axes[0].axhline(s['exact_completion_psnr'], linestyle=':', color=line.get_color(), alpha=.7)
        row = next(r for r in rows if r['name'] == name and r['seed'] == 0)
        noise = row['noise_seed0_first_four_images']
        axes[1].plot(range(3), [x['shots1024']['psnr_db'] for x in noise], marker='o', label=labels[name])
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(info['shots'], [str(s) for s in info['shots']])
    axes[0].set_xlabel('Shots per coordinate · dotted: exact expectations')
    axes[0].set_title('32 excluded images · 3 training seeds · 5 shot seeds')
    axes[1].set_xticks(range(3), ['No gate noise', 'p2 = 0.001', 'p2 = 0.01'])
    axes[1].set_xlabel('Synthetic Pauli/readout noise · includes preparation')
    axes[1].set_title('First 4 excluded images · seed 0 · 1,024 shots')
    for ax in axes:
        ax.set_ylabel('Withheld-pixel PSNR (dB)')
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.savefig(root / 'feasibility.png', dpi=170)
    fig.savefig(root / 'feasibility.pdf')
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    report(parser.parse_args().root)
