"""Combine Project Z's paired studies without duplicating reference training."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
import torch
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.qvf_project_a.train import sha256, write_json
from .report import LABELS


def combine(latent, circuit, output):
    latent, circuit, output = Path(latent), Path(circuit), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    p = json.loads((circuit / 'protocol.json').read_text())
    reference = p['referenced_controls']
    assert str((latent / 'protocol.json').resolve()) == reference['protocol']
    assert sha256(latent / 'protocol.json') == reference['protocol_sha256']
    assert sha256(latent / 'verified/results.json') == reference['verified_results_sha256']
    old = json.loads((latent / 'protocol.json').read_text())
    for key in ('indices', 'completion', 'sampling', 'optimization', 'edge_weight'):
        assert p[key] == old[key], key
    for spec in p['runs']:
        base = next(r for r in old['runs'] if r['name'] == 'attention_amp' and r['config']['seed'] == spec['config']['seed'])
        for key, value in base['config'].items():
            if key != 'architecture':
                assert spec['config'][key] == value
        previous = torch.load(latent / base['directory'] / 'fit/checkpoints/step0.pt', weights_only=False)
        new = torch.load(circuit / spec['directory'] / 'fit/checkpoints/step0.pt', weights_only=False)
        assert torch.equal(previous['codes'], new['codes'])
        for key, value in previous['decoder_state'].items():
            if not key.startswith('quantum.'):
                assert torch.equal(value, new['decoder_state'][key]), (spec['name'], key)
        a = json.loads((latent / base['directory'] / 'COMPLETE.json').read_text())
        b = json.loads((circuit / spec['directory'] / 'COMPLETE.json').read_text())
        assert a['stream_sha256'] == b['stream_sha256']
    all_rows = []
    summaries = {}
    for root in (latent, circuit):
        assert json.loads((root / 'verified/verification.json').read_text())['source_and_checkpoint_hashes_match']
        all_rows.extend(json.loads((root / 'verified/results.json').read_text())['runs'])
        summaries.update(json.loads((root / 'verified/summary.json').read_text())['models'])
    # Independently validate report aggregations using NumPy's sample variance.
    metric_paths = {'train_psnr': ('fit', 'psnr_db'), 'train_edge': ('fit', 'gradient_mse'),
        'completion_psnr': ('completion_unobserved', 'psnr_db'), 'completion_edge': ('completion_unobserved_edges', 'gradient_mse')}
    for name, summary in summaries.items():
        runs = sorted([r for r in all_rows if r['name'] == name], key=lambda r: r['seed'])
        for metric, (group, key) in metric_paths.items():
            values = np.array([r[group][key] for r in runs])
            assert abs(values.mean() - summary[metric]['mean']) < 1e-12
            assert abs(values.std(ddof=1) - summary[metric]['sample_sd']) < 1e-12
    comparisons = [('latent_conv', 'attention_amp'), ('latent_conv', 'latent_pointwise'),
        ('attention_local_shared', 'attention_amp'), ('attention_local_untied', 'attention_amp'),
        ('attention_local_shared', 'attention_local_untied'), ('conv_local_shared', 'latent_conv'),
        ('conv_local_untied', 'latent_conv'), ('conv_local_shared', 'attention_local_shared'),
        ('conv_local_untied', 'attention_local_untied'), ('conv_local_shared', 'conv_local_untied')]
    paired = {}
    for a, b in comparisons:
        values = {}
        for metric in metric_paths:
            delta = np.array(summaries[a][metric]['seeds']) - summaries[b][metric]['seeds']
            values[metric] = {'mean_difference': float(delta.mean()), 'paired_differences': delta.tolist()}
            if 'edge' in metric:
                values[metric]['mean_reduction_percent'] = 100 * (1 - summaries[a][metric]['mean'] / summaries[b][metric]['mean'])
        paired[f'{a} minus {b}'] = values
    write_json(output / 'summary.json', {'models': summaries, 'paired_differences': paired,
        'controls': reference, 'cross_study_initialization_and_streams_checked': True,
        'means_and_sample_sds_independently_checked': True})
    names = list(summaries)
    colours = ['#555555', '#267bb5', '#929292', '#d17b22', '#a75817', '#26885d', '#175936']
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True, layout='constrained')
    for ax, metric, label, factor in zip(axes, ['train_psnr', 'completion_edge', 'completion_psnr'],
            ['Training PSNR (dB) · higher is better', 'Completion edge MSE × 1,000 · lower is better', 'Completion PSNR (dB) · higher is better'], [1, 1000, 1]):
        for y, (name, colour) in enumerate(zip(names, colours)):
            s = summaries[name][metric]
            ax.scatter(np.array(s['seeds']) * factor, [y] * 3, color=colour, alpha=.45, s=25)
            ax.scatter(s['mean'] * factor, y, color=colour, marker='D', s=55)
        ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Project Z · 1,024 images · 10,000 updates · three seeds · five qubits')
    fig.savefig(output / 'quality.png', dpi=180)
    fig.savefig(output / 'quality.pdf')
    plt.close(fig)
    # Fixed examples across every arm, independently verified before combining.
    train, complete = [], []
    originals = None
    for name in names:
        root = latent if name in json.loads((latent / 'verified/summary.json').read_text())['models'] else circuit
        directory = root / f'{name}_seed0' / 'final'
        t = torch.load(directory / 'train.pt', weights_only=False)
        c = torch.load(directory / 'completion.pt', weights_only=False)
        train.append(t['reconstructions'][:12]); complete.append(c['reconstructions'][:12])
        if originals is None:
            originals = (t['originals'][:12], c['originals'][:12])
    mask = torch.zeros_like(originals[1]); observed = p['completion']['observed_pixels']
    mask.flatten(2)[:, :, observed] = originals[1].flatten(2)[:, :, observed]
    save_image(torch.stack((originals[0], *train), 1).flatten(0, 1), output / 'training.png', nrow=len(names) + 1)
    save_image(torch.stack((originals[1], mask, *complete), 1).flatten(0, 1), output / 'completion.png', nrow=len(names) + 2)
    lines = []
    for name, s in summaries.items():
        lines.append(f"{name}: train {s['train_psnr']['mean']:.3f} dB / edge {s['train_edge']['mean']:.7f}; completion {s['completion_psnr']['mean']:.3f} dB / edge {s['completion_edge']['mean']:.7f}")
    for pair, d in paired.items():
        lines.append(f"{pair}: train PSNR {d['train_psnr']['mean_difference']:+.3f}; completion PSNR {d['completion_psnr']['mean_difference']:+.3f}; completion edge reduction {d['completion_edge']['mean_reduction_percent']:+.2f}%")
    (output / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent', type=Path, required=True)
    parser.add_argument('--circuit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    combine(args.latent, args.circuit, args.output)
