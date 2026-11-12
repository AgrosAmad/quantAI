"""Compare Z5 with Z4 on fixed evaluation IDs and audit training duration."""
import argparse
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments.qvf_project_a.train import sha256, write_json
from experiments.qvf_project_a.verify_token_study import metrics_numpy, check_metrics
from .scale_report import LABELS, stats


def run(root):
    root = Path(root).resolve(); output = root / 'verified'
    protocol = json.loads((root / 'protocol.json').read_text())
    pilot = Path(protocol['pilot_protocol']).parent
    assert sha256(pilot / 'protocol.json') == protocol['pilot_protocol_sha256']
    assert sha256(pilot / 'verified/results.json') == protocol['pilot_verified_results_sha256']
    current = json.loads((output / 'results.json').read_text())['runs']
    previous = json.loads((pilot / 'verified/results.json').read_text())['runs']
    indexed = {(r['name'], r['seed']): r for r in previous}
    records = []
    for row in current:
        name, seed = row['name'], row['seed']; old = indexed[name, seed]
        fit = torch.load(root / f'{name}_seed{seed}/final/train.pt', weights_only=False)
        prior = torch.load(pilot / f'{name}_seed{seed}/final/train.pt', weights_only=False)
        assert fit['dataset_indices'][:1024] == prior['dataset_indices']
        assert torch.equal(fit['originals'][:1024], prior['originals'])
        common = metrics_numpy(fit['reconstructions'][:1024], fit['originals'][:1024])
        additional = metrics_numpy(fit['reconstructions'][1024:], fit['originals'][1024:])
        checked_pilot = metrics_numpy(prior['reconstructions'], prior['originals'])
        check_metrics(checked_pilot, old['fit'])
        records.append({'name': name, 'seed': seed, 'common_training_images': 1024,
            'additional_training_images': 3720, 'pilot_common_fit': checked_pilot,
            'large_common_fit': common, 'large_additional_fit': additional,
            'large_full_fit': row['fit'], 'large_10000_fit': row['step10000'],
            'large_completion': row['completion_unobserved'], 'pilot_completion': old['completion_unobserved'],
            'large_completion_edges': row['completion_unobserved_edges'], 'pilot_completion_edges': old['completion_unobserved_edges']})
        del fit, prior
    models = {}
    for name in dict.fromkeys(r['name'] for r in current):
        rows = sorted([r for r in records if r['name'] == name], key=lambda r: r['seed'])
        value = {}
        for group, metrics in (
            ('pilot_common_fit', ('psnr_db', 'gradient_mse')),
            ('large_common_fit', ('psnr_db', 'gradient_mse')),
            ('large_additional_fit', ('psnr_db', 'gradient_mse')),
            ('large_full_fit', ('psnr_db', 'gradient_mse')),
            ('large_10000_fit', ('psnr_db', 'gradient_mse')),
            ('large_completion', ('psnr_db', 'mse')),
            ('pilot_completion', ('psnr_db', 'mse')),
            ('large_completion_edges', ('gradient_mse',)),
            ('pilot_completion_edges', ('gradient_mse',))):
            value[group] = {metric: stats([r[group][metric] for r in rows]) for metric in metrics}
        comparisons = [('large_common_fit', 'pilot_common_fit', 'same_1024_fit_change', ('psnr_db', 'gradient_mse')),
            ('large_full_fit', 'large_10000_fit', 'same_4744_duration_change', ('psnr_db', 'gradient_mse')),
            ('large_completion', 'pilot_completion', 'same_completion_change', ('psnr_db', 'mse')),
            ('large_completion_edges', 'pilot_completion_edges', 'same_completion_edge_change', ('gradient_mse',))]
        for left, right, label, metrics in comparisons:
            value[label] = {metric: stats([r[left][metric]-r[right][metric] for r in rows]) for metric in metrics}
        models[name] = value
    write_json(output / 'pilot_comparison.json', {'models': models,
        'large_protocol_sha256': sha256(root / 'protocol.json'), 'pilot_protocol_sha256': protocol['pilot_protocol_sha256'],
        'large_results_sha256': sha256(output / 'results.json'), 'pilot_results_sha256': protocol['pilot_verified_results_sha256'],
        'limits': 'Data and update count change together between studies. Same 32 development completion images/mask and same 1000-step code inference. The common1024 training subset is scored separately. Within the large study, step10k versus50k compares fitting duration on identical4744 targets. These are not convergence or fresh-test claims.'})
    write_json(output / 'pilot_comparison_per_seed.json', {'runs': records})
    main = ['attention_chain_q5', 'spatial_star_q5', 'attention_chain_q7', 'spatial_star_q7']
    labels = ['Attention 5q', 'Spatial 5q', 'Attention 7q', 'Spatial 7q']
    colours = ['#1674a8', '#c86b19', '#1674a8', '#c86b19']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.8, 4.3), layout='constrained')
    for name, label, colour in zip(main, labels, colours):
        subset = [r for r in current if r['name'] == name]
        style = '--' if name.endswith('q7') else '-'
        curves = [[p['fit']['psnr_db'] for p in r['curve'] if p['step'] >= 10000] for r in subset]
        points = [p['step'] for p in subset[0]['curve'] if p['step'] >= 10000]
        a.plot(points, np.mean(curves, axis=0), style, color=colour, marker='o', label=label)
        y = [models[name][group]['psnr_db']['mean'] for group in ('pilot_completion', 'large_completion')]
        b.plot([0, 1], y, style, color=colour, marker='o', label=label)
    a.set_title('Longer training on the same 4,744 images')
    a.set_xlabel('Updates'); a.set_ylabel('Training PSNR (dB)')
    b.set_title('Completion on the same 32 development images')
    b.set_xticks([0, 1], ['1,024 images\n10,000 updates', '4,744 images\n50,000 updates'])
    b.set_ylabel('Withheld-pixel PSNR (dB)')
    for ax in (a, b):
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('Three-seed means · dataset and budget change together between studies')
    fig.savefig(output / 'pilot_comparison.png', dpi=180)
    fig.savefig(output / 'pilot_comparison.pdf')
    plt.close(fig)
    folder = output / 'comparison_source'; folder.mkdir(exist_ok=True)
    shutil.copy2(__file__, folder / Path(__file__).name)
    write_json(folder / 'manifest.json', {Path(__file__).name: sha256(Path(__file__))})
    for name, v in models.items():
        print(name, '10k->50k fit PSNR', v['same_4744_duration_change']['psnr_db']['mean'],
            'pilot->large completion PSNR', v['same_completion_change']['psnr_db']['mean'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
