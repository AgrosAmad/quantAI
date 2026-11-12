"""Compact static Z4 decision figure, paired contrasts and fixed-ID galleries."""
import argparse
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments.qvf_project_a.train import write_json, sha256

MAIN = ['attention_chain_q5', 'spatial_star_q5', 'attention_chain_q7', 'spatial_star_q7']
LABELS = ['Attention + shared chain · 5q', 'Spatial slots + shared star · 5q',
          'Attention + shared chain · 7q', 'Spatial slots + shared star · 7q']


def run(root, audit):
    root, audit = Path(root), Path(audit)
    dest = root / 'verified'
    summary = json.loads((dest / 'summary.json').read_text())['models']
    quantum = json.loads((audit / 'summary.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    checks = json.loads((dest / 'verification.json').read_text())
    assert checks['run_count'] == 27 and checks['maximum_reload_rgb_error'] == 0
    assert all(x['parameters_bit_identical_to_z3'] and x['codes_bit_identical_to_z3'] for x in checks['z3_reproduction'])
    assert json.loads((audit / 'independent_verification.json').read_text())['all_passed']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.3, 4.2), width_ratios=[1, 1.25], layout='constrained')
    for y, name in enumerate(MAIN):
        colour = '#1674a8' if name.startswith('attention') else '#c86b19'
        s = summary[name]['completion_psnr']
        a.scatter(s['seeds'], np.full(3, y), color=colour, alpha=.35, s=35)
        a.scatter(s['mean'], y, color=colour, marker='D', s=65, zorder=4)
        resource = quantum[name]['resources']
        cnot, depth = resource['all_to_all_primitives']['CNOT'], resource['all_to_all_depth']
        b.barh(y-.17, cnot, height=.30, color='#337b72', label='CNOT gates' if y == 0 else None)
        b.barh(y+.17, depth, height=.30, color='#91959a', label='Depth (gate layers)' if y == 0 else None)
        b.text(cnot+5, y-.17, str(cnot), va='center', fontsize=10)
        b.text(depth+5, y+.17, str(depth), va='center', fontsize=10)
    a.set_yticks(range(4), LABELS); b.set_yticks(range(4), ['']*4)
    a.set_xlabel('Withheld-pixel PSNR (dB) · higher is better')
    a.set_xlim(20.25, 21.75)
    a.set_title('Completion quality')
    b.set_xlabel('Per coordinate · includes input preparation')
    b.set_title('Logical circuit resources')
    b.legend(loc='upper center', bbox_to_anchor=(.5, -.19), ncol=2, fontsize=9, frameon=False)
    b.set_xlim(0, 395)
    for ax in (a, b):
        ax.set_ylim(3.6, -.6); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Project Z · 1,024 images · 10,000 updates · 3 seeds', fontsize=14)
    fig.savefig(dest / 'decision.png', dpi=190)
    fig.savefig(dest / 'decision.pdf')
    plt.close(fig)
    contrasts = {}
    pairs = [(f'spatial_star_q{q}', f'attention_chain_q{q}') for q in (5, 7)]
    pairs += [(f'spatial_star_q{q}', f'center_star_q{q}') for q in (5, 7)]
    pairs += [(f'{family}_q7', f'{family}_q5') for family in ('attention_chain', 'attention_star', 'spatial_star', 'center_star')]
    pairs += [('attention_chain_wide_q5', 'attention_chain_q7'), ('attention_star_q5', 'attention_chain_q5')]
    for left, right in pairs:
        entry = {}
        for metric in ('train_psnr', 'train_edge', 'completion_psnr', 'completion_edge'):
            x, y = summary[left][metric], summary[right][metric]
            d = np.array(x['seeds']) - np.array(y['seeds'])
            entry[metric] = {'mean_difference': float(d.mean()), 'paired_seed_differences': d.tolist(),
                'sample_sd_difference': float(d.std(ddof=1))}
            if metric.endswith('edge'):
                entry[metric]['relative_error_reduction_percent'] = 100*(1-x['mean']/y['mean'])
        contrasts[f'{left} minus {right}'] = entry
    write_json(dest / 'decision.json', {'comparisons': contrasts,
        'study_protocol_sha256': sha256(root / 'protocol.json'), 'audit_protocol_sha256': sha256(audit / 'protocol.json'),
        'scope': 'Three paired seeds, repeatedly used development images, fixed training budget; no significance or convergence claim.'})

    for kind in ('train', 'completion'):
        first = torch.load(root / f'{MAIN[0]}_seed0/final/{kind}.pt', weights_only=False)
        images, titles = [first['originals'][:6]], ['Original']
        if kind == 'completion':
            masked = torch.zeros_like(images[0])
            pixels = protocol['completion']['observed_pixels']
            masked.flatten(2)[:, :, pixels] = images[0].flatten(2)[:, :, pixels]
            images.append(masked); titles.append('Observed 25%')
        for name, label in zip(MAIN, ('Attention 5q', 'Spatial 5q', 'Attention 7q', 'Spatial 7q')):
            images.append(torch.load(root / f'{name}_seed0/final/{kind}.pt', weights_only=False)['reconstructions'][:6])
            titles.append(label)
        fig, axes = plt.subplots(6, len(images), figsize=(1.5*len(images), 8.8), layout='constrained')
        for j, (column, label) in enumerate(zip(images, titles)):
            axes[0, j].set_title(label, fontsize=10)
            for i in range(6):
                axes[i, j].imshow(column[i].clamp(0, 1).permute(1, 2, 0), interpolation='nearest')
                axes[i, j].axis('off')
        fig.suptitle(f'{kind.capitalize()} · first 6 protocol images · seed 0\nRGB clipped for display only; metrics use raw RGB', fontsize=12)
        fig.savefig(dest / f'main_{kind}_gallery.png', dpi=160)
        plt.close(fig)
    folder = dest / 'decision_source'; folder.mkdir(exist_ok=True)
    shutil.copy2(__file__, folder / Path(__file__).name)
    write_json(folder / 'manifest.json', {Path(__file__).name: sha256(Path(__file__))})
    print(dest / 'decision.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.audit)
