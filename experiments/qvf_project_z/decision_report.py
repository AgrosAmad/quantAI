"""Compact scientific figures for the completed initial Project Z studies."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(results):
    results = Path(results)
    output = results / '20261110_z_summary'
    models = json.loads((output / 'summary.json').read_text())['models']
    names = list(models)
    labels = ['QVF head', 'Latent 3×3 conv', 'Pointwise control', 'Shared local head',
              'Untied local head', 'Conv + shared head', 'Conv + untied head']
    colours = ['#626262', '#277cb8', '#999999', '#d47c20', '#97551d', '#2b8c62', '#195c40']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.7), sharey=True, layout='constrained',
                             gridspec_kw={'width_ratios': [1, 1, .85]})
    for axis, metric, scale, title in zip(axes[:2], ['train_psnr', 'completion_edge'], [1, 1000],
            ['Training PSNR (dB)\nHigher is better', 'Completion edge MSE × 1,000\nLower is better']):
        for y, (name, colour) in enumerate(zip(names, colours)):
            values = models[name][metric]
            axis.scatter(np.array(values['seeds']) * scale, [y] * 3, s=24, alpha=.45, color=colour)
            axis.scatter(values['mean'] * scale, y, s=60, marker='D', color=colour)
        axis.set_xlabel(title)
        axis.grid(axis='x', alpha=.18)
    counts = [230 if 'local' not in n else 110 for n in names]
    axes[2].barh(range(7), counts, height=.5, color=colours, alpha=.8)
    for y, value in enumerate(counts):
        axes[2].text(value + 4, y, str(value), va='center', fontsize=10)
    axes[2].set_xlim(0, 270)
    axes[2].set_xticks([0, 100, 200])
    axes[2].set_xlabel('CNOT gates per query\nIncludes state preparation')
    axes[0].set_yticks(range(7), labels)
    axes[0].invert_yaxis()
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'decision.{extension}', dpi=180)
    plt.close(fig)

    old_root, new_root = results / '20261110_z2_feasibility', results / '20261110_z3_feasibility'
    for root in (old_root, new_root):
        assert json.loads((root / 'independent_verification.json').read_text())['all_passed']
    means = {**json.loads((old_root / 'summary.json').read_text()), **json.loads((new_root / 'summary.json').read_text())}
    rows = json.loads((old_root / 'results.json').read_text())['runs'] + json.loads((new_root / 'results.json').read_text())['runs']
    selected = ['attention_amp', 'attention_local_shared', 'attention_local_untied']
    selected_labels = ['QVF head', 'Shared local head', 'Untied local head']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
    for name, label, colour in zip(selected, selected_labels, ['#626262', '#d47c20', '#97551d']):
        row = means[name]
        shots = [256, 1024, 4096]
        axes[0].plot(shots, [row['shot_psnr'][str(s)] for s in shots], marker='o', color=colour, label=label)
        axes[0].axhline(row['exact_completion_psnr'], linestyle=':', color=colour, alpha=.7)
        noise = next(x for x in rows if x['name'] == name and x['seed'] == 0)['noise_seed0_first_four_images']
        axes[1].plot(range(3), [x['shots1024']['psnr_db'] for x in noise], marker='o', color=colour, label=label)
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(shots, [str(s) for s in shots])
    axes[0].set_xlabel('Shots per coordinate\nDotted lines: exact expectations')
    axes[0].set_title('32 images · 3 training seeds · 5 shot seeds', fontsize=11)
    axes[1].set_xticks(range(3), ['No gate noise', 'p2 = 0.001', 'p2 = 0.01'])
    axes[1].set_xlabel('Synthetic gate/readout noise, including preparation')
    axes[1].set_title('First 4 images · seed 0 · 1,024 shots', fontsize=11)
    for ax in axes:
        ax.set_ylabel('Withheld-pixel PSNR (dB)')
        ax.legend(fontsize=9)
        ax.grid(alpha=.18)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'feasibility_comparison.{extension}', dpi=180)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    report(parser.parse_args().results)
