"""Presentation-only Z8 plots; leave frozen experiment sources and metrics intact.

Use --watch beside the existing queue to reapply the style after report updates.
The companion exits when the queue finishes, fails, or stops.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QVF_RED = '#d62728'
OTHER_COLOR = '#007e9a'
PANELS = {'cifar32': 'CIFAR-10 dogs · 32×32',
          'dogs256': 'Stanford Dogs · 256×256'}


def render(root):
    root = Path(root)
    source = (root / 'comparison.json').read_bytes()
    data = json.loads(source)
    outputs = []
    for panel, dataset in PANELS.items():
        rows = [r for r in data['rows'] if r['panel'] == panel]
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), layout='constrained')
        for ax, key, title in zip(axes, ('completion_psnr_db', 'completion_edge_mse'),
                                 ('Completion PSNR · higher is better',
                                  'Completion edge MSE · lower is better')):
            numeric = [v for r in rows for v in r[key + '_values']]
            lo, hi = (min(numeric), max(numeric)) if numeric else (0., 1.)
            span = max(hi - lo, abs(hi) * .02, 1e-6)
            ax.set_xlim(lo - .12 * span, hi + .38 * span)
            for i, row in enumerate(rows):
                baseline = row['architecture'] == 'vanilla_qvf'
                color = QVF_RED if baseline else OTHER_COLOR
                if baseline:
                    ax.axhspan(i - .43, i + .43, color=QVF_RED, alpha=.06)
                values = row[key + '_values']
                if values:
                    ax.scatter(values, [i] * len(values), s=20, facecolors='white',
                               edgecolors=color, zorder=3)
                    if row['complete']:
                        ax.scatter([row[key + '_mean']], [i], marker='D', color=color,
                                   s=35, zorder=4)
                    label = (f'{row[key + "_mean"]:.3f}' if 'psnr' in key
                             else f'{row[key + "_mean"]:.6f}')
                    if not row['complete']:
                        label += f' ({row["seed_count"]}/3)'
                else:
                    label = 'pending'
                ax.text(hi + .34 * span, i, label, ha='right', va='center', fontsize=8,
                        color=QVF_RED if baseline else 'black',
                        fontweight='bold' if baseline else 'normal')
            ax.set_yticks(range(len(rows)), [r['label'] for r in rows], fontsize=8)
            for tick, row in zip(ax.get_yticklabels(), rows):
                if row['architecture'] == 'vanilla_qvf':
                    tick.set_color(QVF_RED)
                    tick.set_fontweight('bold')
            ax.set_ylim(len(rows) - .5, -.7)
            ax.grid(axis='x', alpha=.2)
            ax.set_title(title, fontsize=10)
        state = 'COMPLETE' if data['complete'] else 'PROVISIONAL · measurements or audits pending'
        fig.suptitle(f'{dataset} | 4,744 training images | 50,000 updates | 3 seeds\n{state}',
                     fontsize=12)
        for extension in ('png', 'pdf'):
            target = root / f'{panel}_comparison.{extension}'
            with tempfile.TemporaryDirectory(dir=root) as tmp:
                temporary = Path(tmp) / target.name
                fig.savefig(temporary, dpi=160)
                temporary.replace(target)
            outputs.append(target)
        plt.close(fig)
    status = {'rendered_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
              'vanilla_qvf_color': QVF_RED, 'other_methods_color': OTHER_COLOR,
              'comparison_sha256': hashlib.sha256(source).hexdigest(),
              'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
              'note': 'Presentation only; comparison values and frozen sources are unchanged.'}
    (root / 'plot_style.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)


def fingerprint(root):
    paths = [root / 'comparison.json', root / 'REPORT.md']
    paths += [root / f'{p}_comparison.{ext}' for p in PANELS for ext in ('png', 'pdf')]
    return tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in paths)


def watch(root):
    """Wait for stable report files, then restyle without touching queue state."""
    previous = rendered = None
    while True:
        current = fingerprint(root)
        try:
            queue = json.loads((root / 'queue_status.json').read_text())
            terminal = (queue['phase'] == 'complete' or queue['phase'].startswith('failed')
                        or not Path('/proc', str(queue.get('runner_pid', 'missing'))).exists())
            if current == previous:
                if current != rendered:
                    render(root)
                    current = fingerprint(root)
                    rendered = current
                if terminal:
                    break
        except json.JSONDecodeError:
            pass  # The frozen report writer may still be writing its JSON.
        previous = current
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.watch:
        watch(args.root.resolve())
    else:
        render(args.root.resolve())
