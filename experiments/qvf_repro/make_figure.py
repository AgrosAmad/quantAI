"""Side-by-side reconstructions of the first ten training images for several models.

Usage:
    python make_figure.py                 # paper settings at learning rate 0.01, seed 0
    python make_figure.py "" 0            # paper settings at the stated learning rate 0.001, seed 0

Reads the image grids that train.py saves (top row originals, bottom row reconstructions)
and writes results/reconstructions_paper<tag>_seed<seed>.png.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
tag = sys.argv[1] if len(sys.argv) > 1 else "_lr0.01"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

MODELS = [
    ("Plain MLP", f"paper_mlp_relu_n50_seed{seed}{tag}"),
    ("QVF without circuit", f"paper_qvf_gaussian_relu_n50_seed{seed}{tag}_depth0"),
    ("QVF, Gaussian init", f"paper_qvf_gaussian_relu_n50_seed{seed}{tag}"),
    ("QVF, identity init", f"paper_qvf_identity_relu_n50_seed{seed}{tag}"),
]


def rows(name):
    grid = np.asarray(Image.open(HERE / "results" / f"{name}.png").convert("RGB"))
    return grid[1:33], grid[34:66]  # originals, reconstructions (1 px padding around 32 px tiles)


strips = [("Original", rows(MODELS[0][1])[0])]
for label, name in MODELS:
    psnr = json.loads((HERE / "results" / f"{name}.json").read_text())["psnr_mean"]
    strips.append((f"{label}\n{psnr:.1f} dB", rows(name)[1]))

fig, axes = plt.subplots(len(strips), 1, figsize=(11, 1.18 * len(strips)), dpi=150)
for ax, (label, strip) in zip(axes, strips):
    ax.imshow(strip, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, labelpad=8)
fig.subplots_adjust(left=0.16, right=0.995, top=0.99, bottom=0.01, hspace=0.06)
out = HERE / "results" / f"reconstructions_paper{tag}_seed{seed}.png"
fig.savefig(out)
print(out)
