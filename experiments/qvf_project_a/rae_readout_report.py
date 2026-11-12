"""Static figures and fixed reconstruction galleries for the verified RAE study."""

import argparse
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def gallery(path, rows, count, columns=8, cell=80):
    label_width, pad = 190, 6
    blocks = (count + columns - 1) // columns
    height = blocks * (len(rows) * (cell + pad) + 14)
    canvas = Image.new("RGB", (label_width + columns * (cell + pad), height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    for block in range(blocks):
        for row, (label, tensors) in enumerate(rows):
            y = block * (len(rows) * (cell + pad) + 14) + row * (cell + pad)
            draw.text((8, y + cell // 2 - 8), label, fill="#17283d", font=font)
            for col in range(columns):
                index = block * columns + col
                if index >= count:
                    break
                pixels = (tensors[index].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                tile = Image.fromarray(pixels).resize((cell, cell), Image.Resampling.NEAREST)
                canvas.paste(tile, (label_width + col * (cell + pad), y))
    canvas.save(path)


def generate(study):
    root = Path(study).resolve()
    output = root / "verified"
    data = json.loads((output / "results.json").read_text())
    if data["verification"]["status"] != "passed":
        raise ValueError("Report requires successful independent verification")
    summaries = data["summaries"]
    seed_count = len(data["protocol"]["seeds"])
    baseline = Path(data["protocol"]["cache_manifest"]["baseline_directory"])
    colours = {"qvf": "#3366bd", "classical": "#b45f35", "reference": "#788695"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.2, 3.4), layout="constrained")
    labels = ["Published RAE\n(no local training)", "QVF head\n50,000 updates", "Classical head\n50,000 updates"]
    values = [data["reference"]["validation"]["raw"]["psnr_db"],
              summaries["qvf"]["final"]["validation"]["psnr_db"]["mean"],
              summaries["classical"]["final"]["validation"]["psnr_db"]["mean"]]
    errors = [0, summaries["qvf"]["final"]["validation"]["psnr_db"]["sample_sd"] or 0,
              summaries["classical"]["final"]["validation"]["psnr_db"]["sample_sd"] or 0]
    bars = ax.bar(labels, values, yerr=errors, capsize=5, width=.58,
                  color=[colours["reference"], colours["qvf"], colours["classical"]])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .6, f"{value:.2f}", ha="center", weight="bold")
    ax.set_ylim(0, max(values) + 3)
    ax.set_ylabel("Validation PSNR (dB; higher is better)")
    ax.set_title("Same 256 validation images, scored at native 32×32 resolution", loc="left", pad=15)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.18)
    fig.savefig(output / "validation_scores.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4), layout="constrained")
    for kind in ("qvf", "classical"):
        rows = data["groups"][kind]
        curves = [[entry["validation"]["psnr_db"] for entry in row["history"] if entry["step"] >= 1000] for row in rows]
        steps = [entry["step"] for entry in rows[0]["history"] if entry["step"] >= 1000]
        curves = np.asarray(curves)
        mean = curves.mean(0)
        sd = curves.std(0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
        ax.plot(steps, mean, color=colours[kind], label="QVF head" if kind == "qvf" else "Classical head")
        ax.fill_between(steps, mean - sd, mean + sd, color=colours[kind], alpha=.15)
    ax.axhline(values[0], color=colours["reference"], linestyle="--", label="Published RAE reference")
    ax.set_xlabel("Optimizer updates (initial transient before 1,000 omitted)")
    ax.set_ylabel("Validation PSNR (dB)")
    ax.set_title(f"Validation history: mean ± sample SD across {seed_count} seeds", loc="left")
    ax.grid(alpha=.18)
    ax.legend(frameon=False)
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)

    for seed in data["protocol"]["seeds"]:
        for stage in ("final", "selected"):
            qpath, cpath = root / f"qvf_seed{seed}" / stage, root / f"classical_seed{seed}" / stage
            for group in ("validation", "train"):
                q = torch.load(qpath / f"{group}.pt", map_location="cpu", weights_only=True)
                c = torch.load(cpath / f"{group}.pt", map_location="cpu", weights_only=True)
                ref = torch.load(baseline / f"{group}.pt", map_location="cpu", weights_only=True)
                assert q["dataset_indices"] == c["dataset_indices"] == ref["dataset_indices"]
                rows = [("Original", q["originals"]), ("Published RAE", ref["reconstructions"]),
                        ("QVF, " + stage, q["reconstructions"]), ("Classical, " + stage, c["reconstructions"])]
                gallery(output / f"{group}_{stage}_seed{seed}_examples.png", rows, 16)
                if group == "validation":
                    gallery(output / f"{group}_{stage}_seed{seed}_all.png", rows, len(q["originals"]), columns=16, cell=64)

    text = ["Frozen DINO/RAE features with trained QVF and classical readouts",
            f"All {2 * seed_count} runs completed {data['protocol']['train_config']['steps']:,} updates. No encoder or pretrained decoder-body weights were trained.",
            "Same 4,744 training and 256 validation images. The official test set was not used.", "",
            f"PSNR below is mean +/- sample SD over {seed_count} seeds, in dB:"]
    for kind in ("qvf", "classical"):
        for stage in ("final", "selected"):
            train = summaries[kind][stage]["train"]["psnr_db"]
            val = summaries[kind][stage]["validation"]["psnr_db"]
            train_sd = "n/a" if train["sample_sd"] is None else f"{train['sample_sd']:.4f}"
            val_sd = "n/a" if val["sample_sd"] is None else f"{val['sample_sd']:.4f}"
            text.append(f"{kind:10} {stage:8} train {train['mean']:.4f} +/- {train_sd}; validation {val['mean']:.4f} +/- {val_sd}")
    text += [f"Published fixed RAE validation reference: {values[0]:.4f} dB.", "",
        "Selected checkpoints minimize validation pixel MSE among scheduled evaluations. Final checkpoints are the primary fixed-budget result.",
        "Readout package parameters: QVF 500,150; classical 500,099. Frozen pretrained feature models remain much larger.",
        "The QVF path uses the original amplitude map, five-qubit depth-five circuit and grouped RGB measurements, without learned postprocessing.",
        "The comparison isolates the trained readout packages on identical pretrained features; it does not isolate the circuit alone or establish quantum advantage.",
        "The RAE reference is zero-shot; new heads were fitted locally to pixel MSE. A score increase is not evidence of improved perceptual quality on every image.",
        "Static examples use the first 16 IDs in fixed split order. All 256 validation examples and all raw tensors are also saved."]
    (output / "summary.txt").write_text("\n".join(text) + "\n")
    # A single static chart for clients that require HTML to display figures.
    encoded = base64.b64encode((output / "validation_scores.png").read_bytes()).decode("ascii")
    html = '<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{margin:0;font:14px var(--font-sans,system-ui)}figure{margin:8px auto;max-width:820px}img{display:block;width:100%;height:320px;object-fit:contain}figcaption{margin-top:8px;color:var(--muted-foreground);line-height:1.5}@media(max-width:500px){img{height:200px}}</style></head><body><figure>'
    html += f'<img alt="Validation PSNR for the published RAE, QVF head and classical head" src="data:image/png;base64,{encoded}">'
    html += f'<figcaption>Error bars show sample SD across {seed_count} training seeds. The published RAE is one frozen reference checkpoint. Both new heads use the same frozen pretrained features.</figcaption></figure></body></html>'
    (output / "validation_scores.html").write_text(html)
    print("\n".join(text[:12]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    generate(parser.parse_args().study)
