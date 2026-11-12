"""Static summaries of independently verified Project Z experiments."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.qvf_project_a.train import write_json


LABELS = {"attention_amp": "Original quantum head", "latent_conv": "3x3 latent-grid convolution",
    "latent_pointwise": "1x1 capacity control", "attention_local_shared": "Local shared quantum filter",
    "attention_local_untied": "Local untied quantum filter", "conv_local_shared": "Latent conv + shared quantum",
    "conv_local_untied": "Latent conv + untied quantum"}


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "seeds": values}


def report(root):
    root = Path(root).resolve()
    directory = root / "verified"
    protocol = json.loads((root / "protocol.json").read_text())
    data = json.loads((directory / "results.json").read_text())["runs"]
    NAMES = list(dict.fromkeys(r["name"] for r in data))
    metrics = {"train_psnr": ("fit", "psnr_db"), "train_edge": ("fit", "gradient_mse"),
        "train_mse": ("fit", "mse"), "completion_psnr": ("completion_unobserved", "psnr_db"),
        "completion_edge": ("completion_unobserved_edges", "gradient_mse"),
        "completion_mse": ("completion_unobserved", "mse")}
    summary, differences = {}, {}
    for name in NAMES:
        subset = sorted([r for r in data if r["name"] == name], key=lambda r: r["seed"])
        assert [r["seed"] for r in subset] == sorted(protocol["seeds"])
        summary[name] = {label: stats([r[group][key] for r in subset]) for label, (group, key) in metrics.items()}
        summary[name]["resources"] = subset[0]["resources"]
        summary[name]["elapsed_seconds"] = stats([r["elapsed_seconds"] for r in subset])
    comparisons = [(a, b) for i, a in enumerate(NAMES) for b in NAMES[:i]]
    for a, b in comparisons:
        differences[f"{a} minus {b}"] = {metric: stats([x - y for x, y in
            zip(summary[a][metric]["seeds"], summary[b][metric]["seeds"])]) for metric in metrics}
        for metric in ("train_edge", "completion_edge"):
            differences[f"{a} minus {b}"][metric + "_reduction_percent"] = 100 * (1 - summary[a][metric]["mean"] / summary[b][metric]["mean"])
    write_json(directory / "summary.json", {"models": summary, "paired_differences": differences})
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "shared_parameters", *[f"{m}_{s}" for m in metrics for s in ("mean", "sample_sd")]])
        for name in NAMES:
            writer.writerow([name, summary[name]["resources"]["shared_parameters"], *[summary[name][m][s] for m in metrics for s in ("mean", "sample_sd")]])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colours = ["#555555", "#999999", "#2166ac", "#4393c3", "#053061", "#878787", "#9970ab", "#b35806", "#762a83"]
    for kind, titles, metric_keys, factor, filename in (
        ("Edge error", ("Training images", "New images: withheld edges"), ("train_edge", "completion_edge"), 1000, "comparison"),
        ("Pixel reconstruction", ("Training images", "New images: withheld pixels"), ("train_psnr", "completion_psnr"), 1, "pixel_reconstruction")):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 6), sharey=True, layout="constrained")
        for ax, title, metric in zip(axes, titles, metric_keys):
            for y, (name, colour) in enumerate(zip(NAMES, colours)):
                values = summary[name][metric]
                mean, samples = values["mean"] * factor, np.array(values["seeds"]) * factor
                ax.scatter(samples, np.full(len(samples), y), s=22, alpha=.4, color=colour)
                ax.scatter(mean, y, s=55, marker="D", color=colour, zorder=3)
            ax.set_yticks(range(len(NAMES)), [LABELS[n] for n in NAMES])
            ax.set_title(title)
            ax.set_xlabel("Edge MSE × 1,000; lower is better" if factor == 1000 else "PSNR (dB); higher is better")
            ax.grid(axis="x", alpha=.2)
        axes[0].invert_yaxis()
        cfg = protocol["runs"][0]["config"]
        fig.suptitle(f"{kind} · {cfg['n_train']:,} images · {cfg['steps']:,} updates · {len(protocol['seeds'])} seeds")
        fig.savefig(directory / f"{filename}.png", dpi=180)
        fig.savefig(directory / f"{filename}.pdf")
        plt.close(fig)

    for stage, names in (("selected", NAMES),):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for name in names:
            runs = [r for r in data if r["name"] == name]
            points = [row["step"] for row in runs[0]["curve"] if row["step"] >= 2000]
            for ax, metric, factor in ((axes[0], "gradient_mse", 1000), (axes[1], "psnr_db", 1)):
                values = np.array([[row["fit"][metric] * factor for row in run["curve"] if row["step"] >= 2000] for run in runs])
                ax.plot(points, values.mean(0), marker=".", label=LABELS[name])
        for ax in axes:
            ax.set_xlabel("Training updates")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Training edge MSE × 1,000")
        axes[1].set_ylabel("Training PSNR (dB)")
        fig.savefig(directory / f"{stage}_learning_curves.png", dpi=160)
        plt.close(fig)
    lines = ["Project Z architecture study: final checkpoints, independently verified.",
        "1024 training images, 10000 updates, three paired seeds, 336 fitted values/image, pixel + 0.5 edge objective.",
        "Five qubits, continuous coordinate-to-RGB field; all outputs use quantum measurements.",
        "Completion: 32 excluded images, 1000 code updates on 256 observed pixels; score 768 withheld pixels and edges whose endpoints are both withheld.",
        "Mean +/- sample SD across seeds; these are not confidence intervals.", ""]
    for name in NAMES:
        v = summary[name]
        lines.append(f"{name} ({v['resources']['shared_parameters']} shared parameters): train PSNR {v['train_psnr']['mean']:.3f} +/- {v['train_psnr']['sample_sd']:.3f}, edge {v['train_edge']['mean']:.7f} +/- {v['train_edge']['sample_sd']:.7f}; completion PSNR {v['completion_psnr']['mean']:.3f} +/- {v['completion_psnr']['sample_sd']:.3f}, edge {v['completion_edge']['mean']:.7f} +/- {v['completion_edge']['sample_sd']:.7f}")
    lines.append("")
    for pair, values in differences.items():
        lines.append(f"{pair}: train edge reduction {values['train_edge_reduction_percent']:+.2f}%, PSNR {values['train_psnr']['mean']:+.3f} dB; completion edge reduction {values['completion_edge_reduction_percent']:+.2f}%, PSNR {values['completion_psnr']['mean']:+.3f} dB")
    lines.extend(["", "Quantum gate counts exclude preparation unless explicitly stated. Ideal simulator training; separate finite-shot/noise inference audit. No encoder/diffusion, hardware, official test use or quantum advantage. Fixed-budget results do not establish convergence. Qubit locality is not image locality."])
    (directory / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)
