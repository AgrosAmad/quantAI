"""Static edge/colour tradeoff plots from independently verified V10 results."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .train import write_json

NAMES = ("attention_pixel", "attention_edge", "staged_all_pixel", "staged_all_edge")
LABELS = {"attention_pixel": "Attention, pixel loss", "attention_edge": "Attention, + edge loss",
          "staged_all_pixel": "Revised stages, pixel loss", "staged_all_edge": "Revised stages, + edge loss"}


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else 0,
            "seeds": values}


def report(root):
    root = Path(root).resolve()
    directory = root / "verified"
    protocol = json.loads((root / "protocol.json").read_text())
    cfg = protocol["runs"][0]["config"]
    completion = protocol["completion"]
    edge_weight = next(r["config"]["edge_weight"] for r in protocol["runs"] if r["config"]["edge_weight"] > 0)
    data = json.loads((directory / "results.json").read_text())["runs"]
    metrics = {"train_psnr": ("fit", "psnr_db"), "train_edge": ("fit", "gradient_mse"),
               "train_mse": ("fit", "mse"), "completion_psnr": ("completion_unobserved", "psnr_db"),
               "completion_edge": ("completion_unobserved_edges", "gradient_mse"),
               "completion_mse": ("completion_unobserved", "mse")}
    summary, differences = {}, {}
    for name in NAMES:
        subset = sorted([r for r in data if r["name"] == name], key=lambda r: r["seed"])
        assert len(subset) == len(protocol["seeds"])
        summary[name] = {label: stats([r[group][key] for r in subset]) for label, (group, key) in metrics.items()}
        summary[name]["shared_parameters"] = subset[0]["shared_parameters"]
        summary[name]["step2000_edge"] = stats([r["step2000"]["gradient_mse"] for r in subset])
    for a, b in (("attention_edge", "attention_pixel"), ("staged_all_edge", "staged_all_pixel"),
                 ("staged_all_pixel", "attention_pixel"), ("staged_all_edge", "attention_edge"),
                 ("staged_all_edge", "attention_pixel")):
        differences[f"{a} minus {b}"] = {label: stats([x - y for x, y in
            zip(summary[a][label]["seeds"], summary[b][label]["seeds"])]) for label in metrics}
        for label in ("train_edge", "completion_edge"):
            differences[f"{a} minus {b}"][label + "_reduction_percent"] = 100 * (1 - summary[a][label]["mean"] / summary[b][label]["mean"])
    write_json(directory / "summary.json", {"models": summary, "paired_differences": differences})
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *[f"{m}_{s}" for m in metrics for s in ("mean", "sample_sd")]])
        for name in NAMES:
            writer.writerow([name, *[summary[name][m][s] for m in metrics for s in ("mean", "sample_sd")]])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    colours = {"pixel": "#2166ac", "edge": "#b35806"}
    plot_specs = (("train_edge", "Training: edge error", 1000), ("train_psnr", "Training: colour reconstruction", 1),
                  ("completion_edge", "New images: unobserved edges", 1000),
                  ("completion_psnr", "New images: unobserved pixels", 1))
    for ax, (metric, title, factor) in zip(axes.flatten(), plot_specs):
        all_values = []
        for objective, shift in (("pixel", -.12), ("edge", .12)):
            for x, architecture in enumerate(("attention", "staged_all")):
                values = summary[f"{architecture}_{objective}"][metric]
                seeds = np.array(values["seeds"]) * factor
                all_values.extend(seeds)
                mean = values["mean"] * factor
                label = "Pixel loss" if objective == "pixel" else f"+ edge loss ({edge_weight:g})"
                ax.plot([x + shift] * len(seeds), seeds, "o", color=colours[objective], alpha=.45, markersize=4)
                ax.plot(x + shift, mean, "D", color=colours[objective], markersize=6, label=label if x == 0 else None)
                ax.annotate(f"{mean:.3f}" if factor == 1000 else f"{mean:.2f}", (x + shift, max(seeds)),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=colours[objective])
        spread = max(max(all_values) - min(all_values), .1)
        ax.set_ylim(min(all_values) - .3 * spread, max(all_values) + .6 * spread)
        ax.set_xlim(-.45, 1.45)
        ax.set_xticks([0, 1], ["Attention", "Revised stages"])
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("Error × 1,000; lower is better" if factor == 1000 else "PSNR (dB); higher is better")
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=10)
    fig.suptitle(f"Five qubits · 336 values per image · {cfg['steps']:,} updates · {len(protocol['seeds'])} seeds", fontsize=13)
    fig.savefig(directory / "comparison.png", dpi=180)
    fig.savefig(directory / "comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name in NAMES:
        runs = [r for r in data if r["name"] == name]
        points = [r["step"] for r in runs[0]["curve"] if r["step"] >= 2000]
        for ax, metric, factor in ((axes[0], "gradient_mse", 1000), (axes[1], "psnr_db", 1)):
            values = np.array([[r["fit"][metric] * factor for r in run["curve"] if r["step"] >= 2000] for run in runs])
            ax.plot(points, values.mean(0), marker=".", label=LABELS[name])
    for ax in axes:
        ax.set_xlabel("Training updates")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Edge error × 1,000")
    axes[1].set_ylabel("PSNR (dB)")
    fig.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(fig)
    lines = ["V10 edge fidelity study: final checkpoints, independently verified.",
             f"Five qubits, 336 fitted values/image; {cfg['n_train']} images; {cfg['steps']} updates; {len(protocol['seeds'])} paired seeds.",
             f"Completion: {len(completion['dataset_indices'])} excluded images, {completion['steps']} pixel-only code updates on {len(completion['observed_pixels'])} observed pixels; evaluate {len(completion['unobserved_pixels'])} withheld pixels and pairs with both endpoints withheld.",
             "Mean +/- sample SD across seeds, not a confidence interval.", ""]
    for name in NAMES:
        v = summary[name]
        lines.append(f"{name}: training edge {v['train_edge']['mean']:.7f} +/- {v['train_edge']['sample_sd']:.7f}; PSNR {v['train_psnr']['mean']:.3f} +/- {v['train_psnr']['sample_sd']:.3f}; completion edge {v['completion_edge']['mean']:.7f} +/- {v['completion_edge']['sample_sd']:.7f}; PSNR {v['completion_psnr']['mean']:.3f} +/- {v['completion_psnr']['sample_sd']:.3f}")
    lines.append("")
    for pair, values in differences.items():
        lines.append(f"{pair}: training edge reduction {values['train_edge_reduction_percent']:+.2f}%; PSNR {values['train_psnr']['mean']:+.3f} dB; completion edge reduction {values['completion_edge_reduction_percent']:+.2f}%; PSNR {values['completion_psnr']['mean']:+.3f} dB")
    lines.extend(["", "No encoder/diffusion or quantum advantage established. One loss weight; finite budget; added rotations cost gates. V9 scores are not directly matched controls because sampling and completion budgets differ."])
    (directory / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)
