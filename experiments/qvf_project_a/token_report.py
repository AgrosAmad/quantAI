"""Static plots and a compact numerical report from verified token-QVF results."""

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


LABELS = {"vanilla64": "Vanilla QVF\n64 values", "global336": "Global QVF\n336 values",
          "bilinear": "Spatial grid\n336 values", "attention": "Token attention\n336 values",
          "attention_staged": "Tokens + stages\n336 values"}


def describe(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else 0,
            "seeds": values}


def report(root):
    root = Path(root).resolve()
    directory = root / "verified"
    data = json.loads((directory / "results.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    budget = protocol["runs"][0]["config"]
    completion = protocol["completion"]
    observed_count = len(completion["observed_pixels"])
    unseen_count = len(completion["unobserved_pixels"])
    records = data["runs"]
    summary, differences = {}, {}
    for q in (5, 7):
        for name in LABELS:
            key = f"q{q}_{name}"
            subset = sorted([r for r in records if r["name"] == key], key=lambda r: r["seed"])
            assert len(subset) == len(protocol["seeds"])
            summary[key] = {"shared_parameters": subset[0]["shared_parameters"],
                "per_image_code_entries": subset[0]["code_entries"],
                "training_psnr_db": describe([r["fit"]["psnr_db"] for r in subset]),
                "training_mse": describe([r["fit"]["mse"] for r in subset]),
                "training_gradient_mse": describe([r["fit"]["gradient_mse"] for r in subset]),
                "completion_psnr_db": describe([r["completion_unobserved"]["psnr_db"] for r in subset]),
                "completion_mse": describe([r["completion_unobserved"]["mse"] for r in subset])}
    pairs = []
    for q in (5, 7):
        pairs.extend((f"q{q}_{a}", f"q{q}_{b}") for a, b in
                     (("attention", "vanilla64"), ("attention", "global336"),
                      ("attention", "bilinear"), ("attention_staged", "attention")))
    pairs.extend((f"q7_{name}", f"q5_{name}") for name in LABELS)
    for a, b in pairs:
        differences[f"{a} minus {b}"] = {metric: describe([x - y for x, y in
            zip(summary[a][metric]["seeds"], summary[b][metric]["seeds"])])
            for metric in ("training_psnr_db", "completion_psnr_db")}
    write_json(directory / "summary.json", {"models": summary, "paired_differences": differences})
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "shared_parameters", "per_image_code_entries", "train_psnr_mean",
                         "train_psnr_sample_sd", "completion_unobserved_psnr_mean", "completion_psnr_sample_sd"])
        for name, values in summary.items():
            writer.writerow([name, values["shared_parameters"], values["per_image_code_entries"],
                values["training_psnr_db"]["mean"], values["training_psnr_db"]["sample_sd"],
                values["completion_psnr_db"]["mean"], values["completion_psnr_db"]["sample_sd"]])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colours = {5: "#2166ac", 7: "#b35806"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.8), layout="constrained")
    for ax, metric, title in zip(axes,
            ("training_psnr_db", "completion_psnr_db"),
            (f"Fitting the {budget['n_train']:,} training images",
             f"New images: score only the {unseen_count} unobserved pixels")):
        x = np.arange(len(LABELS))
        all_values = []
        for q, shift in ((5, -.13), (7, .13)):
            means = [summary[f"q{q}_{name}"][metric]["mean"] for name in LABELS]
            ax.plot(x + shift, means, "D", color=colours[q], label=f"{q} qubits: mean", markersize=6)
            for i, name in enumerate(LABELS):
                seeds = summary[f"q{q}_{name}"][metric]["seeds"]
                all_values.extend(seeds)
                ax.plot([i + shift] * len(seeds), seeds, "o", color=colours[q], alpha=.5, markersize=4)
                ax.annotate(f"{means[i]:.2f}", (i + shift, max(seeds)), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=8, color=colours[q])
        ax.set_ylim(min(all_values) - .8, max(all_values) + 1)
        ax.set_xticks(x, LABELS.values())
        ax.set_ylabel("PSNR (dB); higher is better")
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(axis="y", alpha=.2)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"Token-QVF pilot: fixed {budget['steps']:,}-update budget, {len(protocol['seeds'])} paired seeds", fontsize=13)
    fig.savefig(directory / "comparison.png", dpi=180)
    fig.savefig(directory / "comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for q, ax in zip((5, 7), axes):
        for name in LABELS:
            runs = [r for r in records if r["name"] == f"q{q}_{name}"]
            steps = [r["step"] for r in runs[0]["curve"] if r["step"] >= 500]
            curves = [[r["fit"]["psnr_db"] for r in run["curve"] if r["step"] >= 500] for run in runs]
            ax.plot(steps, np.mean(curves, axis=0), marker=".", label=LABELS[name].replace("\n", " "))
        ax.set_title(f"{q} qubits, mean training PSNR")
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(fig)
    lines = ["V9 token-QVF pilot: verified final-checkpoint results.",
             f"{budget['n_train']} fitted images, {budget['steps']} updates, {len(protocol['seeds'])} seeds; exact simulation.",
             f"Completion: {len(completion['dataset_indices'])} excluded images, frozen decoder, {completion['steps']} code updates on {observed_count} pixels; score {unseen_count} unobserved pixels.",
             "Values below: mean +/- sample SD across seeds, not a confidence interval.", ""]
    for name, values in summary.items():
        fit, transfer = values["training_psnr_db"], values["completion_psnr_db"]
        lines.append(f"{name}: train {fit['mean']:.3f} +/- {fit['sample_sd']:.3f} dB; completion {transfer['mean']:.3f} +/- {transfer['sample_sd']:.3f} dB")
    lines.extend(["", "Controlled same-qubit paired differences:"])
    for pair, values in differences.items():
        lines.append(f"{pair}: train {values['training_psnr_db']['mean']:+.3f} dB; completion {values['completion_psnr_db']['mean']:+.3f} dB")
    lines.extend(["", "Limits: small pilot and incomplete optimization; no encoder/diffusion, hardware or quantum-advantage claim.",
                  "Seven qubits also changes amplitude-MLP output width, gate count and prescribed RGB normalization."])
    (directory / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)
