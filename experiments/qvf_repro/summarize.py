"""Collect results/*.json, average over seeds, and compare with the QVF paper's Table 3.

Usage:
    python summarize.py            # prints the tables and writes results/summary.md and summary.json

Runs are grouped by setting (preset plus overrides such as the learning rate) and by model.
A QVF run with depth 0 is listed as "QVF without circuit": the amplitude encoding and the
Z read-out are kept, and no gate is applied in between.
"""

import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).parent

# QVF paper (arXiv 2508.10900), Table 3, image columns: (MSE x1e-3, its +-, PSNR in dB, its +-).
PAPER_TABLE3 = {
    ("qvf", "gaussian", "relu"): (1.02, 0.11, 29.8, 1.1),
    ("qvf", "identity", "relu"): (1.03, 0.09, 29.5, 1.2),
    ("mlp", None, "relu"): (2.17, 0.13, 26.57, 0.51),
    ("qvf", "gaussian", "sin"): (0.62, 0.05, 32.2, 0.3),
    ("qvf", "identity", "sin"): (0.72, 0.06, 31.4, 0.4),
    ("mlp", None, "sin"): (1.19, 0.08, 29.2, 0.4),
}
MODEL_ORDER = [("qvf", "gaussian"), ("qvf", "identity"), ("qvf_nocircuit", None), ("mlp", None)]
MODEL_LABEL = {("qvf", "gaussian"): "QVF, Gaussian init", ("qvf", "identity"): "QVF, identity init",
               ("qvf_nocircuit", None): "QVF without circuit", ("mlp", None): "Plain MLP"}


def pm(values, digits=2):
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{st.mean(values):.{digits}f} +- {st.stdev(values):.{digits}f}"


def load():
    """Map (setting, model, init, activation) to the list of runs (one per seed)."""
    groups = {}
    for f in sorted((HERE / "results").glob("*.json")):
        r = json.loads(f.read_text())
        if not isinstance(r, dict) or "psnr_per_image" not in r:
            continue  # equivalence reports and summaries
        tag = r["name"].split(f"seed{r['seed']}", 1)[1]
        model, init = r["model"], r["init"]
        if model == "qvf" and r["config"]["num_repetations"] == 0:
            model, init, tag = "qvf_nocircuit", None, tag.replace("_depth0", "")
        groups.setdefault((r["preset"] + tag, model, init, r["activation"]), []).append(r)
    return groups


def main():
    groups = load()
    summary, lines = [], []
    for setting in sorted({k[0] for k in groups}):
        keys = [k for k in groups if k[0] == setting]
        keys.sort(key=lambda k: (k[3], MODEL_ORDER.index(k[1:3])))
        cfg = groups[keys[0]][0]["config"]
        is_paper = groups[keys[0]][0]["preset"].startswith("paper")
        lines += [f"### {setting}", "",
                  f"{cfg['train_samples_use']} images, {cfg['train_epochs']} epochs, batch of {cfg['batch_size']} images, "
                  f"{(cfg['out_dim'] - 1).bit_length()} qubits, depth {groups[keys[0]][0]['config']['num_repetations'] or 'as preset'}, "
                  f"hidden width {cfg['hid_dim']}, learning rate {cfg['lr_model']}.", "",
                  "| Model | Activation | Seeds | Parameters | MSE x1e-3 | PSNR (dB) | Paper MSE x1e-3 | Paper PSNR (dB) | Minutes per run |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key in keys:
            runs = groups[key]
            ok = [r for r in runs if not math.isnan(r["mse_mean"])]
            diverged = len(runs) - len(ok)
            paper = PAPER_TABLE3.get((key[1], key[2], key[3])) if is_paper else None
            paper_cells = f"{paper[0]} +- {paper[1]} | {paper[2]} +- {paper[3]}" if paper else "n/a | n/a"
            seeds = f"{len(ok)}" + (f" ({diverged} diverged)" if diverged else "")
            row = {"setting": setting, "model": key[1], "init": key[2], "activation": key[3],
                   "seeds": len(ok), "diverged": diverged, "learning_rate": cfg["lr_model"],
                   "model_parameters": runs[0]["model_parameters"],
                   "paper_mse_e3": paper[0] if paper else None, "paper_psnr": paper[2] if paper else None}
            if ok:
                mse = [1e3 * r["mse_mean"] for r in ok]
                psnr = [r["psnr_mean"] for r in ok]
                minutes = st.mean(r["train_seconds"] for r in ok) / 60
                row.update({"mse_e3_mean": st.mean(mse), "mse_e3_std": st.stdev(mse) if len(ok) > 1 else None,
                            "psnr_mean": st.mean(psnr), "psnr_std": st.stdev(psnr) if len(ok) > 1 else None,
                            "psnr_of_mean_mse": st.mean(r["psnr_of_mean_mse"] for r in ok),
                            "psnr_per_seed": psnr, "minutes_per_run": minutes})
                cells = f"{pm(mse, 3)} | {pm(psnr)}"
            else:
                minutes = float("nan")
                cells = "diverged | diverged"
            lines.append(f"| {MODEL_LABEL[key[1:3]]} | {key[3]} | {seeds} | {runs[0]['model_parameters']} | {cells} | "
                         f"{paper_cells} | {minutes:.1f} |")
            summary.append(row)
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (HERE / "results" / "summary.md").write_text(
        "PSNR is the mean over images of the per-image PSNR, then mean and standard deviation over seeds.\n\n" + text + "\n")
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
