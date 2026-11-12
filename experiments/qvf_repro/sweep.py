"""Run a group of training runs, a few at a time, skipping runs that already have a result file.

Usage:
    python sweep.py demo          # released demo config: QVF (both initialisations) and the MLP, seeds 0 to 2
    python sweep.py table3        # the paper's 50-image collection table, both readings of the settings
    python sweep.py nocircuit     # QVF with the circuit removed (depth 0), paper settings
    python sweep.py demo_lr       # learning-rate sweep at the demo settings, all models
    python sweep.py more_steps    # paper settings with 5 optimiser steps per epoch instead of 1
    python sweep.py paper_lr      # paper settings with learning rates 3e-3 and 1e-2 instead of 1e-3
    python sweep.py paper_seeds   # seeds 3 to 7 for the ReLU models at the paper settings, all three learning rates
    python sweep.py table3 --jobs 6 --device cuda:0
"""

import argparse
import itertools
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from train import PRESETS, run_name

HERE = Path(__file__).parent
SEEDS = (0, 1, 2)
MODELS = (("qvf", "identity"), ("qvf", "gaussian"), ("mlp", "identity"))


def run(preset, model, init, act, seed, tag="", extra=(), images=None):
    return {"preset": preset, "model": model, "init": init, "act": act, "seed": seed,
            "tag": tag, "extra": list(extra), "images": images}


def group_runs(group):
    if group == "vanilla_single":
        return [run("paper", "qvf", "gaussian", "relu", 0, f"_img{i}",
                    ["--first-image", str(i), "--batch-size", "1"], images=1) for i in range(3)]
    if group in {"vanilla_collection", "vanilla_paper_text"}:
        preset = "paper" if group == "vanilla_collection" else "paper_text"
        return [run(preset, "qvf", "gaussian", "relu", s, "_bs10", ["--batch-size", "10"]) for s in SEEDS]
    if group == "demo":
        return [run("demo", m, i, "relu", s) for (m, i), s in itertools.product(MODELS, SEEDS)]
    if group == "table3":
        return [run(p, m, i, a, s) for p, (m, i), a, s in
                itertools.product(["paper", "paper_text"], MODELS, ["relu", "sin"], SEEDS)]
    if group == "nocircuit":
        # Depth 0: amplitude encoding and Z read-out with no gates in between. The initialisation is irrelevant.
        return [run(p, "qvf", "gaussian", a, s, "_depth0", ["--depth", "0"]) for p, a, s in
                itertools.product(["paper", "paper_text", "demo"], ["relu"], SEEDS)]
    if group == "demo_lr":
        runs = []
        for lr, (m, i), s in itertools.product(["0.0003", "0.001", "0.003", "0.03"], MODELS, SEEDS):
            runs.append(run("demo", m, i, "relu", s, f"_lr{lr}", ["--set", f"lr_model={lr}", f"lr_latent={lr}"]))
        return runs
    if group == "more_steps":
        runs = [run("paper", m, i, "relu", s, "_bs10", ["--batch-size", "10"])
                for (m, i), s in itertools.product(MODELS, SEEDS)]
        runs += [run("paper", "qvf", "gaussian", "relu", s, "_bs10_depth0", ["--batch-size", "10", "--depth", "0"])
                 for s in SEEDS]
        return runs
    if group == "paper_lr":
        # The paper states 1e-3; the released config uses 1e-2. Same 50 images and 5,000 epochs.
        runs = []
        for lr in ["0.003", "0.01"]:
            extra = ["--set", f"lr_model={lr}", f"lr_latent={lr}"]
            runs += [run("paper", m, i, "relu", s, f"_lr{lr}", extra) for (m, i), s in itertools.product(MODELS, SEEDS)]
            runs += [run("paper", "qvf", "gaussian", "relu", s, f"_lr{lr}_depth0", extra + ["--depth", "0"]) for s in SEEDS]
        return runs
    if group == "paper_seeds":
        # Five more seeds (3 to 7) for the ReLU comparison at the paper settings, at all three learning rates.
        runs = []
        for lr in ["0.001", "0.003", "0.01"]:
            tag = "" if lr == "0.001" else f"_lr{lr}"
            extra = [] if lr == "0.001" else ["--set", f"lr_model={lr}", f"lr_latent={lr}"]
            for s in range(3, 8):
                runs += [run("paper", m, i, "relu", s, tag, extra) for m, i in MODELS]
                runs.append(run("paper", "qvf", "gaussian", "relu", s, tag + "_depth0", extra + ["--depth", "0"]))
        return runs
    raise SystemExit(f"unknown group {group}")


def launch(r, device, output_dir=None, save_checkpoint=False):
    output_dir = Path(output_dir) if output_dir is not None else HERE / "results"
    n_img = r.get("images") or PRESETS[r["preset"]]["train_samples_use"]
    name = run_name(r["preset"], r["model"], r["init"], r["act"], n_img, r["seed"], r["tag"])
    if (output_dir / f"{name}.json").exists():
        return name, "skipped (result exists)"
    log = output_dir / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "train.py"), "--preset", r["preset"], "--model", r["model"],
           "--init", r["init"], "--act", r["act"], "--seed", str(r["seed"]), "--device", device]
    if r["tag"]:
        cmd += ["--tag", r["tag"]]
    if r.get("images") is not None:
        cmd += ["--images", str(r["images"])]
    cmd += ["--output-dir", str(output_dir.resolve())]
    if save_checkpoint:
        cmd += ["--save-checkpoint"]
    cmd += r["extra"]
    t0 = time.perf_counter()
    with open(log, "w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE).returncode
    return name, f"{'ok' if code == 0 else f'FAILED ({code})'} in {time.perf_counter() - t0:.0f} s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=["demo", "table3", "nocircuit", "demo_lr", "more_steps", "paper_lr", "paper_seeds",
                                      "vanilla_single", "vanilla_collection", "vanilla_paper_text"])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", type=Path, default=HERE / "results")
    ap.add_argument("--save-checkpoint", action="store_true")
    args = ap.parse_args()
    runs = group_runs(args.group)
    print(f"{len(runs)} runs, {args.jobs} at a time", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        for name, status in pool.map(lambda r: launch(r, args.device, args.output_dir, args.save_checkpoint), runs):
            print(f"  {name}: {status}", flush=True)
