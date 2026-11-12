"""Predeclare a longer-training test followed by a fixed-qubit capacity test."""

import argparse
import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

from .reconstruction import DEFAULT_REFERENCE
from .train import HERE, REPO, sha256
from .vanilla_vae import vanilla_configs


def capacity_configs(seed, reference_run=DEFAULT_REFERENCE):
    base = dict(vanilla_configs(seed, reference_run))["larger"]
    base.vae_max_steps = 50000
    base.vae_epochs = math.ceil(base.vae_max_steps / (base.n_train // base.batch_size))
    variants = []
    for label, width in (("baseline50k", 128), ("wide51250k", 512)):
        cfg = copy.deepcopy(base)
        cfg.model.field_width = width
        cfg.validate()
        variants.append((label, cfg))
    return variants


def write_protocol(root, tag, seeds, reference_run=DEFAULT_REFERENCE):
    root, reference_run = Path(root).resolve(), Path(reference_run).resolve()
    path = root / f"{tag}_capacity_protocol.json"
    runs, prior = [], []
    for seed in seeds:
        previous = root / f"20261108_v3_vanilla_vae_larger_seed{seed}"
        prior.append({"seed": seed, "directory": str(previous),
                      "best_sha256": sha256(previous / "vae/checkpoints/best.pt"),
                      "last_sha256": sha256(previous / "vae/checkpoints/last.pt")})
        for variant, cfg in capacity_configs(seed, reference_run):
            runs.append({"variant": variant, "config": asdict(cfg),
                         "directory": str(root / f"{tag}_capacity_{variant}_seed{seed}")})
    protocol = {
        "kind": "qvf_vae_duration_and_capacity", "tag": tag, "seeds": seeds,
        "execution_order": ["Complete baseline50k for all seeds", "Then run wide51250k for all seeds"],
        "runs": runs, "previous_runs": prior,
        "reference_run": str(reference_run), "reference_split_sha256": sha256(reference_run / "split.json"),
        "decoder_source_sha256": sha256(REPO / "experiments/qvf_repro/qvf_fast.py"),
        "baseline_change": "Only total optimizer steps: 7680 to 50000. Same architecture, learning rate, VAE objective, data and random seeds.",
        "capacity_change": "Only classical amplitude-network hidden width: 128 to 512. Both configurations use 50000 updates.",
        "fixed_quantum_settings": {"qubits": 5, "depth": 5, "initialization": "Gaussian std 0.1", "readout": "original grouped Z; no affine RGB map"},
        "initialization": "Fresh runs. Replaying the same seed avoids an approximate resume from the middle of an epoch. Baseline history through step 7680 must reproduce the previous run.",
        "budget": {"steps": 50000, "batch_size": 32, "pixels_per_image": 256,
                   "image_observations": 1600000, "coordinate_queries": 409600000,
                   "training_images": 4744, "validation_images": 256},
        "optimizer": "Existing Adam, constant learning rate 0.001, existing gradient clipping. No new scheduler.",
        "selection": "Report both final 50000-step checkpoint and checkpoint selected by validation sampled-code MSE + final KL weight * KL.",
        "precision": "Existing deterministic runtime, TF32 cuDNN convolutions allowed; TF32 matmul disabled.",
        "resource_scope": "GPU 0 only, existing quantai environment, up to three training processes at a time",
        "diffusion_training": False, "official_test_set_used": False,
        "convolutional_extension": "Research/design only in this study; no CNN substitution is included in the width comparison.",
    }
    with path.open("x") as handle:
        json.dump(protocol, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    print(write_protocol(args.root, args.tag, args.seeds, args.reference_run))


if __name__ == "__main__":
    main()
