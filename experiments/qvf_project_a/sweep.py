"""Named, bounded experiments; fresh tags prevent overwriting completed runs.

Run from the repository root with the existing quantai Python environment:
  CUDA_VISIBLE_DEVICES=0 python -m experiments.qvf_project_a.sweep pilot --tag first
"""

import argparse
import copy
import json
import re
from dataclasses import asdict
from pathlib import Path

from .models import ModelConfig
from .reconstruction import DEFAULT_REFERENCE, run_reconstruction, study_configs
from .train import DEFAULT_DATA, HERE, RunConfig, run_pipeline
from .vanilla_vae import run_vanilla_vae, vanilla_configs
from .capacity_study import capacity_configs
from .direct_qvf import direct_configs, run_direct_qvf
from .spatial_study import spatial_configs, run_spatial


def group_configs(group, seed):
    if group == "smoke":
        return [RunConfig(model=ModelConfig(encoder_width=8, field_width=32, circuit_depth=1), seed=seed,
                          n_train=32, n_val=16, batch_size=8, pixels_per_image=64, vae_epochs=2,
                          kl_warmup_epochs=1, vae_eval_every=1, diffusion_steps=20, diffusion_batch_size=16,
                          diffusion_width=32, diffusion_layers=2, diffusion_timesteps=20,
                          diffusion_eval_every=10, sample_count=8)]
    if group == "pilot":
        return [RunConfig(seed=seed)]
    if group == "controls":
        configs = []
        for decoder, mode in [("qvf", "vae"), ("no_circuit", "vae"), ("mlp", "vae"), ("qvf", "ae")]:
            cfg = RunConfig(seed=seed)
            cfg.model.decoder, cfg.model.mode = decoder, mode
            configs.append(cfg)
        return configs
    raise ValueError(f"Unknown experiment group {group}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=["smoke", "pilot", "controls", "reconstruction", "vanilla_vae", "capacity", "direct_qvf", "spatial"])
    parser.add_argument("--tag", required=True, help="Fresh experiment label; existing runs are never overwritten.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--results-root", type=str, default=str(HERE / "results"))
    parser.add_argument("--vae-epochs", type=int, help="Override the named group's VAE budget.")
    parser.add_argument("--diffusion-steps", type=int, help="Override the named group's diffusion budget.")
    parser.add_argument("--select", nargs="+",
                        help="Run only these members of a group; useful when a baseline already exists.")
    parser.add_argument("--reference-run", default=str(DEFAULT_REFERENCE),
                        help="Reference data split and configuration for the reconstruction study.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        parser.error("Use letters, digits, dots, underscores or hyphens in --tag.")
    runs = []
    for seed in args.seeds:
        named = (spatial_configs(seed, args.reference_run) if args.group == "spatial" else
                 direct_configs(seed, args.reference_run) if args.group == "direct_qvf" else
                 capacity_configs(seed, args.reference_run) if args.group == "capacity" else
                 vanilla_configs(seed, args.reference_run) if args.group == "vanilla_vae" else
                 study_configs(seed, args.reference_run) if args.group == "reconstruction" else
                 [(f"{cfg.model.decoder}_{cfg.model.mode}", cfg) for cfg in group_configs(args.group, seed)])
        if args.select and set(args.select) - {name for name, _ in named}:
            parser.error("--select contains a variant outside the chosen group.")
        if args.group in {"reconstruction", "vanilla_vae", "capacity", "direct_qvf", "spatial"} and (args.vae_epochs is not None or args.diffusion_steps is not None):
            parser.error("The reconstruction study uses fixed step budgets; use its named variants.")
        for label, base in named:
            if args.select and label not in args.select:
                continue
            cfg = copy.deepcopy(base)
            if args.vae_epochs is not None:
                cfg.vae_epochs = args.vae_epochs
            if args.diffusion_steps is not None:
                cfg.diffusion_steps = args.diffusion_steps
            cfg.validate()
            name = f"{args.tag}_{args.group}_{label}_seed{seed}"
            directory = Path(args.results_root) / name
            if not args.dry_run and directory.exists():
                parser.error(f"Run already exists: {directory}. Choose a fresh --tag.")
            runs.append((label, cfg, directory))
    if not runs:
        parser.error("No experiments match --select in the chosen group.")
    if len({directory for _, _, directory in runs}) != len(runs):
        parser.error("Duplicate seeds would create duplicate runs.")
    for label, cfg, directory in runs:
        if args.dry_run:
            print(json.dumps({"directory": str(directory), "config": asdict(cfg)}, indent=2))
        else:
            print(f"Starting {directory.name} on {args.device}", flush=True)
            if args.group == "spatial":
                run_spatial(cfg, directory, args.data_root, args.device, args.reference_run, label)
            elif args.group == "direct_qvf":
                run_direct_qvf(cfg, directory, args.data_root, args.device, args.reference_run, label)
            elif args.group in {"vanilla_vae", "capacity"}:
                run_vanilla_vae(cfg, directory, args.data_root, args.device, args.reference_run, label)
            elif args.group == "reconstruction":
                run_reconstruction(cfg, directory, args.data_root, args.device, args.reference_run, label)
            else:
                run_pipeline(cfg, directory, args.data_root, args.device)


if __name__ == "__main__":
    main()
