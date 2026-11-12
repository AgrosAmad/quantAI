"""VAE-only controls with fixed validation IDs and optimizer-step schedules."""

import copy
import json
import math
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch

from .data import load_data
from .models import ModelConfig
from .train import (DEFAULT_DATA, HERE, RunConfig, configure_runtime, evaluate_vae,
                    load_vae, sha256, snapshot_sources, train_vae, write_json)

DEFAULT_REFERENCE = HERE / "results" / "20261108_v1_pilot_qvf_vae_seed0"


def reference_config(reference_run):
    raw = json.loads((Path(reference_run) / "config.json").read_text())
    names = {item.name for item in fields(RunConfig)} - {"model"}
    return RunConfig(model=ModelConfig.from_saved(raw["model"]), **{key: raw[key] for key in names if key in raw})


def study_configs(seed, reference_run=DEFAULT_REFERENCE):
    base = reference_config(reference_run)
    base.seed = seed
    original_batches = math.ceil(base.n_train / base.batch_size)
    base_steps = base.vae_max_steps or base.vae_epochs * original_batches
    base.vae_max_steps = base_steps
    base.vae_eval_every_steps = base.vae_eval_every * original_batches
    base.kl_warmup_steps = base.kl_warmup_epochs * original_batches
    base.vae_drop_last = True
    available = 50000 if base.class_idx == -1 else 5000
    variants = []
    for label in ("baseline", "longer", "latent128", "wider", "more_data", "more_data_longer"):
        cfg = copy.deepcopy(base)
        if label == "latent128":
            cfg.model.latent_dim *= 2
        if label == "wider":
            cfg.model.field_width *= 2
        if label in {"longer", "more_data_longer"}:
            cfg.vae_max_steps = 3 * base_steps
        if label in {"more_data", "more_data_longer"}:
            cfg.n_train = available - base.n_val
        cfg.vae_epochs = math.ceil(cfg.vae_max_steps / (cfg.n_train // cfg.batch_size))
        cfg.validate()
        variants.append((label, cfg))
    return variants


def run_reconstruction(cfg, run_dir, data_root=DEFAULT_DATA, device_name="cuda:0",
                       reference_run=DEFAULT_REFERENCE, variant="baseline"):
    cfg.validate()
    run_dir = Path(run_dir).resolve()
    reference_run = Path(reference_run).resolve()
    reference_split_path = reference_run / "split.json"
    reference_split = json.loads(reference_split_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        device = configure_runtime(device_name, cfg.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        write_json(run_dir / "config.json", {**asdict(cfg), "data_root": str(Path(data_root).resolve()),
                                              "device": device_name, "variant": variant,
                                              "reference_run": str(reference_run)})
        snapshot_sources(run_dir)
        images, indices = load_data(data_root, cfg.class_idx, cfg.n_train, cfg.n_val,
                                    cfg.split_seed, reference_split["indices"])
        write_json(run_dir / "split.json", {"dataset": "CIFAR10", "source": "official training set",
                                            "official_test_set_used": False, "indices": indices,
                                            "reference_sha256": sha256(reference_split_path)})
        write_json(run_dir / "environment.json", {"torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})
        path, metrics = train_vae(cfg, images, run_dir, device)
        model, _ = load_vae(path, device)
        _, mean_images, sampled_images = evaluate_vae(model, images["validation"], cfg.batch_size,
                                                     device, cfg.seed + 1001)
        target = images["validation"]
        mean_mse = (mean_images - target).square().flatten(1).mean(1)
        sampled_mse = (sampled_images - target).square().flatten(1).mean(1)
        # Error in adjacent-pixel differences, a simple diagnostic of edge detail.
        error = mean_images - target
        gradient_mse = 0.5 * ((error[:, :, 1:] - error[:, :, :-1]).square().flatten(1).mean(1) +
                              (error[:, :, :, 1:] - error[:, :, :, :-1]).square().flatten(1).mean(1))
        write_json(run_dir / "validation_details.json", {
            "dataset_indices": indices["validation"], "mean_code_mse": mean_mse.tolist(),
            "mean_code_psnr_db": (-10 * mean_mse.clamp_min(1e-12).log10()).tolist(),
            "sampled_code_mse": sampled_mse.tolist(), "adjacent_pixel_gradient_mse": gradient_mse.tolist(),
        })
        result = {"status": "complete", "kind": "reconstruction_study", "variant": variant,
                  "elapsed_seconds": time.perf_counter() - started, "vae": metrics,
                  "validation_gradient_mse": gradient_mse.mean().item(),
                  "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  "scope": "representation training only; diffusion not retrained"}
        write_json(run_dir / "COMPLETE.json", result)
        print(f"Complete: {run_dir}", flush=True)
        return result
    except BaseException as error:
        write_json(run_dir / "FAILED.json", {"error": str(error), "type": type(error).__name__})
        raise
