"""Representation-only runs using the exact verified vanilla QVF decoder."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .data import load_data
from .reconstruction import DEFAULT_REFERENCE, study_configs
from .train import (DEFAULT_DATA, HERE, RunConfig, configure_runtime,
                    evaluate_vae, load_vae, save_checkpoint, sha256,
                    snapshot_sources, train_vae, write_json)

VANILLA_ROOT = HERE.parent / "qvf_repro" / "results" / "20261108_vanilla"
VANILLA_NAME = "paper_qvf_gaussian_relu_n50_seed0_bs10"


def vanilla_configs(seed, reference_run=DEFAULT_REFERENCE):
    # The 50-image run tests fit capacity with an encoder on the original images.
    # Final-checkpoint training reconstructions are kept separately from the
    # validation-selected checkpoint; these answer different questions.
    fit = RunConfig(seed=seed, n_train=50, n_val=256, batch_size=10,
                    pixels_per_image=1024, vae_epochs=5000, vae_max_steps=25000,
                    vae_eval_every_steps=1000, kl_warmup_steps=500, vae_drop_last=True)
    # Reuse exactly the existing larger-data protocol, changing only the field
    # implementation/readout to vanilla. The encoder and VAE loss stay the same.
    large = dict(study_configs(seed, reference_run))["more_data_longer"]
    large.model.readout = "vanilla"
    return [("fit50", fit), ("larger", large)]


def fitting_split(data_root, n_val):
    original = json.loads((VANILLA_ROOT / f"{VANILLA_NAME}.json").read_text())
    training = original["dataset_indices"]
    dataset = CIFAR10(str(data_root), train=True, download=False)
    candidates = [i for i, label in enumerate(dataset.targets) if label == 5 and i not in set(training)]
    if len(candidates) < n_val:
        raise ValueError("Not enough held-out dogs for the requested validation split")
    return {"train": training, "validation": candidates[:n_val]}


@torch.no_grad()
def save_reconstructions(model, images, indices, directory, cfg, device):
    """Save raw tensors for all images, plus clipped display grids and raw metrics."""
    directory.mkdir()
    metrics = {}
    for name, values in images.items():
        result, mean_images, sampled_images = evaluate_vae(
            model, values, cfg.batch_size, device, cfg.seed + (1001 if name == "validation" else 1002))
        error = mean_images - values
        per_image_mse = error.square().flatten(1).mean(1)
        gradient_mse = 0.5 * ((error[:, :, 1:] - error[:, :, :-1]).square().flatten(1).mean(1)
                               + (error[:, :, :, 1:] - error[:, :, :, :-1]).square().flatten(1).mean(1))
        result.update({"count": len(values), "dataset_indices": indices[name],
                       "mean_code_per_image_psnr_db": (-10 * per_image_mse.clamp_min(1e-12).log10()).tolist(),
                       "mean_code_gradient_mse": gradient_mse.mean().item(),
                       "mean_code_raw_range": [mean_images.min().item(), mean_images.max().item()],
                       "mean_code_fraction_outside_rgb": ((mean_images < 0) | (mean_images > 1)).float().mean().item()})
        metrics[name] = result
        save_checkpoint(directory / f"{name}.pt", {
            "originals": values, "mean_reconstructions": mean_images,
            "sampled_reconstructions": sampled_images, "dataset_indices": indices[name]})
        # Every image remains available in the raw tensor. 50-image grids show
        # all training examples; larger training previews use the first 50.
        count = min(50, len(values))
        save_image(torch.cat((values[:count], mean_images[:count], sampled_images[:count])),
                   directory / f"{name}.png", nrow=10, padding=1)
    write_json(directory / "metrics.json", metrics)
    return metrics


def run_vanilla_vae(cfg, run_dir, data_root=DEFAULT_DATA, device_name="cuda:0",
                    reference_run=DEFAULT_REFERENCE, variant="larger"):
    cfg.validate()
    if cfg.model.readout != "vanilla" or cfg.model.decoder != "qvf" or cfg.model.mode != "vae":
        raise ValueError("This group is specifically the VAE with the vanilla QVF field")
    run_dir = Path(run_dir).resolve()
    reference_run = Path(reference_run).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        device = configure_runtime(device_name, cfg.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        reference = (fitting_split(data_root, cfg.n_val) if variant == "fit50" else
                     json.loads((reference_run / "split.json").read_text())["indices"])
        images, indices = load_data(data_root, cfg.class_idx, cfg.n_train, cfg.n_val,
                                    cfg.split_seed, reference)
        write_json(run_dir / "config.json", {**asdict(cfg), "variant": variant,
            "data_root": str(Path(data_root).resolve()), "device": device_name,
            "reference_run": str(reference_run), "initialization": "encoder and vanilla field trained from scratch",
            "scope": "VAE reconstruction only; no diffusion training"})
        snapshot_sources(run_dir)
        write_json(run_dir / "split.json", {"dataset": "CIFAR10", "source": "official training set",
            "official_test_set_used": False, "indices": indices,
            "training_ids_match_vanilla50": variant == "fit50"})
        write_json(run_dir / "environment.json", {"torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})
        best_path, training = train_vae(cfg, images, run_dir, device)
        # Store final fit separately so a validation selection cannot conceal
        # whether the encoder can reproduce the known 50 training images.
        selected, _ = load_vae(best_path, device)
        selected_metrics = save_reconstructions(selected, images, indices, run_dir / "selected", cfg, device)
        final, _ = load_vae(run_dir / "vae/checkpoints/last.pt", device)
        final_metrics = save_reconstructions(final, images, indices, run_dir / "final", cfg, device)
        result = {"status": "complete", "kind": "vanilla_qvf_vae", "variant": variant,
            "elapsed_seconds": time.perf_counter() - started, "vae": training,
            "selected": selected_metrics, "final": final_metrics,
            "checkpoint_sha256": {"best": sha256(best_path), "last": sha256(run_dir / "vae/checkpoints/last.pt")},
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "scope": "representation training; no diffusion or direct per-image code optimization",
            "quantum_execution": "exact classical simulation"}
        write_json(run_dir / "COMPLETE.json", result)
        print(f"Complete: {run_dir}", flush=True)
        return result
    except BaseException as error:
        write_json(run_dir / "FAILED.json", {"error": str(error), "type": type(error).__name__})
        raise
