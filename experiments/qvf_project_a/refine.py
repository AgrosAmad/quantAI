"""Fit codes for known validation images with the trained decoder frozen.

This is an image-specific reconstruction diagnostic, not feed-forward inference
or a measurement of unconditional generation quality. The default minimizes
deterministic reconstruction MSE, not the full stochastic VAE objective.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from .data import load_data
from .reconstruction import DEFAULT_REFERENCE, reference_config
from .train import (DEFAULT_DATA, configure_runtime, load_vae, sha256, snapshot_sources, write_json)


def refine_codes(model, images, *, batch_size=32, steps=400, lr=0.03, prior_weight=0.0, log_every=25):
    if min(batch_size, steps, log_every) < 1 or lr <= 0 or prior_weight < 0 or len(images) == 0:
        raise ValueError("Invalid code-refinement settings")
    device = next(model.parameters()).device
    model.eval().requires_grad_(False)
    before_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_codes, selected_codes, initial_errors, selected_errors, final_errors = [], [], [], [], []
    best_steps, batches_history = [], []
    for batch_index, batch in enumerate(images.split(batch_size)):
        batch = batch.to(device)
        with torch.no_grad():
            mean, _ = model.encoder(batch)
        initial_codes.append(mean.cpu())
        code = mean.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([code], lr=lr)
        best_score = torch.full((len(batch),), float("inf"), device=device)
        best_code = code.detach().clone()
        best_mse = best_score.clone()
        best_step = torch.zeros(len(batch), device=device, dtype=torch.long)
        history = []
        for step in range(steps + 1):
            predictions = model.render(code)
            per_image_mse = (predictions - batch).square().flatten(1).mean(1)
            score = per_image_mse + 0.5 * prior_weight * code.square().sum(1)
            if not torch.isfinite(score).all():
                raise FloatingPointError("Non-finite code-refinement objective")
            if step == 0:
                initial_errors.append(per_image_mse.detach().cpu())
            with torch.no_grad():
                improved = score < best_score
                best_score[improved] = score[improved]
                best_code[improved] = code[improved]
                best_mse[improved] = per_image_mse[improved]
                best_step[improved] = step
                if step % log_every == 0 or step == steps:
                    history.append({"step": step, "count": len(batch),
                        "mse_sum": per_image_mse.sum().item(),
                        "psnr_sum": (-10 * per_image_mse.clamp_min(1e-12).log10()).sum().item(),
                        "best_mse_sum": best_mse.sum().item(),
                        "best_psnr_sum": (-10 * best_mse.clamp_min(1e-12).log10()).sum().item()})
            if step == steps:
                final_errors.append(per_image_mse.detach().cpu())
                break
            optimizer.zero_grad(set_to_none=True)
            score.mean().backward()
            torch.nn.utils.clip_grad_norm_([code], 1.0, error_if_nonfinite=True)
            optimizer.step()
        selected_codes.append(best_code.cpu())
        selected_errors.append(best_mse.cpu())
        best_steps.append(best_step.cpu())
        batches_history.append(history)
        print(f"Code refinement: batch {batch_index + 1}/{(len(images) + batch_size - 1) // batch_size}, "
              f"best mean PSNR {(-10 * best_mse.clamp_min(1e-12).log10()).mean().item():.2f} dB", flush=True)
    unchanged = all(torch.equal(value.detach().cpu(), before_state[key]) for key, value in model.state_dict().items())
    if not unchanged or any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Code fitting changed or differentiated a model weight")
    history = []
    for index, first in enumerate(batches_history[0]):
        row = {"step": first["step"]}
        for key in ("mse", "psnr", "best_mse", "best_psnr"):
            row[key] = sum(batch[index][key + "_sum"] for batch in batches_history) / len(images)
        history.append(row)
    return {"initial_codes": torch.cat(initial_codes), "refined_codes": torch.cat(selected_codes),
            "initial_mse": torch.cat(initial_errors), "refined_mse": torch.cat(selected_errors),
            "final_mse": torch.cat(final_errors), "best_steps": torch.cat(best_steps),
            "history": history, "model_weights_unchanged": unchanged}


def run_refinement(run, output, device_name="cuda:0", data_root=DEFAULT_DATA,
                   steps=400, lr=0.03, batch_size=32, prior_weight=0.0):
    run, output = Path(run).resolve(), Path(output).resolve()
    cfg = reference_config(run)
    checkpoint_path = run / "vae/checkpoints/best.pt"
    checkpoint_hash = sha256(checkpoint_path)
    split = json.loads((run / "split.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        device = configure_runtime(device_name, cfg.seed)
        model, _ = load_vae(checkpoint_path, device)
        images, indices = load_data(data_root, cfg.class_idx, cfg.n_train, cfg.n_val,
                                    cfg.split_seed, split["indices"])
        write_json(output / "config.json", {"source_run": str(run), "checkpoint_sha256": checkpoint_hash,
            "steps": steps, "learning_rate": lr, "batch_size": batch_size,
            "prior_weight": prior_weight, "objective": "pixel MSE + prior_weight * squared code norm / 2",
            "fit_pixels": "all 1024 pixels of each validation image", "device": device_name})
        snapshot_sources(output)
        result = refine_codes(model, images["validation"], batch_size=batch_size, steps=steps, lr=lr, prior_weight=prior_weight)
        if sha256(checkpoint_path) != checkpoint_hash:
            raise AssertionError("Source checkpoint changed")
        initial_psnr = -10 * result["initial_mse"].clamp_min(1e-12).log10()
        refined_psnr = -10 * result["refined_mse"].clamp_min(1e-12).log10()
        gains = (refined_psnr - initial_psnr).numpy()
        rng = np.random.default_rng(17)
        bootstrap = gains[rng.integers(0, len(gains), size=(10000, len(gains)))].mean(1)
        movement = (result["refined_codes"] - result["initial_codes"]).square().sum(1).sqrt()
        metrics = {"status": "complete", "kind": "frozen_decoder_code_refinement", "images": len(gains),
            "initial_psnr_db": initial_psnr.mean().item(), "refined_psnr_db": refined_psnr.mean().item(),
            "mean_gain_db": float(gains.mean()), "median_gain_db": float(np.median(gains)),
            "paired_image_bootstrap_95_percent": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "interval_scope": "uncertainty over these images, conditional on one trained model; not across seeds",
            "initial_mse": result["initial_mse"].mean().item(), "refined_mse": result["refined_mse"].mean().item(),
            "mean_code_displacement_l2": movement.mean().item(),
            "initial_mean_code_norm": result["initial_codes"].norm(dim=1).mean().item(),
            "refined_mean_code_norm": result["refined_codes"].norm(dim=1).mean().item(),
            "model_weights_unchanged": result["model_weights_unchanged"], "checkpoint_unchanged": True,
            "elapsed_seconds": time.perf_counter() - started,
            "interpretation": "Per-image fitting using the target photograph. Not feed-forward inference, generation quality, or proof of a globally optimal code."}
        write_json(output / "history.json", result["history"])
        write_json(output / "per_image.json", {"dataset_indices": indices["validation"],
            "initial_psnr_db": initial_psnr.tolist(), "refined_psnr_db": refined_psnr.tolist(),
            "gain_db": gains.tolist(), "best_step": result["best_steps"].tolist(),
            "code_displacement_l2": movement.tolist()})
        torch.save({key: value for key, value in result.items() if isinstance(value, torch.Tensor)}, output / "refined_codes.pt")
        with torch.no_grad():
            count = min(16, len(images["validation"]))
            initial = model.render(result["initial_codes"][:count].to(device)).cpu()
            refined = model.render(result["refined_codes"][:count].to(device)).cpu()
        save_image(torch.cat((images["validation"][:count], initial, refined)), output / "reconstruction.png", nrow=count)
        write_json(output / "COMPLETE.json", metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        return metrics
    except BaseException as error:
        write_json(output / "FAILED.json", {"error": str(error), "type": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    args = parser.parse_args()
    run_refinement(args.run, args.output, args.device, args.data_root, args.steps, args.lr, args.batch_size, args.prior_weight)


if __name__ == "__main__":
    main()
