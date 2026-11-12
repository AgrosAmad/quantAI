"""Direct per-image-code fitting with the unchanged vanilla QVF field.

There is no image encoder, posterior sampling, KL penalty, or diffusion model.
Run training through the named ``direct_qvf`` group in sweep.py. This module's
CLI writes the protocol before launching the study.
"""

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from .data import load_data
from .models import CoordinateDecoder, ModelConfig, pixel_coordinates
from .reconstruction import DEFAULT_REFERENCE
from .train import (DEFAULT_DATA, HERE, REPO, configure_runtime, image_metrics,
                    save_checkpoint, sha256, snapshot_sources, write_json)


@dataclass
class DirectConfig:
    model: ModelConfig = field(default_factory=lambda: ModelConfig(mode="ae"))
    seed: int = 0
    split_seed: int = 20261108
    class_idx: int = 5
    n_train: int = 4744
    n_val: int = 256
    batch_size: int = 32
    pixels_per_image: int = 256
    steps: int = 50000
    model_lr: float = 0.001
    code_lr: float = 0.001
    code_init_std: float = 0.01
    code_norm_weight: float = 0.0001
    eval_every: int = 5000

    def validate(self):
        if (self.model.decoder, self.model.readout, self.model.mode) != ("qvf", "vanilla", "ae"):
            raise ValueError("Direct fitting requires the deterministic vanilla QVF decoder")
        if min(self.n_train, self.n_val, self.batch_size, self.steps, self.eval_every) < 1:
            raise ValueError("Counts and budgets must be positive")
        if self.n_train < self.batch_size or not 1 <= self.pixels_per_image <= self.model.image_size ** 2:
            raise ValueError("Invalid batch or pixel count")
        if min(self.model_lr, self.code_lr, self.code_init_std) <= 0 or self.code_norm_weight < 0:
            raise ValueError("Invalid optimization settings")


def direct_configs(seed, reference_run=DEFAULT_REFERENCE):
    # The reference split is supplied to the runner, as for the earlier studies.
    cfg = DirectConfig(seed=seed)
    cfg.validate()
    return [("large50k", cfg)]


def evaluation_steps(cfg):
    return sorted({0, 1, cfg.steps, *range(cfg.eval_every, cfg.steps + 1, cfg.eval_every),
                   *([7680] if cfg.steps >= 7680 else [])})


def direct_objective(decoder, codes, image_rows, coordinates, targets, weight):
    selected_codes = codes[image_rows]
    predictions = decoder(selected_codes, coordinates)
    mse = F.mse_loss(predictions, targets)
    # Repeating each code for every queried pixel gives the same mean norm.
    penalty = weight * selected_codes.norm(dim=1).mean()
    return mse + penalty, mse, penalty


@torch.no_grad()
def render_direct(decoder, codes, batch_size=32, pixel_chunk=1024):
    decoder.eval()
    size = decoder.cfg.image_size
    coordinates = pixel_coordinates(size, size, device=codes.device)
    images = []
    for batch in codes.split(batch_size):
        pixels = torch.cat([decoder(batch, points) for points in coordinates.split(pixel_chunk)], dim=1)
        images.append(pixels.transpose(1, 2).reshape(len(batch), 3, size, size).cpu())
    return torch.cat(images)


def reconstruction_metrics(prediction, target):
    error = prediction - target
    per_image_mse = error.square().flatten(1).mean(1)
    gradient_mse = 0.5 * ((error[:, :, 1:] - error[:, :, :-1]).square().flatten(1).mean(1)
        + (error[:, :, :, 1:] - error[:, :, :, :-1]).square().flatten(1).mean(1))
    return {**image_metrics(prediction, target), "count": len(target),
            "per_image_psnr_db": (-10 * per_image_mse.clamp_min(1e-12).log10()).tolist(),
            "gradient_mse": gradient_mse.mean().item(),
            "raw_range": [prediction.min().item(), prediction.max().item()],
            "fraction_outside_rgb": ((prediction < 0) | (prediction > 1)).float().mean().item()}


def load_direct(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "direct_qvf" or checkpoint.get("format_version") != 1:
        raise ValueError("Not a direct QVF checkpoint")
    decoder = CoordinateDecoder(ModelConfig.from_saved(checkpoint["model_config"])).to(device)
    decoder.load_state_dict(checkpoint["decoder_state"])
    codes = torch.nn.Parameter(checkpoint["codes"].to(device))
    return decoder.eval(), codes, checkpoint


def train_direct(cfg, images, dataset_indices, run_dir, device, *,
                 decoder_factory=CoordinateDecoder, checkpoint_kind="direct_qvf"):
    """Fit one shared field and a table of independent deterministic image codes."""
    cfg.validate()
    if len(images) != cfg.n_train or len(dataset_indices) != cfg.n_train:
        raise ValueError("Training tensors and IDs must match the declared dataset size")
    directory = Path(run_dir) / "fit"
    directory.mkdir()
    # Same code initialization convention as the released direct-fitting method.
    codes = torch.nn.Parameter(torch.randn(cfg.n_train, cfg.model.latent_dim, device=device) * cfg.code_init_std)
    decoder = decoder_factory(cfg.model).to(device)
    optimizer_model = torch.optim.Adam(decoder.parameters(), lr=cfg.model_lr)
    optimizer_codes = torch.optim.Adam([codes], lr=cfg.code_lr)
    order_rng = torch.Generator().manual_seed(cfg.seed + 11)
    pixel_rng = torch.Generator(device=device).manual_seed(cfg.seed + 12)
    coordinates = pixel_coordinates(cfg.model.image_size, cfg.model.image_size, device=device)
    targets = images.flatten(2).transpose(1, 2).contiguous().to(device)
    visits = torch.zeros(cfg.n_train, dtype=torch.int64)
    steps, observations, coordinate_queries = 0, 0, 0
    best_mse, best_step = float("inf"), None
    eval_at = set(evaluation_steps(cfg))
    start = time.perf_counter()

    with (directory / "history.jsonl").open("x") as history:
        def evaluate(epoch):
            nonlocal best_mse, best_step
            prediction = render_direct(decoder, codes, cfg.batch_size)
            metrics = reconstruction_metrics(prediction, images)
            record = {"event": "fitting_evaluation", "step": steps, "epoch": epoch,
                      "fit": {k: v for k, v in metrics.items() if k != "per_image_psnr_db"},
                      "mean_code_norm": codes.detach().norm(dim=1).mean().item()}
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()
            improved = metrics["mse"] < best_mse
            if improved:
                best_mse, best_step = metrics["mse"], steps
            payload = {"format_version": 1, "kind": checkpoint_kind, "model_config": asdict(cfg.model),
                       "run_config": asdict(cfg), "step": steps, "epoch": epoch,
                       "decoder_state": decoder.state_dict(), "codes": codes,
                       "dataset_indices": dataset_indices, "visits": visits,
                       "optimizer_model": optimizer_model.state_dict(),
                       "optimizer_codes": optimizer_codes.state_dict(),
                       "rng": {"order": order_rng.get_state(), "pixels": pixel_rng.get_state()},
                       "fit": record["fit"], "best_fitting_step": best_step,
                       "training_image_observations": observations,
                       "training_coordinate_queries": coordinate_queries}
            save_checkpoint(directory / "checkpoints/last.pt", payload)
            if improved:
                save_checkpoint(directory / "checkpoints/best_fit.pt", payload)
            if steps in {0, 7680}:
                save_checkpoint(directory / "checkpoints" / f"step{steps}.pt", payload)
            print(f"Direct QVF step {steps}/{cfg.steps}: full training PSNR {metrics['psnr_db']:.3f} dB, "
                  f"MSE {metrics['mse']:.6f}, edge error {metrics['gradient_mse']:.6f}", flush=True)
            return prediction, metrics

        evaluate(0)
        epochs = math.ceil(cfg.steps / (cfg.n_train // cfg.batch_size))
        for epoch in range(1, epochs + 1):
            sums = torch.zeros(3, device=device)
            epoch_observations = 0
            for rows in torch.randperm(cfg.n_train, generator=order_rng).split(cfg.batch_size):
                # Match the VAE's shuffled full-image batches and pixel sampler.
                if len(rows) < cfg.batch_size:
                    continue
                gpu_rows = rows.to(device)
                points = torch.randperm(cfg.model.image_size ** 2, device=device, generator=pixel_rng)[:cfg.pixels_per_image]
                decoder.train()
                optimizer_model.zero_grad(set_to_none=True)
                optimizer_codes.zero_grad(set_to_none=True)
                loss, mse, penalty = direct_objective(decoder, codes, gpu_rows, coordinates[points],
                                                       targets[gpu_rows][:, points], cfg.code_norm_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite direct-fitting loss at step {steps + 1}")
                loss.backward()
                # As in the original direct QVF trainer, no gradient clipping.
                optimizer_model.step()
                optimizer_codes.step()
                steps += 1
                observations += len(rows)
                coordinate_queries += len(rows) * len(points)
                visits[rows] += 1
                epoch_observations += len(rows)
                sums += torch.stack((loss.detach(), mse.detach(), penalty.detach())) * len(rows)
                if steps in eval_at:
                    final_prediction, final_metrics = evaluate(epoch)
                elif steps % 1000 == 0:
                    print(f"Direct QVF step {steps}/{cfg.steps}: current-epoch sampled MSE "
                          f"{(sums[1] / epoch_observations).item():.6f}", flush=True)
                if steps == cfg.steps:
                    break
            values = (sums / epoch_observations).tolist()
            history.write(json.dumps({"event": "epoch", "step": steps, "epoch": epoch,
                "loss": values[0], "pixel_mse": values[1], "code_penalty": values[2]}, allow_nan=False) + "\n")
            history.flush()

    final_dir = Path(run_dir) / "final"
    final_dir.mkdir()
    save_checkpoint(final_dir / "train.pt", {"originals": images, "reconstructions": final_prediction,
                                             "dataset_indices": dataset_indices})
    positions = torch.linspace(0, len(images) - 1, min(256, len(images))).round().long()
    save_image(torch.cat((images[positions], final_prediction[positions])), final_dir / "train.png", nrow=16, padding=1)
    write_json(final_dir / "metrics.json", final_metrics)
    return {"optimizer_steps": steps, "training_image_observations": observations,
            "training_coordinate_queries": coordinate_queries, "unique_training_images_seen": int((visits > 0).sum()),
            "visits_per_image": visits.tolist(), "equivalent_dataset_passes": observations / cfg.n_train,
            "elapsed_seconds": time.perf_counter() - start, "best_fitting_step": best_step,
            "parameter_counts": {"decoder": sum(p.numel() for p in decoder.parameters()),
                                 "circuit": sum(p.numel() for p in decoder.circuit.parameters()),
                                 "image_codes": codes.numel(), "encoder": 0},
            "final_fit": final_metrics,
            "checkpoint_sha256": {name: sha256(directory / "checkpoints" / f"{name}.pt")
                                  for name in ("last", "best_fit", "step0")},
            "gallery_training_positions": positions.tolist()}


def run_direct_qvf(cfg, run_dir, data_root=DEFAULT_DATA, device_name="cuda:0",
                   reference_run=DEFAULT_REFERENCE, variant="large50k"):
    run_dir, reference_run = Path(run_dir).resolve(), Path(reference_run).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        device = configure_runtime(device_name, cfg.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        reference = json.loads((reference_run / "split.json").read_text())["indices"]
        images, indices = load_data(data_root, cfg.class_idx, cfg.n_train, cfg.n_val, cfg.split_seed, reference)
        write_json(run_dir / "config.json", {**asdict(cfg), "variant": variant, "device": device_name,
            "data_root": str(Path(data_root).resolve()), "reference_run": str(reference_run)})
        write_json(run_dir / "split.json", {"dataset": "CIFAR10", "official_test_set_used": False,
            "indices": indices, "validation_images_used_for_optimization_or_selection": False})
        snapshot_sources(run_dir)
        write_json(run_dir / "environment.json", {"torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})
        result = train_direct(cfg, images["train"], indices["train"], run_dir, device)
        result.update({"status": "complete", "kind": "direct_qvf_fitting", "variant": variant,
                       "scope": "fit the training images; no encoder, VAE, diffusion or unseen-image inference",
                       "quantum_execution": "exact classical simulation; five-qubit circuit for the named study",
                       "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0})
        write_json(run_dir / "COMPLETE.json", result)
        print(f"Complete: {run_dir}", flush=True)
        return result
    except BaseException as error:
        write_json(run_dir / "FAILED.json", {"error": str(error), "type": type(error).__name__})
        raise


def write_protocol(root, tag, seeds, reference_run=DEFAULT_REFERENCE):
    root, reference_run = Path(root).resolve(), Path(reference_run).resolve()
    runs, previous = [], []
    for seed in seeds:
        label, cfg = direct_configs(seed)[0]
        runs.append({"variant": label, "config": asdict(cfg),
                     "directory": str(root / f"{tag}_direct_qvf_{label}_seed{seed}")})
        directory = root / f"20261108_v4_capacity_baseline50k_seed{seed}"
        previous.append({"seed": seed, "directory": str(directory),
                         "last_sha256": sha256(directory / "vae/checkpoints/last.pt")})
    protocol = {"kind": "direct_qvf_larger_dataset", "tag": tag, "seeds": seeds,
        "runs": runs, "vae_comparisons": previous, "reference_run": str(reference_run),
        "reference_split_sha256": sha256(reference_run / "split.json"),
        "decoder_source_sha256": sha256(REPO / "experiments/qvf_repro/qvf_fast.py"),
        "question": "Can the original shared QVF field fit the same 4744 images using directly optimized image codes?",
        "fixed": "Same image IDs, 64-number codes, width 128, three hidden layers, five qubits, depth five, original grouped RGB readout; 50000 updates, batch 32 and 256 pixel queries per image.",
        "objective": "Pixel MSE + 0.0001 * mean L2 code norm, as in released direct QVF. No VAE KL or posterior sampling.",
        "optimizer": "Separate dense Adam optimizers for field and code table, both constant lr 0.001. No gradient clipping, as in original direct QVF. No plateau scheduler, matching the previous 50000-step comparison's constant lr.",
        "initialization": "Fresh Gaussian code table (std 0.01) and original field initialization. Seeds 0,1,2; no encoder or previous weights are used.",
        "sampling": "Same independent order and pixel RNG seeds and shuffled full-batch policy as the VAE; trailing partial batches dropped.",
        "comparison": "Primary: FINAL full-training-image metrics versus the VAE's FINAL training reconstructions. This removes encoder inference and VAE regularization together; it does not isolate their individual effects.",
        "selection": "Final 50000-step checkpoint is primary; best fitting MSE checkpoint is also saved. Validation images never select models or codes.",
        "evaluation": "All 1024 pixels of all 4744 fitted training images; raw unclipped predictions. PSNR and gradient error plus a fixed 256-image visual gallery.",
        "gallery_training_positions": torch.linspace(0, 4743, 256).round().long().tolist(),
        "limits": "This is fitting, not held-out generalization. A poor result cannot establish a fundamental circuit capacity limit or convergence. Adapted large-data budget, not a reproduction of a published large-data benchmark.",
        "resource_scope": "Existing quantai environment; GPU 0 only; at most three training processes",
        "convolutions_added": False, "diffusion_trained": False, "official_test_set_used": False}
    output = root / f"{tag}_direct_qvf_protocol.json"
    with output.open("x") as handle:
        json.dump(protocol, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--root", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use a simple fresh tag and distinct seeds")
    print(write_protocol(args.root, args.tag, args.seeds))


if __name__ == "__main__":
    main()
