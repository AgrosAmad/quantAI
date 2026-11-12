"""Reproducible two-stage training. Run via the named groups in sweep.py."""

import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from .data import encode_images, load_data, posterior_normalization
from .diffusion import LatentDDPM, LatentDenoiser
from .models import ModelConfig, QVFAutoencoder, kl_per_image, sample_posterior

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_DATA = HERE.parent / "qvf_repro" / "data"


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 0
    split_seed: int = 20261108
    class_idx: int = 5
    n_train: int = 1024
    n_val: int = 256
    batch_size: int = 32
    pixels_per_image: int = 256
    vae_epochs: int = 80
    vae_lr: float = 0.001
    kl_weight: float = 0.00001
    kl_warmup_epochs: int = 10
    vae_eval_every: int = 10
    vae_max_steps: int | None = None
    vae_eval_every_steps: int | None = None
    kl_warmup_steps: int | None = None
    vae_drop_last: bool = False
    diffusion_steps: int = 4000
    diffusion_batch_size: int = 256
    diffusion_lr: float = 0.0003
    diffusion_width: int = 256
    diffusion_layers: int = 4
    diffusion_timesteps: int = 100
    diffusion_eval_every: int = 250
    ema_decay: float = 0.995
    sample_count: int = 64

    def validate(self):
        positive = (self.batch_size, self.vae_epochs, self.vae_eval_every, self.diffusion_steps,
                    self.diffusion_batch_size, self.diffusion_eval_every, self.sample_count)
        if any(value < 1 for value in positive):
            raise ValueError("Training budgets, batches, evaluation intervals and sample counts must be positive.")
        if not 1 <= self.pixels_per_image <= self.model.image_size ** 2:
            raise ValueError("Invalid number of sampled pixels")
        if self.vae_lr <= 0 or self.diffusion_lr <= 0 or self.kl_weight < 0 or self.kl_warmup_epochs < 0:
            raise ValueError("Invalid learning rate, KL weight or warmup")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        if self.n_train < 2 or self.n_val < 2 or self.class_idx not in range(-1, 10):
            raise ValueError("Invalid dataset split or class selection")
        if self.diffusion_width < 1 or self.diffusion_layers < 1 or self.diffusion_timesteps < 2:
            raise ValueError("Invalid diffusion architecture or timestep count")
        if self.vae_max_steps is not None and self.vae_max_steps < 1:
            raise ValueError("vae_max_steps must be positive")
        if self.vae_eval_every_steps is not None and self.vae_eval_every_steps < 1:
            raise ValueError("vae_eval_every_steps must be positive")
        if self.kl_warmup_steps is not None and self.kl_warmup_steps < 0:
            raise ValueError("kl_warmup_steps must be nonnegative")
        if self.vae_drop_last and self.n_train < self.batch_size:
            raise ValueError("Dropping partial batches requires at least one full training batch.")


def configure_runtime(device, seed):
    if device not in {"cpu", "cuda:0"}:
        raise ValueError("Only cpu or cuda:0 is allowed on this shared machine.")
    if device == "cuda:0":
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "0"}:
            raise ValueError("Set CUDA_VISIBLE_DEVICES=0 to avoid using physical GPU 1.")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; choose --device cpu explicitly.")
        torch.cuda.init()
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch.device(device)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(cpu_tree(payload), temporary)
    temporary.replace(path)


def load_vae(path, device):
    # Checkpoints are generated locally by this experiment; do not load untrusted pickles.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "qvf_vae" or checkpoint.get("format_version") != 1:
        raise ValueError("Not a supported Project A autoencoder checkpoint")
    model = QVFAutoencoder(ModelConfig.from_saved(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), checkpoint


def image_metrics(predictions, targets):
    errors = (predictions - targets).square().flatten(1).mean(dim=1)
    psnr = -10 * errors.clamp_min(1e-12).log10()
    return {"mse": errors.mean().item(), "psnr_db": psnr.mean().item()}


@torch.no_grad()
def evaluate_vae(model, images, batch_size, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    means, samples, shuffled, kls = [], [], [], []
    for batch in images.split(batch_size):
        batch = batch.to(device)
        mean, logvar = model.encoder(batch)
        means.append(model.render(mean).cpu())
        samples.append(model.render(sample_posterior(mean, logvar, generator=generator)).cpu())
        shuffled.append(model.render(mean.roll(1, dims=0)).cpu())
        kls.append(kl_per_image(mean, logvar).cpu())
    mean_images, sampled_images = torch.cat(means), torch.cat(samples)
    result = {"mean_code": image_metrics(mean_images, images),
              "sampled_code": image_metrics(sampled_images, images),
              "shuffled_code": image_metrics(torch.cat(shuffled), images),
              "kl_nats_per_image": torch.cat(kls).mean().item()}
    return result, mean_images, sampled_images


def train_vae(cfg, images, run_dir, device):
    directory = run_dir / "vae"
    directory.mkdir()
    model = QVFAutoencoder(cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr)
    order_rng = torch.Generator().manual_seed(cfg.seed + 11)
    pixel_rng = torch.Generator(device=device).manual_seed(cfg.seed + 12)
    posterior_rng = torch.Generator(device=device).manual_seed(cfg.seed + 13)
    initial, _, _ = evaluate_vae(model, images["validation"], cfg.batch_size, device, cfg.seed + 1001)
    mean_image = images["train"].mean(dim=0, keepdim=True).expand_as(images["validation"])
    initial["training_mean_image_baseline"] = image_metrics(mean_image, images["validation"])
    write_json(directory / "initial.json", initial)
    best_score, best_epoch = float("inf"), None
    start = time.perf_counter()
    steps = 0
    observations = 0
    seen = torch.zeros(len(images["train"]), dtype=torch.bool)
    batches_per_epoch = (len(images["train"]) // cfg.batch_size if cfg.vae_drop_last else
                         math.ceil(len(images["train"]) / cfg.batch_size))
    if batches_per_epoch < 1:
        raise ValueError("Training has no batches")
    target_steps = cfg.vae_max_steps or cfg.vae_epochs * batches_per_epoch
    epochs = math.ceil(target_steps / batches_per_epoch)

    def validate_and_save(epoch, record):
        nonlocal best_score, best_epoch
        validation, mean_images, sampled_images = evaluate_vae(
            model, images["validation"], cfg.batch_size, device, cfg.seed + 1001)
        score = validation["sampled_code"]["mse"] + (cfg.kl_weight if cfg.model.mode == "vae" else 0) * validation["kl_nats_per_image"]
        record["validation"] = validation
        improved = score < best_score
        if improved:
            best_score, best_epoch = score, epoch
        checkpoint = {"format_version": 1, "kind": "qvf_vae", "model_config": asdict(cfg.model),
                      "run_config": asdict(cfg), "epoch": epoch, "step": steps,
                      "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "validation": validation, "best_epoch": best_epoch, "best_score": best_score,
                      "rng": {"order": order_rng.get_state(), "pixels": pixel_rng.get_state(),
                              "posterior": posterior_rng.get_state()}}
        save_checkpoint(directory / "checkpoints" / "last.pt", checkpoint)
        if improved:
            save_checkpoint(directory / "checkpoints" / "best.pt", checkpoint)
            count = min(16, len(images["validation"]))
            save_image(torch.cat((images["validation"][:count], mean_images[:count], sampled_images[:count])),
                       directory / "validation_reconstruction.png", nrow=count)
        print(f"VAE step {steps}/{target_steps} (epoch {epoch}): train MSE {record['pixel_mse']:.5f}, "
              f"validation mean-code PSNR {validation['mean_code']['psnr_db']:.2f} dB, "
              f"KL {validation['kl_nats_per_image']:.2f}", flush=True)

    with open(directory / "history.jsonl", "x") as history:
        for epoch in range(1, epochs + 1):
            sums = torch.zeros(3, device=device)
            epoch_observations = 0
            for indices in torch.randperm(len(images["train"]), generator=order_rng).split(cfg.batch_size):
                if cfg.vae_drop_last and len(indices) < cfg.batch_size:
                    continue
                model.train()
                warmup = ((steps + 1) / max(1, cfg.kl_warmup_steps) if cfg.kl_warmup_steps is not None else
                          epoch / max(1, cfg.kl_warmup_epochs))
                beta = cfg.kl_weight * min(1.0, warmup) if cfg.model.mode == "vae" else 0.0
                batch = images["train"][indices].to(device)
                pixel_indices = torch.randperm(cfg.model.image_size ** 2, device=device, generator=pixel_rng)[:cfg.pixels_per_image]
                predictions, mean, logvar = model(batch, pixel_indices, generator=posterior_rng)
                targets = batch.flatten(2).transpose(1, 2)[:, pixel_indices]
                mse = F.mse_loss(predictions, targets)
                kl = kl_per_image(mean, logvar).mean()
                loss = mse + beta * kl
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite VAE loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                sums += torch.stack((loss.detach(), mse.detach(), kl.detach())) * len(batch)
                steps += 1
                observations += len(batch)
                epoch_observations += len(batch)
                seen[indices] = True
                if cfg.vae_eval_every_steps is not None and (
                        steps == 1 or steps % cfg.vae_eval_every_steps == 0 or steps == target_steps):
                    average_loss, average_mse, average_kl = (sums / epoch_observations).tolist()
                    record = {"event": "validation", "epoch": epoch, "step": steps, "loss": average_loss,
                              "pixel_mse": average_mse, "kl_nats": average_kl, "kl_weight": beta}
                    validate_and_save(epoch, record)
                    history.write(json.dumps(record, allow_nan=False) + "\n")
                    history.flush()
                if steps == target_steps:
                    break
            loss, mse, kl = (sums / epoch_observations).tolist()
            record = {"event": "epoch", "epoch": epoch, "step": steps, "loss": loss,
                      "pixel_mse": mse, "kl_nats": kl, "kl_weight": beta}
            if cfg.vae_eval_every_steps is None and (epoch == 1 or epoch % cfg.vae_eval_every == 0 or epoch == epochs):
                validate_and_save(epoch, record)
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()
    best_path = directory / "checkpoints" / "best.pt"
    model, checkpoint = load_vae(best_path, device)
    train_eval, _, _ = evaluate_vae(model, images["train"][:min(256, len(images["train"]))], cfg.batch_size, device, cfg.seed + 1002)
    result = {"best_epoch": best_epoch, "best_step": checkpoint["step"], "optimizer_steps": steps,
              "training_image_observations": observations, "unique_training_images_seen": int(seen.sum()),
              "equivalent_dataset_passes": observations / len(images["train"]),
              "elapsed_seconds": time.perf_counter() - start,
              "initial_validation": initial, "validation": checkpoint["validation"],
              "training_subset": train_eval, "training_subset_count": min(256, len(images["train"])),
              "selection": "minimum validation sampled-code MSE + final KL weight * KL",
              "parameter_counts": {"encoder": sum(p.numel() for p in model.encoder.parameters()),
                                   "decoder": sum(p.numel() for p in model.decoder.parameters()),
                                   "circuit": sum(p.numel() for p in model.decoder.circuit.parameters()) if model.decoder.circuit is not None else 0},
              "training_coordinate_queries": observations * cfg.pixels_per_image,
              "quantum_execution": "exact classical simulation; no finite shots or hardware"}
    write_json(directory / "metrics.json", result)
    return best_path, result


def cache_latents(cfg, images, vae_path, run_dir, device):
    model, _ = load_vae(vae_path, device)
    model.requires_grad_(False)
    posteriors = {name: encode_images(model, values, cfg.batch_size, device) for name, values in images.items()}
    normalization = posterior_normalization(posteriors["train"])
    cache = {"format_version": 1, "kind": "qvf_posterior_cache", "posteriors": posteriors,
             "normalization": normalization, "vae_sha256": sha256(vae_path),
             "latent_policy": "fresh posterior samples for VAE; means for deterministic AE",
             "normalization_source": "training posterior mixture only"}
    torch.save(cache, run_dir / "latents.pt")
    write_json(run_dir / "latent_statistics.json", {
        "mean": normalization["mean"].tolist(), "std": normalization["std"].tolist(),
        "active_mean_dimensions_variance_gt_0.01": int((posteriors["train"]["mean"].var(0, unbiased=False) > 0.01).sum()),
        "normalization_source": cache["normalization_source"], "latent_policy": cache["latent_policy"],
        "vae_sha256": cache["vae_sha256"],
    })
    return cache


def draw_codes(posterior, indices, normalization, generator):
    mean = posterior["mean"][indices]
    logvar = posterior["logvar"][indices] if posterior["logvar"] is not None else None
    return (sample_posterior(mean, logvar, generator=generator) - normalization["mean"]) / normalization["std"]


@torch.no_grad()
def evaluate_diffusion(model, diffusion, posterior, normalization, batch_size, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    # Four fixed posterior/noise draws per validation image reduce evaluation noise.
    for _ in range(4):
        for indices in torch.arange(len(posterior["mean"]), device=device).split(batch_size):
            clean = draw_codes(posterior, indices, normalization, generator)
            timesteps = torch.randint(diffusion.timesteps, (len(indices),), device=device, generator=generator)
            noise = torch.randn(clean.shape, device=device, generator=generator)
            noisy = diffusion.q_sample(clean, timesteps, noise)
            loss = F.mse_loss(model(noisy, timesteps), diffusion.target(clean, timesteps, noise))
            total += loss.item() * len(indices)
            count += len(indices)
    return total / count


def train_diffusion(cfg, cache, vae_path, run_dir, device):
    if cache["vae_sha256"] != sha256(vae_path):
        raise ValueError("Latent cache and VAE checkpoint do not match.")
    directory = run_dir / "diffusion"
    directory.mkdir()
    denoiser_config = {"latent_dim": cfg.model.latent_dim, "width": cfg.diffusion_width,
                       "layers": cfg.diffusion_layers, "timesteps": cfg.diffusion_timesteps}
    torch.manual_seed(cfg.seed + 100)
    model = LatentDenoiser(**denoiser_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = LatentDDPM(cfg.diffusion_timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.diffusion_lr, weight_decay=0.0001)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 101)
    posteriors = {name: {key: value.to(device) if value is not None else None for key, value in posterior.items()}
                  for name, posterior in cache["posteriors"].items()}
    normalization = {key: value.to(device) for key, value in cache["normalization"].items()}
    initial = evaluate_diffusion(ema, diffusion, posteriors["validation"], normalization,
                                 cfg.diffusion_batch_size, device, cfg.seed + 1003)
    best_loss, best_step = float("inf"), None
    start = time.perf_counter()
    running, running_steps = 0.0, 0
    # Embedding the trained autoencoder makes the generation checkpoint portable.
    vae = torch.load(vae_path, map_location="cpu", weights_only=False)
    autoencoder = {"model_config": vae["model_config"], "model_state": vae["model_state"], "source_sha256": cache["vae_sha256"]}
    with open(directory / "history.jsonl", "x") as history:
        for step in range(1, cfg.diffusion_steps + 1):
            model.train()
            indices = torch.randint(len(posteriors["train"]["mean"]), (cfg.diffusion_batch_size,), device=device, generator=generator)
            clean = draw_codes(posteriors["train"], indices, normalization, generator)
            timesteps = torch.randint(diffusion.timesteps, (len(indices),), device=device, generator=generator)
            noise = torch.randn(clean.shape, device=device, generator=generator)
            noisy = diffusion.q_sample(clean, timesteps, noise)
            loss = F.mse_loss(model(noisy, timesteps), diffusion.target(clean, timesteps, noise))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite diffusion loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            decay = min(cfg.ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
                    ema_parameter.lerp_(parameter, 1 - decay)
            running += loss.item()
            running_steps += 1
            if step == 1 or step % cfg.diffusion_eval_every == 0 or step == cfg.diffusion_steps:
                validation = evaluate_diffusion(ema, diffusion, posteriors["validation"], normalization,
                                                cfg.diffusion_batch_size, device, cfg.seed + 1003)
                improved = validation < best_loss
                if improved:
                    best_loss, best_step = validation, step
                checkpoint = {"format_version": 1, "kind": "qvf_latent_diffusion", "prediction_type": "v",
                              "schedule": "cosine", "variance": "fixed posterior", "denoiser_config": denoiser_config,
                              "run_config": asdict(cfg), "step": step, "best_step": best_step,
                              "model_state": model.state_dict(), "ema_state": ema.state_dict(),
                              "optimizer_state": optimizer.state_dict(), "normalization": normalization,
                              "autoencoder": autoencoder, "validation_v_mse": validation, "rng_state": generator.get_state()}
                save_checkpoint(directory / "checkpoints" / "last.pt", checkpoint)
                if improved:
                    save_checkpoint(directory / "checkpoints" / "best.pt", checkpoint)
                record = {"step": step, "training_v_mse": running / running_steps, "validation_v_mse": validation}
                history.write(json.dumps(record, allow_nan=False) + "\n")
                history.flush()
                running, running_steps = 0.0, 0
                print(f"Diffusion step {step}/{cfg.diffusion_steps}: validation v-MSE {validation:.5f}", flush=True)
    result = {"initial_validation_v_mse": initial, "best_validation_v_mse": best_loss, "best_step": best_step,
              "optimizer_steps": cfg.diffusion_steps, "elapsed_seconds": time.perf_counter() - start,
              "parameters": sum(p.numel() for p in model.parameters()), "prediction_type": "v",
              "vae_sha256": cache["vae_sha256"], "qvf_calls_during_diffusion_training": 0,
              "note": "Denoising validation loss is not an image generation quality metric."}
    write_json(directory / "metrics.json", result)
    return directory / "checkpoints" / "best.pt", result


@torch.no_grad()
def generate(checkpoint_path, output_dir, device, count=64, seed=1234):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Sample directory already exists: {output_dir}")
    if count < 1:
        raise ValueError("count must be positive")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "qvf_latent_diffusion" or checkpoint.get("format_version") != 1:
        raise ValueError("Not a supported Project A diffusion checkpoint")
    if checkpoint["prediction_type"] != "v" or checkpoint["schedule"] != "cosine":
        raise ValueError("Unsupported diffusion parameterization")
    output_dir.mkdir(parents=True, exist_ok=False)
    model = LatentDenoiser(**checkpoint["denoiser_config"]).to(device).eval()
    model.load_state_dict(checkpoint["ema_state"])
    diffusion = LatentDDPM(model.timesteps).to(device)
    vae_cfg = ModelConfig.from_saved(checkpoint["autoencoder"]["model_config"])
    vae = QVFAutoencoder(vae_cfg).to(device).eval()
    vae.load_state_dict(checkpoint["autoencoder"]["model_state"])
    mean, std = (checkpoint["normalization"][name].to(device) for name in ("mean", "std"))
    generator = torch.Generator(device=device).manual_seed(seed)
    normalized_codes = diffusion.sample(model, count, vae_cfg.latent_dim, generator)
    codes = normalized_codes * std + mean
    samples = torch.cat([vae.render(batch).cpu() for batch in codes.split(16)])
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Non-finite generated images")
    gaussian = torch.randn(codes.shape, device=device, generator=generator)
    gaussian_images = torch.cat([vae.render(batch).cpu() for batch in (gaussian * std + mean).split(16)])
    prior_images = torch.cat([vae.render(batch).cpu() for batch in gaussian.split(16)])
    columns = min(8, count)
    save_image(samples, output_dir / "diffusion_samples.png", nrow=columns)
    save_image(gaussian_images, output_dir / "fitted_gaussian_samples.png", nrow=columns)
    save_image(prior_images, output_dir / "standard_normal_samples.png", nrow=columns)
    torch.save({"images": samples, "latent_codes": codes.cpu(), "normalized_codes": normalized_codes.cpu()}, output_dir / "samples.pt")
    write_json(output_dir / "metadata.json", {
        "seed": seed, "count": count, "checkpoint_sha256": sha256(checkpoint_path),
        "vae_sha256": checkpoint["autoencoder"]["source_sha256"],
        "diffusion_timesteps": model.timesteps, "prediction_type": "v",
        "normalized_code_mean": normalized_codes.mean(0).cpu().tolist(),
        "normalized_code_std": normalized_codes.std(0, unbiased=False).cpu().tolist(),
        "sample_pixel_std": samples.std(0, unbiased=False).mean().item(),
        "image_range": [samples.min().item(), samples.max().item()],
    })
    return samples


def snapshot_sources(run_dir):
    manifest = {}
    paths = sorted(HERE.glob("*.py")) + [HERE.parent / "qvf_repro" / "qvf_fast.py"]
    for path in paths:
        relative = path.relative_to(REPO)
        target = run_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[str(relative)] = sha256(path)
    write_json(run_dir / "source_manifest.json", manifest)


def run_pipeline(cfg, run_dir, data_root=DEFAULT_DATA, device_name="cuda:0"):
    cfg.validate()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        device = configure_runtime(device_name, cfg.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        write_json(run_dir / "config.json", {**asdict(cfg), "data_root": str(Path(data_root).resolve()), "device": device_name})
        snapshot_sources(run_dir)
        write_json(run_dir / "environment.json", {"torch": torch.__version__, "numpy": np.__version__,
                                                  "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                                                  "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")})
        images, indices = load_data(data_root, cfg.class_idx, cfg.n_train, cfg.n_val, cfg.split_seed)
        write_json(run_dir / "split.json", {"dataset": "CIFAR10", "source": "official training set",
                                            "official_test_set_used": False, "indices": indices})
        vae_path, vae_metrics = train_vae(cfg, images, run_dir, device)
        cache = cache_latents(cfg, images, vae_path, run_dir, device)
        diffusion_path, diffusion_metrics = train_diffusion(cfg, cache, vae_path, run_dir, device)
        samples = generate(diffusion_path, run_dir / "generation", device, cfg.sample_count, cfg.seed + 1004)
        distances = torch.cdist(samples.flatten(1), images["train"].flatten(1)).square() / samples[0].numel()
        nearest_mse, nearest = distances.min(dim=1)
        count = min(16, len(samples))
        save_image(torch.cat((samples[:count], images["train"][nearest[:count]])),
                   run_dir / "generation" / "nearest_training_images.png", nrow=count)
        write_json(run_dir / "generation" / "nearest_training_images.json", {
            "training_dataset_indices": [indices["train"][index] for index in nearest.tolist()],
            "pixel_mse": nearest_mse.tolist(), "note": "Pixel nearest neighbors are a visual diagnostic, not a proof against memorization."})
        result = {"status": "complete", "elapsed_seconds": time.perf_counter() - started,
                  "vae": vae_metrics, "diffusion": diffusion_metrics,
                  "generation_checkpoint": str(diffusion_path.relative_to(run_dir)),
                  "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  "scope": "single-run pilot, no hardware or quantum-advantage claim"}
        write_json(run_dir / "COMPLETE.json", result)
        print(f"Complete: {run_dir}", flush=True)
        return result
    except BaseException as error:
        write_json(run_dir / "FAILED.json", {"error": str(error), "type": type(error).__name__})
        raise
