"""Train QVF (fast simulator) or the classical MLP baseline on a collection of CIFAR-10 images.

The training loop follows the official script (src/Images/QVF_for_image.py): one latent
code per image, two Adam optimisers (model and codes), loss = MSE + c * mean ||z||.

Presets:
  demo        the released config_img.yaml, unchanged (100 images, 4 qubits, depth 2)
  paper       the paper's 50-image collection experiment (Sec. 5.2, Table 3, App. B.2) with
              the released code's conventions: 5 qubits, depth 5, hidden width 128, 5,000
              epochs, Adam at 1e-3 with a plateau scheduler (factor 0.9, window 50).
  paper_text  the same, but following the paper's text where it differs from the released
              code: a factor of pi in the positional encoding (Eq. 1), the colour read from
              the first three qubits (Sec. 4.2) instead of the "parity" read-out, and a code
              penalty of 1e-3 (Sec. 5) instead of 1e-4.

The paper does not state the code length, the positional-encoding depth, the batch size,
the image subset or the code learning rate, so those are kept from the released config.

Examples:
    python train.py --preset demo --model qvf --init identity --seed 0
    python train.py --preset paper --model mlp --seed 0
"""

import argparse
import functools
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from qvf_fast import MLPImage, QVFImage, count_parameters, positional_embedding

HERE = Path(__file__).parent
print = functools.partial(print, flush=True)  # progress lines reach a redirected log as they happen

COMMON = {
    "latent_size": 64, "L_embed": 6, "num_hidden_layers_energy": 3,
    "skip_connection": False, "use_parity": True, "output_channels": 3,
    "gaussian_std": 0.1, "class_idx": 5,
    "pe_scale": 1.0,       # frequency scale of the positional encoding (official code: 1)
    "latent_reg": 1e-4,    # the official sigma_Rval = 0.01 enters the loss squared
}
PAPER = {**COMMON, "train_samples_use": 50, "train_epochs": 5000, "batch_size": 50,
         "lr_model": 1e-3, "lr_latent": 1e-3, "scheduler": {"factor": 0.9, "patience": 50},
         "hid_dim": 128, "out_dim": 32, "num_repetations": 5}
PRESETS = {
    "demo": {**COMMON, "train_samples_use": 100, "train_epochs": 400, "batch_size": 50,
             "lr_model": 0.01, "lr_latent": 0.01, "scheduler": None,
             "hid_dim": 256, "out_dim": 16, "num_repetations": 2},
    "paper": PAPER,
    "paper_text": {**PAPER, "pe_scale": math.pi, "use_parity": False, "latent_reg": 1e-3},
}


def run_name(preset, model, init, act, n_img, seed, tag=""):
    name = f"{preset}_{model}"
    if model == "qvf":
        name += f"_{init}"
    return name + f"_{act}_n{n_img}_seed{seed}{tag}"


def load_images(cfg, root):
    trainset = torchvision.datasets.CIFAR10(str(root), train=True, download=False)
    images = torch.tensor(trainset.data)[torch.tensor(trainset.targets) == cfg["class_idx"]] / 255.0
    first = cfg.get("first_image", 0)
    if first < 0 or cfg["train_samples_use"] < 1 or first + cfg["train_samples_use"] > len(images):
        raise ValueError("Requested image range is outside the selected CIFAR-10 class")
    return images[first:first + cfg["train_samples_use"]]


def parse_overrides(pairs):
    """key=value pairs for --set; values are read as JSON when possible (numbers, true/false)."""
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def make_coords(h, w):
    coords = torch.tensor([[i, j] for i in range(h) for j in range(w)], dtype=torch.float32) / h
    return coords


def model_inputs(latents, coord_feat):
    """Rows of [code, coordinate features] for every pixel of every image in the batch."""
    b, hw = latents.shape[0], coord_feat.shape[0]
    z = latents.unsqueeze(1).expand(-1, hw, -1)
    c = coord_feat.unsqueeze(0).expand(b, -1, -1)
    return torch.cat([z, c], dim=-1).reshape(b * hw, -1)


def evaluate(model, latent_codes, coord_feat, pixels):
    model.eval()
    with torch.no_grad():
        pred = model(model_inputs(latent_codes, coord_feat)).reshape(pixels.shape)
    mse = ((pred - pixels) ** 2).mean(dim=(1, 2))
    mse_clamped = ((pred.clamp(0, 1) - pixels) ** 2).mean(dim=(1, 2))
    return pred, mse, mse_clamped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default="paper")
    p.add_argument("--model", choices=["qvf", "mlp"], default="qvf")
    p.add_argument("--init", choices=["identity", "gaussian"], default="identity")
    p.add_argument("--act", choices=["relu", "sin"], default="relu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None, help="override the preset's epoch count")
    p.add_argument("--images", type=int, default=None, help="override the preset's image count")
    p.add_argument("--batch-size", type=int, default=None, help="override the preset's batch size (images)")
    p.add_argument("--depth", type=int, default=None,
                   help="override the circuit depth J; 0 removes the circuit and keeps the amplitude encoding and read-out")
    p.add_argument("--first-image", type=int, default=0, help="index of the first image of the class to use")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override any config entry, for example --set L_embed=10 pe_scale=3.141592653589793")
    p.add_argument("--tag", default="")
    p.add_argument("--output-dir", type=Path, default=HERE / "results")
    p.add_argument("--save-checkpoint", action="store_true", help="Archive final weights, fitted codes and raw reconstructions.")
    args = p.parse_args()

    cfg = dict(PRESETS[args.preset])
    cfg["use_identity_block"] = args.init == "identity"
    cfg["activation"] = args.act
    if args.epochs is not None:
        cfg["train_epochs"] = args.epochs
    if args.images is not None:
        cfg["train_samples_use"] = args.images
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.depth is not None:
        cfg["num_repetations"] = args.depth
    cfg["first_image"] = args.first_image
    overrides = parse_overrides(args.set)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise SystemExit(f"unknown config entries: {sorted(unknown)}")
    cfg.update(overrides)

    if min(cfg["train_epochs"], cfg["batch_size"], cfg["train_samples_use"]) < 1:
        raise ValueError("Epoch, batch and image counts must be positive")
    if args.device.startswith("cuda") and args.device != "cuda:0":
        raise ValueError("Only GPU 0 is authorized for this project")
    if args.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=0 for GPU execution")
    name = run_name(args.preset, args.model, args.init, args.act, cfg["train_samples_use"], args.seed, args.tag)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / f"{name}{suffix}").exists() for suffix in (".json", ".png", ".started.json")):
        raise FileExistsError(f"Run already exists: {name}. Choose a fresh tag or output directory.")
    if (out_dir / "checkpoints" / f"{name}.pt").exists():
        raise FileExistsError(f"Checkpoint already exists for {name}")
    started = {"name": name, "config": cfg, "seed": args.seed, "device": args.device,
               "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "torch": torch.__version__, "numpy": np.__version__, "training_scope": "direct fitting of the evaluation images"}
    source_dir = out_dir / "source_snapshot" / name
    source_dir.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for source in (Path(__file__).resolve(), HERE / "qvf_fast.py", HERE / "sweep.py"):
        manifest[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, source_dir / source.name)
    started["source_sha256"] = manifest
    (out_dir / f"{name}.started.json").write_text(json.dumps(started, indent=2))

    torch.set_num_threads(4)  # shared machine; the CPU work here is small-tensor bookkeeping
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    images = load_images(cfg, HERE / "data")
    n_img, h, w, _ = images.shape
    pixels = images.reshape(n_img, h * w, 3).to(device)
    coords = make_coords(h, w).to(device)
    coord_feat = positional_embedding(coords, cfg["L_embed"], scale=cfg["pe_scale"]) if cfg["L_embed"] else coords

    latent_codes = torch.normal(0.0, 0.01, size=(n_img, cfg["latent_size"]), device=device).requires_grad_(True)
    model = (QVFImage(cfg) if args.model == "qvf" else MLPImage(cfg)).to(device)

    opt_model = torch.optim.Adam(model.parameters(), lr=cfg["lr_model"])
    opt_latent = torch.optim.Adam([latent_codes], lr=cfg["lr_latent"])
    schedulers = []
    if cfg["scheduler"]:
        schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(o, mode="min", **cfg["scheduler"])
                      for o in (opt_model, opt_latent)]

    print(f"{name}: {count_parameters(model)} model parameters"
          + (f", {count_parameters(model.quantum)} in the circuit" if args.model == "qvf" else "")
          + f", {latent_codes.numel()} code values, device {device}")

    history = []
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    sync()
    t0 = time.perf_counter()
    for epoch in range(cfg["train_epochs"]):
        model.train()
        epoch_loss, epoch_mse, n_batches = 0.0, 0.0, 0
        for idx in torch.randperm(n_img, device=device).split(cfg["batch_size"]):
            x = model_inputs(latent_codes[idx], coord_feat)
            y = pixels[idx].reshape(-1, 3)
            opt_model.zero_grad()
            opt_latent.zero_grad()
            pred = model(x)
            mse = F.mse_loss(pred, y)
            reg = cfg["latent_reg"] * torch.mean(torch.linalg.norm(x[:, :cfg["latent_size"]], dim=1))
            if not torch.isfinite(mse + reg):
                raise FloatingPointError(f"Non-finite objective at epoch {epoch}")
            (mse + reg).backward()
            opt_model.step()
            opt_latent.step()
            epoch_loss += (mse + reg).item()
            epoch_mse += mse.item()
            n_batches += 1
        for s in schedulers:
            s.step(epoch_loss)
        if epoch % max(1, cfg["train_epochs"] // 200) == 0 or epoch == cfg["train_epochs"] - 1:
            history.append([epoch, epoch_mse / n_batches])
        if epoch % max(1, cfg["train_epochs"] // 10) == 0 or epoch == cfg["train_epochs"] - 1:
            print(f"  epoch {epoch:5d}  train mse {epoch_mse / n_batches:.6f}  "
                  f"psnr {-10 * math.log10(epoch_mse / n_batches):.2f} dB  lr {opt_model.param_groups[0]['lr']:.2e}")
    sync()
    train_seconds = time.perf_counter() - t0

    pred, mse, mse_clamped = evaluate(model, latent_codes, coord_feat, pixels)
    psnr = -10 * torch.log10(mse)
    psnr_clamped = -10 * torch.log10(mse_clamped)
    trainset = torchvision.datasets.CIFAR10(str(HERE / "data"), train=True, download=False)
    class_indices = np.flatnonzero(np.asarray(trainset.targets) == cfg["class_idx"])
    dataset_indices = class_indices[cfg["first_image"]:cfg["first_image"] + n_img].tolist()
    spatial_error = (pred - pixels).reshape(n_img, h, w, 3)
    gradient_mse = 0.5 * ((spatial_error[:, 1:] - spatial_error[:, :-1]).square().flatten(1).mean(1)
                          + (spatial_error[:, :, 1:] - spatial_error[:, :, :-1]).square().flatten(1).mean(1))
    result = {
        "status": "complete", "dataset_indices": dataset_indices,
        "evaluation_scope": "all pixels of fitted training images; no held-out generalization test",
        "official_test_set_used": False, "optimizer_steps": cfg["train_epochs"] * math.ceil(n_img / cfg["batch_size"]),
        "training_coordinate_queries": cfg["train_epochs"] * n_img * h * w,
        "output_readout": "released parity grouping" if cfg["use_parity"] else "first three raw Pauli-Z expectations",
        "output_affine_rescaling": False,
        "name": name, "preset": args.preset, "model": args.model, "init": args.init if args.model == "qvf" else None,
        "activation": args.act, "seed": args.seed, "config": cfg, "device": str(device),
        "model_parameters": count_parameters(model),
        "circuit_parameters": count_parameters(model.quantum) if args.model == "qvf" else 0,
        "code_values": latent_codes.numel(),
        "train_seconds": train_seconds, "seconds_per_epoch": train_seconds / cfg["train_epochs"],
        "mse_mean": mse.mean().item(), "mse_std_over_images": mse.std().item() if n_img > 1 else 0.0,
        "psnr_of_mean_mse": -10 * math.log10(mse.mean().item()),
        "psnr_mean": psnr.mean().item(), "psnr_std_over_images": psnr.std().item() if n_img > 1 else 0.0,
        "code_norm_mean": latent_codes.detach().norm(dim=1).mean().item(),
        "final_lr": opt_model.param_groups[0]["lr"],
        "mse_clamped_mean": mse_clamped.mean().item(), "psnr_clamped_mean": psnr_clamped.mean().item(),
        "psnr_per_image": psnr.tolist(), "history_epoch_mse": history,
        "mse_per_image": mse.tolist(), "gradient_mse_mean": gradient_mse.mean().item(),
        "raw_prediction_min": pred.min().item(), "raw_prediction_max": pred.max().item(),
        "fraction_outside_rgb_range": ((pred < 0) | (pred > 1)).float().mean().item(),
        "source_sha256": manifest,
    }
    if args.save_checkpoint:
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save({"format_version": 1, "config": cfg, "seed": args.seed, "model": args.model,
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "latent_codes": latent_codes.detach().cpu(), "coordinates": coords.cpu(),
                    "dataset_indices": dataset_indices, "original_images": images,
                    "raw_reconstructions": pred.detach().cpu().reshape(n_img, h, w, 3),
                    "optimizer_model": opt_model.state_dict(), "optimizer_latent": opt_latent.state_dict(),
                    "source_sha256": manifest}, checkpoint_dir / f"{name}.pt")

    k = min(10, n_img)
    grid = torch.cat([images[:k].to(device), pred[:k].reshape(k, h, w, 3).clamp(0, 1)], dim=0)
    torchvision.utils.save_image(grid.permute(0, 3, 1, 2), out_dir / f"{name}.png", nrow=k, padding=1)
    if args.save_checkpoint:
        torchvision.utils.save_image(torch.cat([images, pred.detach().cpu().reshape(n_img, h, w, 3).clamp(0, 1)]).permute(0, 3, 1, 2),
                                     out_dir / f"{name}_all.png", nrow=min(10, n_img), padding=1)
    with (out_dir / f"{name}.json").open("x") as handle:
        json.dump(result, handle, indent=1, allow_nan=False)

    print(f"  done in {train_seconds:.1f} s ({1e3 * train_seconds / cfg['train_epochs']:.1f} ms per epoch)")
    print(f"  MSE {1e3 * result['mse_mean']:.3f}e-3   PSNR {result['psnr_mean']:.2f} dB per image "
          f"(std over images {result['psnr_std_over_images']:.2f}), {result['psnr_of_mean_mse']:.2f} dB from the mean MSE")


if __name__ == "__main__":
    main()
