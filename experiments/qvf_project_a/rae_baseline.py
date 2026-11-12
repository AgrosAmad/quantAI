"""Evaluate the published, frozen DINOv2-B RAE on the existing CIFAR dog split.

This is a zero-shot transfer/quality reference, not a retraining of the RAE paper,
a 64-dimensional compression baseline, or a matched quantum/classical ablation.
The upstream encoder and decoder implementations are imported without edits.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

from .rae_assets import DEFAULT_ASSETS, DECODER_FILE, STATS_FILE, sha256, verify_assets, write_json


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_REFERENCE = HERE / "results/20261108_v4_capacity_baseline50k_seed0"
DEFAULT_DATA = HERE.parent / "qvf_repro/data"


def configure(device):
    if device not in {"cpu", "cuda:0"}:
        raise ValueError("Only CPU or GPU 0 may be used for this experiment")
    if device == "cuda:0":
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "0"}:
            raise ValueError("CUDA_VISIBLE_DEVICES must be 0")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device(device)


def load_model(assets, device):
    """Use the official modules; enforce complete checkpoint loading ourselves."""
    assets = Path(assets).resolve()
    sys.path.insert(0, str(assets / "official/src"))
    from stage1 import RAE

    model = RAE(
        encoder_cls="Dinov2withNorm", encoder_config_path=str(assets / "encoder"),
        encoder_input_size=224,
        encoder_params={"dinov2_path": str(assets / "encoder"), "normalize": True},
        decoder_config_path=str(assets / "official/configs/decoder/ViTXL"),
        decoder_patch_size=16, pretrained_decoder_path=None, noise_tau=0.0,
        reshape_to_2d=True, normalization_stat_path=None,
    )
    state = torch.load(assets / "models" / DECODER_FILE, map_location="cpu", weights_only=True)
    model.decoder.load_state_dict(state, strict=True)
    del state
    stats = torch.load(assets / "models" / STATS_FILE, map_location="cpu", weights_only=True)
    model.latent_mean, model.latent_var = stats["mean"], stats["var"]
    model.eps, model.do_normalization = 1e-5, True
    model.requires_grad_(False).eval().to(device)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("The pretrained reference must be completely frozen")
    return model


def native_reconstruction(output):
    """Score unrounded, unclipped output at the dataset's native resolution."""
    if output.ndim != 4 or output.shape[1:] != (3, 256, 256):
        raise ValueError(f"Expected published decoder's 256x256 RGB output, got {output.shape}")
    return F.interpolate(output, size=(32, 32), mode="bicubic", align_corners=False, antialias=True)


def metrics(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Predictions and targets must have matching NCHW shapes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Non-finite image data")
    error = prediction.double() - target.double()
    mse = error.square().flatten(1).mean(1)
    psnr = -10 * mse.clamp_min(1e-12).log10()
    gradient = 0.5 * (
        (error[:, :, 1:] - error[:, :, :-1]).square().flatten(1).mean(1)
        + (error[:, :, :, 1:] - error[:, :, :, :-1]).square().flatten(1).mean(1)
    )
    return {"images": len(target), "mse": mse.mean().item(), "psnr_db": psnr.mean().item(),
            "gradient_mse": gradient.mean().item(), "per_image_mse": mse.tolist(),
            "per_image_psnr_db": psnr.tolist()}


def load_split(reference, data_root):
    saved = json.loads((Path(reference) / "split.json").read_text())
    indices = saved["indices"]
    if set(indices) != {"train", "validation"}:
        raise ValueError("Unexpected split groups")
    train, validation = indices["train"], indices["validation"]
    if len(train) != 4744 or len(validation) != 256 or len(set(train + validation)) != 5000:
        raise ValueError("Expected the established disjoint 4744/256 split")
    dataset = CIFAR10(str(data_root), train=True, download=False)
    images = {}
    for group, ids in indices.items():
        if any(i < 0 or i >= len(dataset) or dataset.targets[i] != 5 for i in ids):
            raise ValueError("Split contains invalid IDs or non-dog images")
        images[group] = torch.from_numpy(dataset.data[ids].copy()).permute(0, 3, 1, 2).float() / 255
    return images, indices


def save_examples(path, originals, reconstructed, previous=None, count=16):
    """Fixed first examples, nearest-neighbor display; never select by quality."""
    groups = [("Original", originals), ("DINO-RAE (pretrained)", reconstructed)]
    if previous is not None:
        groups.append(("Earlier QVF-VAE", previous))
    cell, label_width, margin = 96, 195, 8
    columns = 8
    blocks = (min(count, len(originals)) + columns - 1) // columns
    canvas = Image.new("RGB", (label_width + columns * (cell + margin), blocks * (len(groups) * (cell + 6) + 20)), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 14) if Path(font_path).exists() else ImageFont.load_default()
    for block in range(blocks):
        for row, (label, data) in enumerate(groups):
            y = block * (len(groups) * (cell + 6) + 20) + row * (cell + 6)
            draw.text((8, y + cell // 2 - 8), label, fill="black", font=font)
            for col in range(columns):
                index = block * columns + col
                if index >= min(count, len(originals)):
                    break
                pixels = (data[index].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                tile = Image.fromarray(pixels).resize((cell, cell), Image.Resampling.NEAREST)
                canvas.paste(tile, (label_width + col * (cell + margin), y))
    canvas.save(path)


@torch.inference_mode()
def evaluate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = configure(args.device)
    start = time.perf_counter()
    assets = verify_assets(args.assets)
    images, indices = load_split(args.reference, args.data_root)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for name in ("rae_assets.py", "rae_baseline.py", "requirements-rae.txt"):
        shutil.copy2(HERE / name, snapshot / name)
        source_hashes[name] = sha256(snapshot / name)
    reference_files = [args.reference / name for name in ("split.json", "COMPLETE.json", "selected/train.pt", "selected/validation.pt")]
    protocol = {
        "kind": "published_dinov2_rae_zero_shot_transfer", "optimizer_steps": 0,
        "assets": assets, "assets_directory": str(args.assets.resolve()),
        "reference": str(args.reference.resolve()),
        "reference_file_hashes": {str(p.resolve()): sha256(p) for p in reference_files},
        "source_hashes": source_hashes, "indices": indices, "official_test_set_used": False,
        "data_root": str(args.data_root.resolve()), "device": str(device), "batch_size": args.batch_size,
        "precision": "float32; TF32 disabled for matmul and cuDNN; deterministic algorithms",
        "preprocessing": "RGB [0,1], official RAE bicubic 32->224 align_corners=False, no antialias; ImageNet mean/std",
        "encoder": "DINOv2-B with four registers; final affine LayerNorm removed per RAE; CLS/registers discarded",
        "latent": {"shape": [768, 16, 16], "entries": 196608,
                   "normalization": "published ImageNet stats, inverted inside decode", "noise_tau": 0},
        "decoder": "published ViT-XL (28 blocks, width 1152), patch16 output256; strict checkpoint load",
        "primary_scoring": "bicubic antialiased 256->32; unclipped/unrounded float outputs; mean per-image PSNR",
        "secondary_scoring": "same native outputs clamped to [0,1]",
        "selection": "No training, tuning or checkpoint selection; fixed released checkpoint and all split images",
        "limitations": ["External pretraining and much larger model/latent; not resource-matched to QVF",
                        "Local validation holdout does not certify absence from external pretraining data",
                        "CIFAR upsampling adds no information; this is transfer to 32px images",
                        "No RAE retraining, QVF change, diffusion training or generation"],
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "numpy", "transformers", "safetensors", "huggingface-hub", "tokenizers")},
    }
    write_json(output / "protocol.json", protocol)
    model = load_model(args.assets, device)
    counts = {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in ("encoder", "decoder")}
    print(f"Loaded frozen model: {counts}", flush=True)
    # A source-level cross-check: encode->decode must equal upstream forward.
    sample = images["train"][:2].to(device)
    latent = model.encode(sample)
    manual = model.decode(latent)
    upstream = model(sample)
    torch.testing.assert_close(manual, upstream, atol=0, rtol=0)
    if tuple(latent.shape) != (2, 768, 16, 16):
        raise AssertionError(f"Unexpected latent shape: {latent.shape}")
    torch.testing.assert_close(native_reconstruction(manual), native_reconstruction(upstream), atol=0, rtol=0)
    result = {"status": "complete", "optimizer_steps": 0, "parameters": counts,
              "latent_entries_per_image": latent[0].numel(), "groups": {}, "reference": {}}
    for group in ("validation", "train"):
        predictions = []
        begin = time.perf_counter()
        last_log = begin
        for offset in range(0, len(images[group]), args.batch_size):
            batch = images[group][offset:offset + args.batch_size].to(device)
            decoded = model(batch)
            predictions.append(native_reconstruction(decoded).cpu())
            now = time.perf_counter()
            if now - last_log > 20 or offset + len(batch) == len(images[group]):
                print(f"{group}: {offset + len(batch)}/{len(images[group])} images; {now - begin:.1f}s", flush=True)
                last_log = now
        predicted = torch.cat(predictions)
        # Interpolation-only reference exposes the cost of resolution conversion.
        # Process in small batches to avoid full-dataset high-resolution allocations.
        roundtrip = torch.cat([native_reconstruction(F.interpolate(
            F.interpolate(b, size=(224, 224), mode="bicubic", align_corners=False),
            size=(256, 256), mode="bicubic", align_corners=False))
            for b in images[group].split(32)])
        target = images[group]
        result["groups"][group] = {
            "raw": metrics(predicted, target), "clipped": metrics(predicted.clamp(0, 1), target),
            "resize_only": metrics(roundtrip, target), "elapsed_seconds": time.perf_counter() - begin,
        }
        previous = torch.load(args.reference / "selected" / f"{group}.pt", map_location="cpu", weights_only=True)
        if previous["dataset_indices"] != indices[group] or not torch.equal(previous["originals"], target):
            raise AssertionError("Historical comparison is not using the same images")
        result["reference"][group] = metrics(previous["mean_reconstructions"], target)
        torch.save({"dataset_indices": indices[group], "originals": target, "reconstructions": predicted,
                    "resize_only": roundtrip}, output / f"{group}.pt")
        save_examples(output / f"{group}_examples.png", target, predicted, previous["mean_reconstructions"])
        print(f"{group} native PSNR: {result['groups'][group]['raw']['psnr_db']:.4f} dB", flush=True)
    result["elapsed_seconds"] = time.perf_counter() - start
    result["verification"] = {"strict_decoder_load": True, "upstream_forward_error": 0.0,
                              "all_parameters_frozen": True, "expected_latent_shape": True}
    write_json(output / "results.json", result)
    # Mark done only after all outputs are serialized; verification is a separate invocation.
    write_json(output / "COMPLETE.json", {"status": "complete", "results_sha256": sha256(output / "results.json")})
    print(f"Completed: {output}", flush=True)


@torch.inference_mode()
def verify(args):
    output = args.output.resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    result = json.loads((output / "results.json").read_text())
    complete = json.loads((output / "COMPLETE.json").read_text())
    if sha256(output / "results.json") != complete["results_sha256"]:
        raise AssertionError("Result manifest mismatch")
    verify_assets(Path(protocol["assets_directory"]))
    for name, expected in protocol["source_hashes"].items():
        assert sha256(output / "source_snapshot" / name) == expected
    for path, expected in protocol["reference_file_hashes"].items():
        assert sha256(path) == expected, path
    images, indices = load_split(Path(protocol["reference"]), Path(protocol["data_root"]))
    assert indices == protocol["indices"]
    device = configure(args.device)
    model = load_model(Path(protocol["assets_directory"]), device)
    maximum = 0.0
    for group in ("validation", "train"):
        payload = torch.load(output / f"{group}.pt", map_location="cpu", weights_only=True)
        assert payload["dataset_indices"] == indices[group]
        torch.testing.assert_close(payload["originals"], images[group], atol=0, rtol=0)
        prediction = payload["reconstructions"]
        # Independent NumPy metric calculation, rather than rerunning metrics().
        e = prediction.numpy().astype(np.float64) - images[group].numpy().astype(np.float64)
        mse = np.mean(e * e, axis=(1, 2, 3))
        psnr = -10 * np.log10(np.maximum(mse, 1e-12))
        gradient = .5 * (np.mean(np.diff(e, axis=2) ** 2) + np.mean(np.diff(e, axis=3) ** 2))
        actual = result["groups"][group]["raw"]
        np.testing.assert_allclose(mse, actual["per_image_mse"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(psnr, actual["per_image_psnr_db"], rtol=0, atol=1e-11)
        np.testing.assert_allclose([mse.mean(), psnr.mean(), gradient],
                                   [actual["mse"], actual["psnr_db"], actual["gradient_mse"]], rtol=0, atol=1e-11)
        # Every held-out image, plus a fixed training subset, is independently rerendered.
        count = len(images[group]) if group == "validation" else 32
        rendered = torch.cat([native_reconstruction(model(b.to(device))).cpu()
                              for b in images[group][:count].split(protocol["batch_size"])])
        error = (rendered - prediction[:count]).abs().max().item()
        maximum = max(maximum, error)
        torch.testing.assert_close(rendered, prediction[:count], rtol=0, atol=2e-6)
        print(f"Verified {group}: {len(mse)} raw metrics, {count} rerendered images, max error={error}", flush=True)
    audit = {"status": "passed", "all_5000_native_metrics_independently_recomputed": True,
             "all_256_validation_and_32_train_images_rerendered": True, "max_rgb_reload_error": maximum,
             "same_4744_256_split": True, "prior_reference_files_unchanged": True,
             "pinned_source_and_pretrained_weights_verified": True, "official_test_set_used": False,
             "optimizer_steps": 0}
    write_json(output / "verification.json", audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evaluate", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    (evaluate if args.action == "evaluate" else verify)(args)


if __name__ == "__main__":
    main()
