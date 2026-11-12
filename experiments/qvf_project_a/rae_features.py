"""Cache frozen RAE decoder features before its RGB projection, without fitting."""

import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from .rae_assets import DEFAULT_ASSETS, sha256, verify_assets, write_json
from .rae_baseline import DEFAULT_DATA, DEFAULT_REFERENCE, HERE, configure, load_model, load_split, native_reconstruction

DEFAULT_CACHE = HERE / "data/20261109_v8_rae_features"
DEFAULT_BASELINE = HERE / "results/20261108_v7_rae_pretrained"


@torch.inference_mode()
def capture_features(model, images):
    """The hook observes pre-RGB hidden features; RGB outputs are audit-only."""
    captured = []
    handle = model.decoder.decoder_pred.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach()))
    try:
        output = model(images)
    finally:
        handle.remove()
    if len(captured) != 1 or tuple(captured[0].shape[1:]) != (257, 1152):
        raise AssertionError("Unexpected decoder hidden-feature layout")
    tokens = captured[0][:, 1:]
    features = tokens.transpose(1, 2).reshape(len(images), 1152, 16, 16).contiguous()
    # Independently replay the existing RGB projection from captured features.
    rgb = model.decoder.unpatchify(model.decoder.decoder_pred(tokens))
    rgb = rgb * model.encoder_std.to(rgb.device) + model.encoder_mean.to(rgb.device)
    torch.testing.assert_close(rgb, output, rtol=0, atol=2e-6)
    return features, native_reconstruction(output), (rgb - output).abs().max().item()


@torch.inference_mode()
def build_cache(args):
    args.output.mkdir(parents=True, exist_ok=False)
    device = configure(args.device)
    assets = verify_assets(args.assets)
    images, indices = load_split(args.reference, args.data_root)
    model = load_model(args.assets, device)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("RAE must remain frozen")
    start, max_replay, max_reference = time.perf_counter(), 0.0, 0.0
    manifest = {"kind": "frozen_rae_pre_rgb_features", "format_version": 1,
        "assets": assets, "assets_directory": str(args.assets.resolve()),
        "reference_directory": str(args.reference.resolve()), "baseline_directory": str(args.baseline.resolve()),
        "data_root": str(args.data_root.resolve()), "indices": indices,
        "shape_per_image": [1152, 16, 16], "dtype": "float32", "batch_size": args.batch_size,
        "tap": "decoder_pred input after decoder_norm; CLS removed, no RGB in feature tensor",
        "frozen_encoder": True, "frozen_decoder_body": True, "optimizer_steps": 0,
        "official_test_set_used": False, "groups": {}}
    for group in ("train", "validation"):
        baseline = torch.load(args.baseline / f"{group}.pt", map_location="cpu", weights_only=True)
        assert baseline["dataset_indices"] == indices[group]
        torch.testing.assert_close(baseline["originals"], images[group], rtol=0, atol=0)
        tensor = torch.empty((len(images[group]), 1152, 16, 16), dtype=torch.float32)
        last_log = time.perf_counter()
        for offset in range(0, len(tensor), args.batch_size):
            batch = images[group][offset:offset + args.batch_size].to(device)
            features, prediction, replay_error = capture_features(model, batch)
            tensor[offset:offset + len(batch)] = features.cpu()
            expected = baseline["reconstructions"][offset:offset + len(batch)]
            error = (prediction.cpu() - expected).abs().max().item()
            torch.testing.assert_close(prediction.cpu(), expected, rtol=0, atol=2e-6)
            max_reference, max_replay = max(max_reference, error), max(max_replay, replay_error)
            if time.perf_counter() - last_log > 20 or offset + len(batch) == len(tensor):
                print(f"Cached {group} {offset + len(batch)}/{len(tensor)}; RAE RGB error={error}", flush=True)
                last_log = time.perf_counter()
        if not torch.isfinite(tensor).all():
            raise ValueError("Non-finite cached features")
        path = args.output / f"{group}.pt"
        torch.save({"features": tensor, "dataset_indices": indices[group]}, path)
        manifest["groups"][group] = {"sha256": sha256(path), "bytes": path.stat().st_size,
            "images": len(tensor), "baseline_sha256": sha256(args.baseline / f"{group}.pt")}
        del tensor
    manifest.update({"elapsed_seconds": time.perf_counter() - start,
        "max_original_rae_reconstruction_error": max_reference,
        "max_replayed_rgb_projection_error": max_replay})
    sources = args.output / "source_snapshot"
    sources.mkdir()
    manifest["source_hashes"] = {}
    for name in ("rae_features.py", "rae_baseline.py", "rae_assets.py"):
        shutil.copy2(HERE / name, sources / name)
        manifest["source_hashes"][name] = sha256(sources / name)
    write_json(args.output / "manifest.json", manifest)
    print(f"Feature cache complete: {args.output}", flush=True)


def load_cache(root, device, verify_hashes=True):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["kind"] != "frozen_rae_pre_rgb_features" or manifest["format_version"] != 1:
        raise ValueError("Unexpected cache format")
    values = {}
    for group in ("train", "validation"):
        path = root / f"{group}.pt"
        if verify_hashes and sha256(path) != manifest["groups"][group]["sha256"]:
            raise ValueError(f"Changed feature cache: {group}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["dataset_indices"] != manifest["indices"][group]:
            raise ValueError("Feature IDs do not match manifest")
        tensor = payload["features"]
        if tensor.dtype != torch.float32 or list(tensor.shape[1:]) != manifest["shape_per_image"]:
            raise ValueError("Unexpected feature shape or precision")
        values[group] = tensor.to(device)
    return values, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    build_cache(args)
