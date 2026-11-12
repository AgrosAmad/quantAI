"""Reproducible token-QVF pilot: architecture, latent budget and qubit controls."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .data import load_data
from .direct_qvf import DirectConfig, render_direct, train_direct
from .models import pixel_coordinates
from .reconstruction import DEFAULT_REFERENCE
from .token_qvf import TokenConfig, build_token_decoder
from .train import (DEFAULT_DATA, REPO, configure_runtime, save_checkpoint,
                    sha256, snapshot_sources, write_json)


def model_configs():
    result = []
    for qubits in (5, 7):
        for name, retrieval, encoding, dim in (
            ("vanilla64", "global", "single", 64),
            ("global336", "global", "single", 336),
            ("bilinear", "bilinear", "single", 336),
            ("attention", "attention", "single", 336),
            ("attention_staged", "attention", "staged", 336),
        ):
            result.append((f"q{qubits}_{name}", TokenConfig(qubits=qubits,
                retrieval=retrieval, encoding=encoding, latent_dim=dim)))
    return result


def load_token(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "token_qvf" or checkpoint.get("format_version") != 1:
        raise ValueError("Not a token-study checkpoint")
    cfg = TokenConfig(**checkpoint["model_config"])
    model = build_token_decoder(cfg, checkpoint["run_config"]["seed"]).to(device)
    model.load_state_dict(checkpoint["decoder_state"], strict=True)
    return model.eval(), checkpoint["codes"].to(device), checkpoint


def masked_metrics(prediction, target, pixels):
    error = (prediction.flatten(2)[:, :, pixels] - target.flatten(2)[:, :, pixels]).double()
    mse = error.square().mean((1, 2))
    return {"mse": mse.mean().item(), "psnr_db": (-10 * mse.clamp_min(1e-12).log10()).mean().item(),
            "per_image_mse": mse.tolist(), "count": len(target), "pixels_per_image": len(pixels)}


def fit_completion(model, images, indices, spec, seed, directory, device):
    """Freeze decoder; optimize fresh codes ONLY against a fixed observed mask."""
    model.eval().requires_grad_(False)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    generator = torch.Generator(device=device).manual_seed(seed + 7001)
    codes = torch.nn.Parameter(torch.randn(len(images), model.cfg.latent_dim,
        generator=generator, device=device) * spec["code_init_std"])
    observed = torch.tensor(spec["observed_pixels"], device=device)
    unobserved = torch.tensor(spec["unobserved_pixels"])
    coordinates = pixel_coordinates(32, 32, device=device)[observed]
    # No unobserved target enters the loss, optimizer or checkpoint selection.
    targets = images.flatten(2).transpose(1, 2)[:, observed.cpu()].to(device)
    optimizer = torch.optim.Adam([codes], lr=spec["code_lr"])
    initial = codes.detach().cpu().clone()
    start = time.perf_counter()
    for _ in range(spec["steps"]):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(codes, coordinates)
        loss = (predictions - targets).square().mean() + spec["code_norm_weight"] * codes.norm(dim=1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite completion loss")
        loss.backward()
        optimizer.step()
    prediction = render_direct(model, codes, batch_size=16)
    assert all(torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
    result = {"observed": masked_metrics(prediction, images, observed.cpu()),
              "unobserved": masked_metrics(prediction, images, unobserved),
              "steps": spec["steps"], "decoder_frozen": True,
              "elapsed_seconds": time.perf_counter() - start}
    save_checkpoint(directory / "completion.pt", {"codes": codes, "initial_codes": initial,
        "originals": images, "reconstructions": prediction, "dataset_indices": indices,
        "observed_pixels": observed.cpu(), "unobserved_pixels": unobserved, "metrics": result})
    return result


def create_protocol(root, seeds, steps, n_train, completion_steps):
    root = Path(root).resolve()
    if n_train < 32 or n_train > 4744 or steps < 1 or completion_steps < 1:
        raise ValueError("Invalid pilot budget")
    root.mkdir(parents=True, exist_ok=False)
    reference = json.loads((DEFAULT_REFERENCE / "split.json").read_text())["indices"]
    _, indices = load_data(DEFAULT_DATA, 5, 4744, 256, 20261108, reference)
    indices["train"] = indices["train"][:n_train]
    heldout = indices["validation"][:32]
    observed = torch.randperm(1024, generator=torch.Generator().manual_seed(20261109)).tolist()
    snapshot_sources(root)
    sources = json.loads((root / "source_manifest.json").read_text())
    runs = []
    for seed in seeds:
        for name, model_cfg in model_configs():
            cfg = DirectConfig(model=model_cfg, seed=seed, n_train=n_train,
                batch_size=16, pixels_per_image=128, steps=steps,
                eval_every=max(1, steps // 4))
            cfg.validate()
            model = build_token_decoder(model_cfg, seed)
            runs.append({"name": name, "directory": f"{name}_seed{seed}", "config": asdict(cfg),
                "shared_parameters": sum(p.numel() for p in model.parameters()),
                "stored_circuit_parameters": sum(p.numel() for p in model.circuit.parameters()),
                "effective_circuit_angles": model_cfg.circuit_depth * model_cfg.qubits ** 2,
                "per_image_code_entries": model_cfg.latent_dim,
                "input_ry_gates": 2 * model_cfg.qubits if model_cfg.encoding == "staged" else 0})
    protocol = {"kind": "token_qvf_pilot_v1", "seeds": seeds, "runs": runs, "indices": indices,
        "data_root": str(DEFAULT_DATA.resolve()), "source_sha256": sources,
        "quantum_execution": "Exact real state-vector simulation, not hardware or finite shots",
        "selection": "Final fixed-budget checkpoint only; no held-out model selection",
        "primary_comparisons": ["attention vs bilinear at each fixed qubit count",
            "attention vs global336 at each fixed qubit count",
            "attention_staged vs attention at each fixed qubit count",
            "7 vs 5 qubits within each architecture"],
        "representation": "2x2,4x4,8x8 grids, four channels each: 336 fitted values/image. Fixed positions; dense classical attention with smooth distance bias. Native row/32,col/32 queries.",
        "readout": "The unchanged released parity_decode function at each size. At 5q: (Z0+Z3,Z1+Z4,Z2); at 7q: (Z0+Z3+Z6,Z1+Z4,Z2+Z5)/2. Qubit scaling also changes MLP output width, circuit gates and this prescribed normalization.",
        "controls": "Same seeds, data/image/pixel streams, updates, code initialization for equal code dimensions, value projections and common amplitude layers. Circuit angles paired within each qubit count. Global336 has more shared MLP parameters; attention adds small query/key maps to bilinear. These are not equal-FLOP controls.",
        "staged": "Original five repetitions split 2+2+1. Coarse amplitude preparation, intermediate/fine RY uploads controlled by pi*tanh(linear(features)); upload weights Gaussian std0.01, bias0. Same register and final readout, no reset.",
        "completion": {"dataset_indices": heldout, "steps": completion_steps, "code_lr": 0.001,
            "code_init_std": 0.01, "code_norm_weight": 0.0001,
            "observed_pixels": observed[:256], "unobserved_pixels": observed[256:],
            "scope": "Frozen decoder, new per-image codes fitted on 256 observed pixels; score 768 unseen pixels. Not encoder generalization."},
        "resource_scope": "Pinned quantai environment; GPU0 only. Two concurrent seed processes allowed; runtime is descriptive, not an isolated speed benchmark.",
        "limits": "Small two-seed pilot; not convergence, large-data validation, encoder/diffusion, hardware scaling or quantum advantage. No retrained circuit-free control in this pilot. Token semantics and frequency specialization are not enforced.",
        "official_test_set_used": False, "encoder_trained": False, "diffusion_trained": False}
    write_json(root / "protocol.json", protocol)
    return root / "protocol.json"


def run_seed(root, seed, device_name):
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if seed not in protocol["seeds"]:
        raise ValueError("Seed is absent from protocol")
    for source, digest in protocol["source_sha256"].items():
        if sha256(REPO / source) != digest:
            raise ValueError(f"Source changed after protocol: {source}")
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(protocol["data_root"], train=True, download=False)
    ids = protocol["indices"]["train"]
    heldout_ids = protocol["completion"]["dataset_indices"]
    images = torch.from_numpy(dataset.data[ids].copy()).permute(0, 3, 1, 2).float() / 255
    heldout = torch.from_numpy(dataset.data[heldout_ids].copy()).permute(0, 3, 1, 2).float() / 255
    for spec in [r for r in protocol["runs"] if r["config"]["seed"] == seed]:
        directory = root / spec["directory"]
        if (directory / "COMPLETE.json").exists():
            continue
        # A partial run is preserved, never silently overwritten or resumed with
        # a different optimizer/random stream. Completed runs can be skipped.
        directory.mkdir(exist_ok=False)
        try:
            cfg = DirectConfig(**{**spec["config"], "model": TokenConfig(**spec["config"]["model"])})
            device = configure_runtime(device_name, seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            write_json(directory / "config.json", asdict(cfg))
            write_json(directory / "split.json", protocol["indices"])
            write_json(directory / "environment.json", {"torch": torch.__version__,
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})
            print(f"Starting {spec['directory']}", flush=True)
            result = train_direct(cfg, images, ids, directory, device,
                decoder_factory=lambda c: build_token_decoder(c, seed), checkpoint_kind="token_qvf")
            model, _, _ = load_token(directory / "fit/checkpoints/last.pt", device)
            result["completion"] = fit_completion(model, heldout, heldout_ids,
                protocol["completion"], seed, directory / "final", device)
            result.update({"status": "complete", "name": spec["name"], "seed": seed,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                "protocol_sha256": sha256(root / "protocol.json")})
            write_json(directory / "COMPLETE.json", result)
            print(f"Complete {spec['directory']}; unobserved completion PSNR "
                  f"{result['completion']['unobserved']['psnr_db']:.3f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except BaseException as error:
            write_json(directory / "FAILED.json", {"type": type(error).__name__, "error": str(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("protocol")
    make.add_argument("--root", type=Path, required=True)
    make.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    make.add_argument("--steps", type=int, default=2000)
    make.add_argument("--n-train", type=int, default=1024)
    make.add_argument("--completion-steps", type=int, default=500)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "protocol":
        print(create_protocol(args.root, args.seeds, args.steps, args.n_train, args.completion_steps))
    else:
        run_seed(args.root, args.seed, args.device)


if __name__ == "__main__":
    main()
