"""V10 fixed-budget 2x2 study: all-scale quantum stages and supervised edges."""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10

from .direct_qvf import reconstruction_metrics, render_direct
from .edge_qvf import (build_edge_decoder, edge_objective, masked_gradient_metrics,
                       rgb_importance_weights, sample_pairs)
from .models import pixel_coordinates
from .token_study import fit_completion
from .train import (HERE, REPO, configure_runtime, save_checkpoint, sha256,
                    snapshot_sources, write_json)

V9 = HERE / "results/20261109_v9_tokens_pilot"


@dataclass
class EdgeConfig:
    architecture: str = "attention"
    edge_weight: float = 0.0
    seed: int = 0
    n_train: int = 1024
    steps: int = 10000
    batch_size: int = 16
    pairs_per_orientation: int = 32
    eval_every: int = 2000
    model_lr: float = 0.001
    code_lr: float = 0.001
    code_init_std: float = 0.01
    code_norm_weight: float = 0.0001

    def validate(self):
        if self.architecture not in {"attention", "staged_all"} or self.edge_weight < 0:
            raise ValueError("Invalid architecture or edge weight")
        if min(self.n_train, self.steps, self.batch_size, self.pairs_per_orientation, self.eval_every) < 1:
            raise ValueError("Counts must be positive")
        if self.n_train < self.batch_size or min(self.model_lr, self.code_lr, self.code_init_std) <= 0 or self.code_norm_weight < 0:
            raise ValueError("Invalid batch or optimization settings")


def load_edge(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("kind") != "qvf_edges_v1":
        raise ValueError("Not an edge-study checkpoint")
    cfg = EdgeConfig(**saved["config"])
    model = build_edge_decoder(cfg.architecture, cfg.seed).to(device)
    model.load_state_dict(saved["decoder_state"], strict=True)
    return model.eval(), saved["codes"].to(device), saved


def train_edge(cfg, images, ids, directory, device):
    cfg.validate()
    if len(images) != cfg.n_train or len(ids) != cfg.n_train:
        raise ValueError("Training data does not match the declared size")
    directory = Path(directory)
    fit = directory / "fit"
    fit.mkdir()
    model = build_edge_decoder(cfg.architecture, cfg.seed).to(device)
    code_rng = torch.Generator(device=device).manual_seed(cfg.seed)
    codes = torch.nn.Parameter(torch.randn(cfg.n_train, 336, generator=code_rng,
        device=device) * cfg.code_init_std)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=cfg.model_lr)
    optimizer_codes = torch.optim.Adam([codes], lr=cfg.code_lr)
    order_rng = torch.Generator().manual_seed(cfg.seed + 11)
    query_rng = torch.Generator().manual_seed(cfg.seed + 12)
    coordinates = pixel_coordinates(32, 32, device=device)
    target = images.flatten(2).transpose(1, 2).contiguous().to(device)
    weights = rgb_importance_weights().to(device)
    visits = torch.zeros(cfg.n_train, dtype=torch.int64)
    stream = hashlib.sha256()
    evaluations = {0, 1, cfg.steps, *range(cfg.eval_every, cfg.steps + 1, cfg.eval_every)}
    if cfg.steps >= 2000:
        evaluations.add(2000)
    step = 0
    last_components = None
    start = time.perf_counter()
    with (fit / "history.jsonl").open("x") as history:
        def evaluate(epoch):
            prediction = render_direct(model, codes, cfg.batch_size)
            metrics = reconstruction_metrics(prediction, images)
            short_metrics = {k: v for k, v in metrics.items() if k != "per_image_psnr_db"}
            record = {"step": step, "epoch": epoch, "fit": short_metrics,
                      "sampled_components": last_components}
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()
            payload = {"kind": "qvf_edges_v1", "config": asdict(cfg), "step": step,
                "epoch": epoch, "decoder_state": model.state_dict(), "codes": codes,
                "dataset_indices": ids, "visits": visits, "fit": short_metrics,
                "optimizer_model": optimizer_model.state_dict(), "optimizer_codes": optimizer_codes.state_dict(),
                "rng": {"order": order_rng.get_state(), "queries": query_rng.get_state()},
                "stream_sha256": stream.hexdigest(),
                "training_coordinate_queries": step * cfg.batch_size * 4 * cfg.pairs_per_orientation}
            save_checkpoint(fit / "checkpoints/last.pt", payload)
            if step in {0, 2000}:
                save_checkpoint(fit / "checkpoints" / f"step{step}.pt", payload)
            print(f"{cfg.architecture} edge={cfg.edge_weight:g} seed={cfg.seed} step={step}/{cfg.steps}: "
                  f"PSNR {metrics['psnr_db']:.3f}, edge MSE {metrics['gradient_mse']:.7f}", flush=True)
            return prediction, metrics

        evaluate(0)
        for epoch in range(1, math.ceil(cfg.steps / (cfg.n_train // cfg.batch_size)) + 1):
            for rows in torch.randperm(cfg.n_train, generator=order_rng).split(cfg.batch_size):
                if len(rows) != cfg.batch_size:
                    continue
                points = sample_pairs(query_rng, cfg.pairs_per_orientation)
                stream.update(rows.numpy().tobytes())
                stream.update(points.numpy().tobytes())
                gpu_rows, gpu_points = rows.to(device), points.to(device)
                selected = codes[gpu_rows]
                model.train()
                optimizer_model.zero_grad(set_to_none=True)
                optimizer_codes.zero_grad(set_to_none=True)
                prediction = model(selected, coordinates[gpu_points])
                terms = edge_objective(prediction, target[gpu_rows][:, gpu_points], weights[gpu_points],
                                       selected, cfg.edge_weight, cfg.code_norm_weight)
                if not torch.isfinite(terms[0]):
                    raise FloatingPointError(f"Non-finite loss at step {step + 1}")
                terms[0].backward()
                optimizer_model.step()
                optimizer_codes.step()
                step += 1
                visits[rows] += 1
                if step in evaluations:
                    last_components = dict(zip(("total", "pixel", "edge", "code_penalty"),
                                               [t.detach().item() for t in terms]))
                    final_prediction, final_metrics = evaluate(epoch)
                if step == cfg.steps:
                    break
    final = directory / "final"
    final.mkdir()
    save_checkpoint(final / "train.pt", {"originals": images, "reconstructions": final_prediction,
                                         "dataset_indices": ids})
    write_json(final / "metrics.json", final_metrics)
    checkpoints = {"last", "step0"} | ({"step2000"} if cfg.steps >= 2000 else set())
    return {"optimizer_steps": step, "training_coordinate_queries": step * cfg.batch_size * 4 * cfg.pairs_per_orientation,
        "training_image_observations": step * cfg.batch_size, "visits_per_image": visits.tolist(),
        "stream_sha256": stream.hexdigest(), "elapsed_seconds": time.perf_counter() - start,
        "shared_parameters": sum(p.numel() for p in model.parameters()),
        "image_code_entries": codes.numel(), "final_fit": final_metrics,
        "checkpoint_sha256": {name: sha256(fit / "checkpoints" / f"{name}.pt") for name in sorted(checkpoints)}}


def create_protocol(root, seeds, steps, edge_weight):
    if len(set(seeds)) != len(seeds) or not seeds or steps < 2000 or edge_weight <= 0:
        raise ValueError("Use unique seeds, at least 2000 steps and a positive edge weight")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    previous = json.loads((V9 / "protocol.json").read_text())
    completion = {**previous["completion"], "steps": 1000,
        "scope": "Same pixel-only code fitting for every arm; train loss variants do not change inference objective. Score edges only when both endpoints are unobserved."}
    snapshot_sources(root)
    runs = []
    for seed in seeds:
        for architecture in ("attention", "staged_all"):
            for weight in (0.0, edge_weight):
                cfg = EdgeConfig(architecture=architecture, edge_weight=weight, seed=seed, steps=steps)
                cfg.validate()
                name = architecture + ("_pixel" if weight == 0 else "_edge")
                model = build_edge_decoder(architecture, seed)
                runs.append({"name": name, "directory": f"{name}_seed{seed}", "config": asdict(cfg),
                    "shared_parameters": sum(p.numel() for p in model.parameters()),
                    "qubits": 5, "code_entries_per_image": 336,
                    "effective_trainable_circuit_angles": 125,
                    "input_dependent_rotations": 10 if architecture == "staged_all" else 0})
    protocol = {"kind": "qvf_edges_v1", "runs": runs, "seeds": seeds,
        "indices": previous["indices"], "data_root": previous["data_root"],
        "source_sha256": json.loads((root / "source_manifest.json").read_text()),
        "v9_protocol_sha256": sha256(V9 / "protocol.json"),
        "question": "Do full initial scale conditioning and an explicit edge objective improve edge accuracy beyond attention at fixed qubits and image-code storage?",
        "sampling": "16 images per update; 32 uniform horizontal and 32 uniform vertical pairs, with replacement. Evaluate all 128 endpoints including duplicates, identically across all arms. CPU RNG independent of model initialization.",
        "pixel_objective": "Mean RGB squared error over endpoints weighted by mean_pixel_degree / pixel_degree. This inverse-probability correction makes the sampled pixel loss unbiased for uniform full-image MSE; do not divide by sampled weight sum.",
        "edge_objective": "Mean squared signed neighbour-difference error; equal orientation counts. No division by coordinate spacing. Pixel MSE + lambda*edge_MSE + 0.0001*mean code L2 norm; lambda fixed before training, not optimized by held-out scores.",
        "initialization": "Fresh codes and common attention/amplitude/circuit parameters paired within seed. Additional stage rotation maps are exactly zero initially, so the initial RGB function agrees up to arithmetic roundoff. No V9 learned weights are reused.",
        "optimization": "Separate Adam optimizers for decoder/codes, lr0.001 constant, no clipping or schedule. Final fixed-budget checkpoint primary; step2000 saved for within-protocol duration comparison.",
        "completion": completion,
        "evaluation": "Full training RGB MSE/PSNR and finite-difference edge error. Excluded-image completion PSNR on unobserved pixels and edge error on unobserved-unobserved pairs. Fixed first 12 image IDs for galleries. No held-out model selection.",
        "resource_scope": "Pinned quantai environment, GPU0 only, up to three seed processes. Same query count is not equal gate/FLOP count; staged circuits add ten data rotations and fifty classical map parameters.",
        "limits": "1024-image fitting study; code optimization for new images, no encoder/diffusion. One edge weight; not an optimal-loss or quantum-advantage claim. Changed sampler means V9 scores are not matched duration controls. Exact simulator, no hardware/shots.",
        "official_test_set_used": False}
    write_json(root / "protocol.json", protocol)
    return root / "protocol.json"


def run_seed(root, seed, device_name):
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if seed not in protocol["seeds"]:
        raise ValueError("Seed not in saved protocol")
    # Initialize CUDA and thread settings before data conversion.
    device = configure_runtime(device_name, seed)
    for source, digest in protocol["source_sha256"].items():
        if sha256(REPO / source) != digest:
            raise ValueError(f"Training source changed: {source}")
    dataset = CIFAR10(protocol["data_root"], train=True, download=False)
    ids = protocol["indices"]["train"]
    heldout_ids = protocol["completion"]["dataset_indices"]
    images = torch.from_numpy(dataset.data[ids].copy()).permute(0, 3, 1, 2).float() / 255
    heldout = torch.from_numpy(dataset.data[heldout_ids].copy()).permute(0, 3, 1, 2).float() / 255
    for spec in [r for r in protocol["runs"] if r["config"]["seed"] == seed]:
        directory = root / spec["directory"]
        if (directory / "COMPLETE.json").exists():
            continue
        directory.mkdir(exist_ok=False)
        try:
            configure_runtime(device_name, seed)
            cfg = EdgeConfig(**spec["config"])
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            write_json(directory / "config.json", asdict(cfg))
            write_json(directory / "environment.json", {"torch": torch.__version__,
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})
            print(f"Starting {spec['directory']}", flush=True)
            result = train_edge(cfg, images, ids, directory, device)
            model, _, _ = load_edge(directory / "fit/checkpoints/last.pt", device)
            completion = fit_completion(model, heldout, heldout_ids, protocol["completion"],
                                        seed, directory / "final", device)
            payload = torch.load(directory / "final/completion.pt", map_location="cpu", weights_only=False)
            completion["unobserved_edges"] = masked_gradient_metrics(payload["reconstructions"], heldout,
                                                                      protocol["completion"]["unobserved_pixels"])
            payload["metrics"] = completion
            save_checkpoint(directory / "final/completion.pt", payload)
            result.update({"status": "complete", "name": spec["name"], "seed": seed,
                "completion": completion, "protocol_sha256": sha256(root / "protocol.json"),
                "completion_sha256": sha256(directory / "final/completion.pt"),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0})
            write_json(directory / "COMPLETE.json", result)
            print(f"Complete {spec['directory']}; held-out edge MSE "
                  f"{completion['unobserved_edges']['gradient_mse']:.7f}, PSNR {completion['unobserved']['psnr_db']:.3f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except BaseException as error:
            write_json(directory / "FAILED.json", {"error": str(error), "type": type(error).__name__})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("protocol")
    make.add_argument("--root", type=Path, required=True)
    make.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    make.add_argument("--steps", type=int, default=10000)
    make.add_argument("--edge-weight", type=float, default=0.5)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "protocol":
        print(create_protocol(args.root, args.seeds, args.steps, args.edge_weight))
    else:
        run_seed(args.root, args.seed, args.device)


if __name__ == "__main__":
    main()
