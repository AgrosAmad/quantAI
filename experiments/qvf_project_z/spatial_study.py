"""Project Z4: spatial-slot encodings and five/seven-qubit comparisons."""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10

from experiments.qvf_project_a.direct_qvf import reconstruction_metrics, render_direct
from experiments.qvf_project_a.edge_qvf import (edge_objective, masked_gradient_metrics,
                       rgb_importance_weights, sample_pairs)
from experiments.qvf_project_a.models import pixel_coordinates
from experiments.qvf_project_a.token_study import fit_completion
from experiments.qvf_project_a.train import (HERE, REPO, configure_runtime, save_checkpoint, sha256,
                    snapshot_sources as snapshot_a, write_json)

from .spatial_models import build_model, model_specs, resources
ALL = tuple(n for n, _ in model_specs())
LATENT = ALL
STAGES = {"selected": ALL}

V10 = HERE / "results/20261109_v10_edges"


@dataclass
class ZTrainConfig:
    architecture: str = "attention_chain_q5"
    edge_weight: float = 0.5
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
        if self.architecture not in {a for group in STAGES.values() for a in group} or self.edge_weight < 0:
            raise ValueError("Invalid architecture or edge weight")
        if min(self.n_train, self.steps, self.batch_size, self.pairs_per_orientation, self.eval_every) < 1:
            raise ValueError("Counts must be positive")
        if self.n_train < self.batch_size or min(self.model_lr, self.code_lr, self.code_init_std) <= 0 or self.code_norm_weight < 0:
            raise ValueError("Invalid batch or optimization settings")


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("kind") != "qvf_project_z_spatial_v1":
        raise ValueError("Not an interface-study checkpoint")
    cfg = ZTrainConfig(**saved["config"])
    model = build_model(cfg.architecture, cfg.seed).to(device)
    model.load_state_dict(saved["decoder_state"], strict=True)
    return model.eval(), saved["codes"].to(device), saved


def train_model(cfg, images, ids, directory, device):
    cfg.validate()
    if len(images) != cfg.n_train or len(ids) != cfg.n_train:
        raise ValueError("Training data does not match the declared size")
    directory = Path(directory)
    fit = directory / "fit"
    fit.mkdir()
    model = build_model(cfg.architecture, cfg.seed).to(device)
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
            payload = {"kind": "qvf_project_z_spatial_v1", "config": asdict(cfg), "step": step,
                "epoch": epoch, "decoder_state": model.state_dict(), "codes": codes,
                "dataset_indices": ids, "visits": visits, "fit": short_metrics,
                "optimizer_model": optimizer_model.state_dict(), "optimizer_codes": optimizer_codes.state_dict(),
                "rng": {"order": order_rng.get_state(), "queries": query_rng.get_state()},
                "stream_sha256": stream.hexdigest(),
                "training_coordinate_queries": step * cfg.batch_size * 4 * cfg.pairs_per_orientation,
                "training_quantum_queries": step * cfg.batch_size * (4 * cfg.pairs_per_orientation)}
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
    return {"optimizer_steps": step,
        "training_quantum_queries": step * cfg.batch_size * (4 * cfg.pairs_per_orientation), "training_coordinate_queries": step * cfg.batch_size * 4 * cfg.pairs_per_orientation,
        "training_image_observations": step * cfg.batch_size, "visits_per_image": visits.tolist(),
        "stream_sha256": stream.hexdigest(), "elapsed_seconds": time.perf_counter() - start,
        "shared_parameters": sum(p.numel() for p in model.parameters()),
        "image_code_entries": codes.numel(), "final_fit": final_metrics,
        "checkpoint_sha256": {name: sha256(fit / "checkpoints" / f"{name}.pt") for name in sorted(checkpoints)}}


def snapshot_sources(root):
    import shutil
    snapshot_a(root)
    manifest = json.loads((root / "source_manifest.json").read_text())
    for source in sorted(Path(__file__).parent.rglob("*.py")):
        if "results" in source.parts:
            continue
        name = source.relative_to(REPO)
        dest = root / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        manifest[str(name)] = sha256(source)
    write_json(root / "source_manifest.json", manifest)


def create_protocol(root, seeds=(0, 1, 2), steps=10000, names=LATENT):
    if not seeds or len(set(seeds)) != len(seeds) or steps < 2000 or not names or any(n not in ALL for n in names):
        raise ValueError("Invalid protocol")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    previous = json.loads((V10 / "protocol.json").read_text())
    snapshot_sources(root)
    runs = []
    for seed in seeds:
        for name in names:
            cfg = ZTrainConfig(architecture=name, seed=seed, steps=steps)
            cfg.validate()
            model = build_model(name, seed)
            runs.append({"name": name, "stage": "selected", "directory": f"{name}_seed{seed}",
                         "config": asdict(cfg), **resources(model, name)})
    protocol = {"kind": "qvf_project_z_spatial_v1", "seeds": list(seeds), "runs": runs,
        "indices": previous["indices"], "completion": previous["completion"],
        "data_root": previous["data_root"], "v10_protocol_sha256": sha256(V10 / "protocol.json"),
        "source_sha256": json.loads((root / "source_manifest.json").read_text()),
        "question": "Spatial neighbour-to-wire encoding versus attention/amplitude encoding at five and seven qubits, retaining a continuous coordinate-to-RGB field.",
        "stage_order": ["selected"],
        "sampling": previous["sampling"], "pixel_objective": previous["pixel_objective"],
        "edge_objective": previous["edge_objective"], "edge_weight": .5,
        "optimization": previous["optimization"], "evaluation": previous["evaluation"],
        "initialization": "Fresh weights/codes. Same code, image and query streams. Common named classical layers and first five quantum wire weights paired; same shared filters. Five-qubit attention/chain reproduces completed Z3. Seven-qubit work-wire weights seeded separately.",
        "spatial_encoding": "Five slots: left, above, centre, below, right. Sample one grid-cell offset per scale with bilinear interpolation and border extension. Each slot sees 12 local values, 26 centre-coordinate positional features and two role offsets; shared 40->128->128->128->5 classical MLP predicts five RY re-upload angles. No classical mixing between slots in this MLP. Five data wires start in |+>; extra two at seven qubits start |0> and receive no data upload.",
        "local_quantum": "Five sweeps. RY on each member, CRY in each direction per pair; four filter angles shared over pairs in a sweep, then one RY per wire. Chain: even then odd neighbouring pairs. Star: centre wire2 with each other wire in ascending order. No pooling/reset; topology is logical, routing not assumed free.",
        "readout": "Fixed at both sizes: red=Z0+Z3, green=Z1+Z4, blue=Z2. Extra wires affect RGB only through their interactions; no extra measurement normalization benefit.",
        "controls": ["attention_star versus attention_chain isolates logical topology at fixed encoding and count", "spatial_star versus center_star isolates access to neighbouring latent features, at identical parameter count", "spatial five versus seven keeps the same five spatial slots and identical classical encoder, adds two work qubits", "five-qubit width148 attention_chain has at least the total/classical parameter count of standard width128 seven-qubit attention_chain"],
        "cost_matching": "336 fitted values/image, same data/queries/updates. Not equal FLOPs or gate budgets. Spatial branch evaluates the shared MLP five times/query versus once in amplitude branch. Seven-qubit amplitude preparation and final MLP are larger; explicit costs reported.",
        "primary_comparisons": ["spatial_star versus attention_chain at each qubit count", "spatial_star versus center_star at each count", "seven versus five within each family", "attention_chain_q7 versus attention_chain_wide_q5"],
        "limits": "Development pilot, not convergence or fresh final-test evidence. Excluded-image completion fits new codes to 256 observed pixels and scores 768 withheld pixels. No encoder/diffusion, no automatic unseen-image coding, no quantum advantage or large-scale claim. Circuit is ordered and positional features are absolute: no rotation or translation equivariance guarantee.",
        "official_test_set_used": False}
    write_json(root / "protocol.json", protocol)
    return root / "protocol.json"


def run_seed(root, seed, stage, device_name="cuda:0"):
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if seed not in protocol["seeds"] or stage not in protocol["stage_order"]:
        raise ValueError("Unknown seed/stage")
    for earlier in protocol["stage_order"][:protocol["stage_order"].index(stage)]:
        for spec in protocol["runs"]:
            if spec["stage"] == earlier and not (root / spec["directory"] / "COMPLETE.json").exists():
                raise RuntimeError(f"Earlier stage unfinished: {spec['directory']}")
    device = configure_runtime(device_name, seed)
    for source, digest in protocol["source_sha256"].items():
        if sha256(REPO / source) != digest:
            raise ValueError(f"Training source changed: {source}")
    dataset = CIFAR10(protocol["data_root"], train=True, download=False)
    ids, heldout_ids = protocol["indices"]["train"], protocol["completion"]["dataset_indices"]
    images = torch.from_numpy(dataset.data[ids].copy()).permute(0, 3, 1, 2).float() / 255
    heldout = torch.from_numpy(dataset.data[heldout_ids].copy()).permute(0, 3, 1, 2).float() / 255
    selected = [r for r in protocol["runs"] if r["stage"] == stage and r["config"]["seed"] == seed]
    # Concurrent seed jobs start different variants, so directions 1/2 run in parallel.
    shift = (2 * protocol["seeds"].index(seed)) % len(selected)
    selected = selected[shift:] + selected[:shift]
    for spec in selected:
        directory = root / spec["directory"]
        if (directory / "COMPLETE.json").exists():
            continue
        directory.mkdir(exist_ok=False)
        try:
            configure_runtime(device_name, seed)
            cfg = ZTrainConfig(**spec["config"])
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            write_json(directory / "config.json", asdict(cfg))
            write_json(directory / "environment.json", {"torch": torch.__version__,
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})
            print(f"Starting {spec['directory']}", flush=True)
            result = train_model(cfg, images, ids, directory, device)
            model, _, _ = load_model(directory / "fit/checkpoints/last.pt", device)
            completion = fit_completion(model, heldout, heldout_ids, protocol["completion"], seed, directory / "final", device)
            payload = torch.load(directory / "final/completion.pt", map_location="cpu", weights_only=False)
            completion["unobserved_edges"] = masked_gradient_metrics(payload["reconstructions"], heldout, protocol["completion"]["unobserved_pixels"])
            payload["metrics"] = completion
            save_checkpoint(directory / "final/completion.pt", payload)
            result.update({"status": "complete", "name": spec["name"], "seed": seed,
                "stage": stage, "resources": resources(model, cfg.architecture), "completion": completion,
                "protocol_sha256": sha256(root / "protocol.json"),
                "completion_sha256": sha256(directory / "final/completion.pt"),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0})
            write_json(directory / "COMPLETE.json", result)
            print(f"Complete {spec['directory']}: completion PSNR {completion['unobserved']['psnr_db']:.3f}, edge {completion['unobserved_edges']['gradient_mse']:.7f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except BaseException as error:
            write_json(directory / "FAILED.json", {"error": str(error), "type": type(error).__name__})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("protocol")
    make.add_argument("--root", type=Path, required=True)
    make.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    make.add_argument("--steps", type=int, default=10000)
    make.add_argument("--names", nargs="+", default=list(LATENT), choices=ALL)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--stage", choices=["selected"], required=True)
    run.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "protocol":
        print(create_protocol(args.root, args.seeds, args.steps, args.names))
    else:
        run_seed(args.root, args.seed, args.stage, args.device)


