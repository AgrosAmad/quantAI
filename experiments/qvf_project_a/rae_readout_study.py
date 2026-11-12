"""Matched 50k-update QVF/classical readouts on frozen DINO-RAE features."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from .rae_assets import sha256, write_json
from .rae_baseline import HERE, configure, load_split, metrics
from .rae_features import DEFAULT_CACHE, load_cache
from .rae_qvf import RAEFeatureReadout, ReadoutConfig, render_readout

REPO = HERE.parents[1]


@dataclass
class TrainConfig:
    seed: int = 0
    steps: int = 50000
    batch_size: int = 32
    pixels_per_image: int = 256
    learning_rate: float = 0.001
    final_learning_rate: float = 0.00001
    warmup_steps: int = 500
    eval_every: int = 1000

    def validate(self):
        if min(self.steps, self.batch_size, self.pixels_per_image, self.eval_every) < 1:
            raise ValueError("Training budgets must be positive")
        if self.warmup_steps < 0 or not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError("Invalid learning-rate schedule")


def learning_rate(step, cfg):
    if cfg.warmup_steps and step <= cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.final_learning_rate + .5 * (cfg.learning_rate - cfg.final_learning_rate) * (1 + math.cos(math.pi * progress))


class BatchStream:
    """Private CPU RNGs: identical image/pixel order across readout architectures."""

    def __init__(self, count, image_size, cfg):
        if count < cfg.batch_size or cfg.pixels_per_image > image_size ** 2:
            raise ValueError("Batch/pixel count exceeds available data")
        self.count, self.image_size, self.cfg = count, image_size, cfg
        self.images_rng = torch.Generator().manual_seed(cfg.seed + 11)
        self.pixels_rng = torch.Generator().manual_seed(cfg.seed + 12)
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor = 0
        self.epoch = 0
        self.digest = "0" * 64

    def next(self):
        if self.cursor + self.cfg.batch_size > len(self.order):
            self.order = torch.randperm(self.count, generator=self.images_rng)
            self.cursor = 0
            self.epoch += 1
        rows = self.order[self.cursor:self.cursor + self.cfg.batch_size]
        self.cursor += len(rows)
        points = torch.randperm(self.image_size ** 2, generator=self.pixels_rng)[:self.cfg.pixels_per_image]
        self.digest = hashlib.sha256(bytes.fromhex(self.digest) + rows.numpy().tobytes() + points.numpy().tobytes()).hexdigest()
        return rows, points

    def state_dict(self):
        return {"images_rng": self.images_rng.get_state(), "pixels_rng": self.pixels_rng.get_state(),
                "order": self.order.clone(), "cursor": self.cursor, "epoch": self.epoch, "digest": self.digest}

    def load_state_dict(self, state):
        self.images_rng.set_state(state["images_rng"])
        self.pixels_rng.set_state(state["pixels_rng"])
        self.order, self.cursor, self.epoch, self.digest = state["order"], state["cursor"], state["epoch"], state["digest"]


def brief(prediction, target):
    return {k: v for k, v in metrics(prediction, target).items() if not isinstance(v, list)}


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_head(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["kind"] != "frozen_rae_readout" or payload["format_version"] != 1:
        raise ValueError("Unexpected readout checkpoint")
    model = RAEFeatureReadout(ReadoutConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval(), payload


def source_snapshot(output):
    names = ["experiments/qvf_project_a/" + name for name in
             ("rae_qvf.py", "rae_features.py", "rae_readout_study.py", "rae_baseline.py", "rae_assets.py")]
    names += ["experiments/qvf_repro/qvf_fast.py"]
    manifest = {}
    for name in names:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
        manifest[name] = sha256(target)
    return manifest


def initialize_study(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cache = json.loads((args.cache / "manifest.json").read_text())
    train_config = asdict(TrainConfig(steps=args.steps))
    protocol = {
        "kind": "frozen_rae_qvf_readout_comparison", "cache": str(args.cache.resolve()),
        "cache_manifest_sha256": sha256(args.cache / "manifest.json"),
        "cache_manifest": cache, "seeds": args.seeds, "variants": ["qvf", "classical"],
        "train_config": train_config, "model_config": asdict(ReadoutConfig()),
        "frozen_path": "RGB image -> pretrained DINO -> pretrained RAE decoder body -> post-norm pre-RGB patch features",
        "trainable_adapter": "1152->128 pointwise conv; SiLU; residual pair of 3x3 convolutions with SiLU",
        "query": "containing 16x16 patch feature + 2 within-patch pixel-centre offsets + 26 original global coordinate features",
        "qvf_head": "original three-layer width-128 amplitude MLP -> 32 amplitudes -> original 5-qubit depth-5 circuit -> grouped RGB",
        "classical_head": "same three-layer width-128 MLP -> 32 unconstrained features -> learned affine RGB projection",
        "matching": "identical adapter/MLP initial weights per seed; same features, IDs, image and pixel streams, loss and optimizer schedule",
        "loss": "native 32x32 RGB pixel MSE only; no KL, fitted image codes, teacher-output loss, adversarial or perceptual term",
        "optimizer": "Adam betas=(0.9,0.999), eps=1e-8, no weight decay, no clipping; linear warmup then cosine decay",
        "selection": "Final 50k checkpoint primary; also report minimum validation pixel MSE among scheduled evaluations",
        "evaluation": "all 256 validation images at step0, step1 and each 1000; fixed first256 training diagnostic; full train/validation at final and selected",
        "precision": "float32; deterministic algorithms; TF32 matmul disabled, cuDNN TF32 enabled as in prior train studies",
        "resource_scope": "GPU 0 only; pinned quantai and separate RAE venv unchanged",
        "official_test_set_used": False, "diffusion_trained": False,
        "limitations": ["Frozen features inherit large externally pretrained classical networks",
                        "Classical control matches the trained head package, not solely the circuit",
                        "Different feature and pretraining budgets from earlier small-code QVF experiments",
                        "Finite-budget validation results, not quantum advantage or hardware scaling"],
        "source_hashes": source_snapshot(args.output),
        "reference_file_hashes": {str(p.resolve()): sha256(p) for p in
            (Path(cache["baseline_directory"]) / "results.json", Path(cache["baseline_directory"]) / "verification.json")},
    }
    # Predeclare/check parameter matching without consuming any training RNG.
    protocol["parameter_counts"] = {}
    for kind in protocol["variants"]:
        torch.manual_seed(0)
        model = RAEFeatureReadout(ReadoutConfig(kind=kind))
        protocol["parameter_counts"][kind] = sum(p.numel() for p in model.parameters())
    write_json(args.output / "protocol.json", protocol)
    print(f"Protocol saved: {args.output / 'protocol.json'}", flush=True)


def train_readout(cfg, model_cfg, features, images, indices, output, cache_hash, *, resume=False):
    cfg.validate()
    device = features["train"].device
    output = Path(output)
    if resume:
        model, saved = load_head(output / "checkpoints/last.pt", device)
        if saved["train_config"] != asdict(cfg) or saved["model_config"] != asdict(model_cfg) or saved["cache_manifest_sha256"] != cache_hash:
            raise ValueError("Resume configuration/cache mismatch")
    else:
        output.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(cfg.seed)
        model = RAEFeatureReadout(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    sampler = BatchStream(len(images["train"]), model_cfg.image_size, cfg)
    visits = torch.zeros(len(images["train"]), dtype=torch.int64)
    steps, best_step, best_mse, earlier_elapsed = 0, None, float("inf"), 0.0
    if resume:
        optimizer.load_state_dict(saved["optimizer_state"])
        sampler.load_state_dict(saved["sampler"])
        visits = saved["visits"]
        steps, best_step, best_mse = saved["step"], saved["best_step"], saved["best_validation_mse"]
        earlier_elapsed = saved["elapsed_seconds"]
        # Keep only history through the resumable checkpoint; preserve later lines separately.
        history_path = output / "history.jsonl"
        rows = [json.loads(line) for line in history_path.read_text().splitlines()]
        future = [row for row in rows if row["step"] > steps]
        if future:
            write_json(output / f"interrupted_history_after_{steps}.json", future)
            history_path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= steps))
    targets = images["train"].flatten(2).transpose(1, 2).contiguous().to(device)
    start = time.perf_counter()
    write_json(output / "config.json", {"train": asdict(cfg), "model": asdict(model_cfg), "cache_manifest_sha256": cache_hash})
    history = (output / "history.jsonl").open("a" if resume else "x")

    def evaluate():
        nonlocal best_step, best_mse
        prediction = render_readout(model, features["validation"], cfg.batch_size)
        validation = brief(prediction, images["validation"])
        count = min(256, len(images["train"]))
        train_prediction = render_readout(model, features["train"][:count], cfg.batch_size)
        training = brief(train_prediction, images["train"][:count])
        if validation["mse"] < best_mse:
            best_mse, best_step = validation["mse"], steps
        elapsed = earlier_elapsed + time.perf_counter() - start
        record = {"event": "evaluation", "step": steps, "validation": validation,
                  "fixed_train_subset": training, "best_step": best_step, "elapsed_seconds": elapsed}
        payload = {"kind": "frozen_rae_readout", "format_version": 1,
                   "model_config": asdict(model_cfg), "train_config": asdict(cfg),
                   "cache_manifest_sha256": cache_hash, "model_state": model.state_dict(),
                   "optimizer_state": optimizer.state_dict(), "sampler": sampler.state_dict(),
                   "visits": visits, "step": steps, "best_step": best_step, "best_validation_mse": best_mse,
                   "dataset_indices": indices, "validation": validation, "elapsed_seconds": elapsed}
        save_checkpoint(output / "checkpoints/last.pt", payload)
        if best_step == steps:
            save_checkpoint(output / "checkpoints/best.pt", payload)
        if steps == 0:
            save_checkpoint(output / "checkpoints/step0.pt", payload)
        history.write(json.dumps(record, allow_nan=False) + "\n")
        history.flush()
        print(f"{model_cfg.kind} seed{cfg.seed} step {steps}/{cfg.steps}: validation PSNR {validation['psnr_db']:.4f}, "
              f"train subset {training['psnr_db']:.4f}; best step {best_step}; elapsed {elapsed:.1f}s", flush=True)

    try:
        if not resume:
            evaluate()
        while steps < cfg.steps:
            rows, points = sampler.next()
            rows_gpu, points_gpu = rows.to(device), points.to(device)
            lr = learning_rate(steps + 1, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features["train"][rows_gpu], points_gpu)
            loss = F.mse_loss(prediction, targets[rows_gpu][:, points_gpu])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at update {steps + 1}")
            loss.backward()
            optimizer.step()
            visits[rows] += 1
            steps += 1
            if steps == 1 or steps % cfg.eval_every == 0 or steps == cfg.steps:
                evaluate()
            elif steps % 250 == 0:
                history.write(json.dumps({"event": "sampled_training", "step": steps,
                    "pixel_mse": loss.item(), "learning_rate": lr}, allow_nan=False) + "\n")
                history.flush()
        elapsed = earlier_elapsed + time.perf_counter() - start
    finally:
        history.close()

    result = {"status": "complete", "kind": model_cfg.kind, "seed": cfg.seed, "optimizer_steps": steps,
              "best_step": best_step, "training_image_observations": steps * cfg.batch_size,
              "training_pixel_queries": steps * cfg.batch_size * cfg.pixels_per_image,
              "unique_training_images_seen": int((visits > 0).sum()), "batch_stream_digest": sampler.digest,
              "trainable_parameters": sum(p.numel() for p in model.parameters()), "elapsed_seconds": elapsed,
              "stages": {}, "checkpoints": {}, "artifacts": {}}
    for stage, name in (("final", "last"), ("selected", "best")):
        model, saved = load_head(output / "checkpoints" / f"{name}.pt", device)
        stage_path = output / stage
        stage_path.mkdir(exist_ok=resume)
        result["stages"][stage] = {"step": saved["step"]}
        for group in ("train", "validation"):
            prediction = render_readout(model, features[group], cfg.batch_size)
            scores = metrics(prediction, images[group])
            result["stages"][stage][group] = scores
            path = stage_path / f"{group}.pt"
            torch.save({"originals": images[group], "reconstructions": prediction,
                        "dataset_indices": indices[group]}, path)
            result["artifacts"][str(path.relative_to(output))] = sha256(path)
    for name in ("last", "best", "step0"):
        result["checkpoints"][name] = sha256(output / "checkpoints" / f"{name}.pt")
    write_json(output / "COMPLETE.json", result)
    print(f"Completed {output}: final validation {result['stages']['final']['validation']['psnr_db']:.4f} dB", flush=True)
    return result


def run(args):
    protocol = json.loads((args.output / "protocol.json").read_text())
    if args.seed not in protocol["seeds"] or args.kind not in protocol["variants"]:
        raise ValueError("Run not declared in protocol")
    for name, digest in protocol["source_hashes"].items():
        if sha256(REPO / name) != digest:
            raise ValueError(f"Source changed after protocol declaration: {name}")
    cache_path = Path(protocol["cache"])
    if sha256(cache_path / "manifest.json") != protocol["cache_manifest_sha256"]:
        raise ValueError("Feature manifest changed")
    device = configure(args.device)
    torch.backends.cudnn.allow_tf32 = True
    features, cache = load_cache(cache_path, device)
    images, indices = load_split(Path(cache["reference_directory"]), Path(cache["data_root"]))
    if indices != cache["indices"]:
        raise ValueError("Image/cache split mismatch")
    cfg = TrainConfig(**{**protocol["train_config"], "seed": args.seed})
    kinds = protocol["variants"] if args.action == "pair" else [args.kind]
    for kind in kinds:
        model_cfg = ReadoutConfig(**{**protocol["model_config"], "kind": kind})
        output = args.output / f"{kind}_seed{args.seed}"
        if args.resume and (output / "COMPLETE.json").exists():
            print(f"Already complete: {output}", flush=True)
            continue
        try:
            train_readout(cfg, model_cfg, features, images, indices, output,
                          protocol["cache_manifest_sha256"],
                          resume=args.resume and (output / "checkpoints/last.pt").exists())
        except BaseException as error:
            if output.exists():
                write_json(output / "FAILED.json", {"error": str(error), "type": type(error).__name__})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "pair"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=("qvf", "classical"), default="qvf")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "init":
        TrainConfig(steps=args.steps).validate()
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("Seeds must be unique")
        initialize_study(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
