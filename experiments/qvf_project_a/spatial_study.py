"""Fixed-budget spatial-feature vs parameter-matched pointwise QVF fitting.

Both arms reuse the direct-code trainer. The original QVF quantum circuit,
amplitude map and RGB measurement remain unchanged. No encoder is trained.
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .conv_qvf import LocalFeatureQVFDecoder
from .data import load_data
from .direct_qvf import DirectConfig, train_direct
from .models import CoordinateDecoder, ModelConfig
from .reconstruction import DEFAULT_REFERENCE
from .train import (DEFAULT_DATA, HERE, REPO, configure_runtime, sha256,
                    snapshot_sources, write_json)


@dataclass
class SpatialConfig(DirectConfig):
    architecture: str = "local_features"
    model: ModelConfig = field(default_factory=lambda: ModelConfig(mode="ae", field_layers=2))

    def validate(self):
        super().validate()
        if self.architecture not in {"local_features", "pointwise"}:
            raise ValueError("Unknown spatial-study architecture")
        if (self.model.latent_dim, self.model.qubits, self.model.circuit_depth,
                self.model.frequencies) != (64, 5, 5, 6):
            raise ValueError("The spatial study fixes code, qubits, depth and coordinates")
        expected = (128, 2) if self.architecture == "local_features" else (357, 3)
        if (self.model.field_width, self.model.field_layers) != expected:
            raise ValueError("Use the declared spatial or parameter-matched pointwise field")


def spatial_configs(seed, reference_run=DEFAULT_REFERENCE):
    return [
        ("conv50k", SpatialConfig(seed=seed)),
        ("matched50k", SpatialConfig(seed=seed, architecture="pointwise",
                                    model=ModelConfig(mode="ae", field_width=357))),
    ]


def build_decoder(cfg):
    cfg.validate()
    return (LocalFeatureQVFDecoder(cfg.model) if cfg.architecture == "local_features"
            else CoordinateDecoder(cfg.model))


def baseline_directory(seed):
    return HERE / "results" / f"20261108_v5_direct_qvf_large50k_seed{seed}"


def initial_circuit(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["kind"] != "direct_qvf" or checkpoint["step"] != 0:
        raise ValueError("Circuit initialization must come from the untrained V5 baseline")
    prefix = "field.quantum."
    state = {key[len(prefix):]: value for key, value in checkpoint["decoder_state"].items()
             if key.startswith(prefix)}
    if set(state) != {"weights", "cond"}:
        raise ValueError("Unexpected original quantum circuit state")
    return state


def load_spatial(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "spatial_qvf" or checkpoint.get("format_version") != 1:
        raise ValueError("Not a spatial QVF checkpoint")
    values = dict(checkpoint["run_config"])
    values["model"] = ModelConfig.from_saved(values["model"])
    cfg = SpatialConfig(**values)
    decoder = build_decoder(cfg).to(device)
    decoder.load_state_dict(checkpoint["decoder_state"])
    codes = torch.nn.Parameter(checkpoint["codes"].to(device))
    return decoder.eval(), codes, checkpoint


def run_spatial(cfg, run_dir, data_root=DEFAULT_DATA, device_name="cuda:0",
                reference_run=DEFAULT_REFERENCE, variant="conv50k"):
    cfg.validate()
    run_dir, reference_run = Path(run_dir).resolve(), Path(reference_run).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        device = configure_runtime(device_name, cfg.seed)
        # Explicitly retain the previous VAE/prototype convolution arithmetic.
        torch.backends.cudnn.allow_tf32 = True
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        reference = json.loads((reference_run / "split.json").read_text())["indices"]
        images, indices = load_data(data_root, cfg.class_idx, cfg.n_train, cfg.n_val,
                                   cfg.split_seed, reference)
        baseline = baseline_directory(cfg.seed)
        initial_path = baseline / "fit/checkpoints/step0.pt"
        circuit = initial_circuit(initial_path)
        write_json(run_dir / "config.json", {**asdict(cfg), "variant": variant,
            "device": device_name, "data_root": str(Path(data_root).resolve()),
            "reference_run": str(reference_run), "initial_circuit_checkpoint": str(initial_path),
            "initial_circuit_checkpoint_sha256": sha256(initial_path)})
        write_json(run_dir / "split.json", {"dataset": "CIFAR10", "official_test_set_used": False,
            "indices": indices, "validation_images_used_for_optimization_or_selection": False})
        snapshot_sources(run_dir)
        write_json(run_dir / "environment.json", {"torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()})

        def factory(model_config):
            decoder = build_decoder(cfg)
            decoder.circuit.load_state_dict(circuit)
            return decoder

        result = train_direct(cfg, images["train"], indices["train"], run_dir, device,
                              decoder_factory=factory, checkpoint_kind="spatial_qvf")
        result.update({"status": "complete", "kind": "spatial_qvf_fitting", "variant": variant,
            "architecture": cfg.architecture,
            "scope": "training-image fits with independent codes; no encoder, VAE or diffusion",
            "quantum_execution": "exact classical simulation; five qubits, depth five, original readout",
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0})
        write_json(run_dir / "COMPLETE.json", result)
        print(f"Complete: {run_dir}", flush=True)
        return result
    except BaseException as error:
        write_json(run_dir / "FAILED.json", {"error": str(error), "type": type(error).__name__})
        raise


def write_protocol(root, tag, seeds, reference_run=DEFAULT_REFERENCE):
    root, reference_run = Path(root).resolve(), Path(reference_run).resolve()
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    runs, baselines = [], []
    for seed in seeds:
        for label, cfg in spatial_configs(seed):
            cfg.validate()
            runs.append({"variant": label, "config": asdict(cfg),
                         "directory": str(root / f"{tag}_spatial_{label}_seed{seed}")})
        baseline = baseline_directory(seed)
        baselines.append({"seed": seed, "directory": str(baseline),
            "checkpoint_sha256": {name: sha256(baseline / "fit/checkpoints" / f"{name}.pt")
                                  for name in ("step0", "last")}})
    protocol = {
        "kind": "spatial_qvf_direct_fitting", "tag": tag, "seeds": seeds,
        "runs": runs, "baselines": baselines, "reference_run": str(reference_run),
        "reference_split_sha256": sha256(reference_run / "split.json"),
        "decoder_source_sha256": sha256(REPO / "experiments/qvf_repro/qvf_fast.py"),
        "question": "Do generated local spatial features improve fitting a larger image collection at similar decoder parameter count, with the original quantum head?",
        "primary_comparison": "Final conv50k vs matched50k, paired seeds; original width128 V5 is a secondary reference.",
        "parameters": {"conv50k": 299894, "matched50k": 299705, "baseline128": 48950},
        "spatial_decoder": "64-code -> 4x4x128 -> upsample/convolutions 8x8x64 -> upsample/convolutions 16x16x32 -> bilinear local features + 26 coordinates -> width128/two-hidden-layer amplitude MLP -> original QVF circuit/RGB",
        "pointwise_decoder": "64-code + 26 coordinates -> width357/three-hidden-layer amplitude MLP -> original QVF circuit/RGB",
        "fixed": "4744 training and 256 excluded validation IDs; 64-code; five qubits/depth five; original amplitude map and grouped Z RGB; 50000 updates; batch32; 256 pixel queries/image; identical order/pixel RNG seeds.",
        "initialization": "Fresh classical weights and code table. Identical initial codes and identical initial Gaussian circuit angles across both arms and the same-seed V5 baseline. No trained weights are reused.",
        "objective": "Pixel MSE + 0.0001 * mean L2 code norm. No VAE KL, perceptual/adversarial/edge loss, or posterior sampling.",
        "optimizer": "Identical original direct-code trainer: separate dense Adam optimizers at constant lr0.001, no gradient clipping or scheduler.",
        "selection": "Final 50000-step checkpoint primary; lowest fitting MSE saved separately. No validation-based selection.",
        "evaluation": "All1024 pixels of all4744 training images, raw unclipped predictions; mean per-image PSNR, pixel MSE, adjacent-pixel gradient MSE. Fixed256-image gallery.",
        "gallery_training_positions": torch.linspace(0, 4743, 256).round().long().tolist(),
        "runtime": "Record wall time and allocated GPU memory; parameter/update/query matches are not FLOP matches. GPU0 is shared with unrelated work, so wall time is descriptive, not an isolated speed benchmark.",
        "resource_scope": "Pinned quantai environment; GPU0 only; at most three training processes.",
        "limits": "One-class32x32 fitting experiment, not large-scale deployment, held-out inference, generation, convergence or quantum advantage. Architecture comparison changes local features, activation and capacity allocation together; does not isolate a single convolution operation.",
        "next_gates": "If the spatial decoder improves fits, test encoder/VAE generalization separately, then broaden data/resolution and measure hardware preparation/shot/query costs. A trained circuit ablation is needed before attributing gains to quantum processing.",
        "sources": {"qvf": "https://arxiv.org/html/2508.10900v2#S5",
                    "local_features_precedent": "https://arxiv.org/abs/2012.09161"},
        "diffusion_trained": False, "official_test_set_used": False}
    output = root / f"{tag}_spatial_protocol.json"
    with output.open("x") as handle:
        json.dump(protocol, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--results-root", type=Path, default=HERE / "results")
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        parser.error("Use a simple fresh tag")
    print(write_protocol(args.results_root, args.tag, args.seeds, args.reference_run))


if __name__ == "__main__":
    main()
