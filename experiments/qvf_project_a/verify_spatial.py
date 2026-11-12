"""Audit spatial-QVF fitting against its predeclared protocol and V5 reference."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .direct_qvf import load_direct, reconstruction_metrics, render_direct
from .spatial_study import load_spatial
from .train import DEFAULT_DATA, REPO, configure_runtime, sha256, write_json
from .verify_direct_qvf import expected_visits


KINDS = ("baseline128", "matched50k", "conv50k")


def read(path):
    return json.loads(Path(path).read_text())


def summary(values):
    return {"mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "per_seed": values}


def independent_metrics(prediction, target):
    error = prediction - target
    mse = error.pow(2).mean(dim=(1, 2, 3))
    psnr = -10 * torch.log10(mse.clamp_min(1e-12))
    vertical = torch.diff(prediction, dim=2) - torch.diff(target, dim=2)
    horizontal = torch.diff(prediction, dim=3) - torch.diff(target, dim=3)
    gradient = (vertical.square().mean() + horizontal.square().mean()) / 2
    return {"mse": mse.mean().item(), "psnr_db": psnr.mean().item(),
            "gradient_mse": gradient.item()}, psnr


def verify(protocol_path, output):
    protocol = read(protocol_path)
    assert protocol["kind"] == "spatial_qvf_direct_fitting"
    assert sha256(REPO / "experiments/qvf_repro/qvf_fast.py") == protocol["decoder_source_sha256"]
    assert sha256(Path(protocol["reference_run"]) / "split.json") == protocol["reference_split_sha256"]
    dataset = CIFAR10(str(DEFAULT_DATA), train=True, download=False)
    gallery = protocol["gallery_training_positions"]
    assert gallery == torch.linspace(0, 4743, 256).round().long().tolist()
    data = {"protocol": protocol, "runs": [], "images": []}
    maximum_reload_error = {kind: 0. for kind in KINDS}
    source_checks = []
    for seed in protocol["seeds"]:
        previous = next(row for row in protocol["baselines"] if row["seed"] == seed)
        baseline = Path(previous["directory"])
        for name, digest in previous["checkpoint_sha256"].items():
            assert sha256(baseline / "fit/checkpoints" / f"{name}.pt") == digest
        initial_reference = torch.load(baseline / "fit/checkpoints/step0.pt", map_location="cpu", weights_only=False)
        reference_last = torch.load(baseline / "fit/checkpoints/last.pt", map_location="cpu", weights_only=False)
        ids = read(baseline / "split.json")["indices"]
        assert len(ids["train"]) == len(set(ids["train"])) == 4744
        assert len(ids["validation"]) == len(set(ids["validation"])) == 256
        assert not set(ids["train"]) & set(ids["validation"])
        assert all(dataset.targets[i] == 5 for i in ids["train"] + ids["validation"])
        originals = torch.from_numpy(dataset.data[ids["train"]].copy()).permute(0, 3, 1, 2).float() / 255
        visits = expected_visits(seed, 4744, 32, 50000)
        row = {"seed": seed}
        predictions, scores = {}, {}
        for kind in KINDS:
            if kind == "baseline128":
                directory, loader = baseline, load_direct
                spec = None
            else:
                spec = next(r for r in protocol["runs"] if r["config"]["seed"] == seed and r["variant"] == kind)
                directory, loader = Path(spec["directory"]), load_spatial
            result = read(directory / "COMPLETE.json")
            cfg = read(directory / "config.json")
            split = read(directory / "split.json")
            if spec:
                assert all(cfg[key] == value for key, value in spec["config"].items())
                assert cfg["initial_circuit_checkpoint_sha256"] == previous["checkpoint_sha256"]["step0"]
                assert cfg["architecture"] == ("local_features" if kind == "conv50k" else "pointwise")
            assert result["status"] == "complete" and not (directory / "FAILED.json").exists()
            assert split["indices"] == ids and not split["official_test_set_used"]
            assert not split["validation_images_used_for_optimization_or_selection"]
            assert not (directory / "vae").exists() and not (directory / "diffusion").exists()
            model = cfg["model"]
            assert (model["latent_dim"], model["qubits"], model["circuit_depth"], model["frequencies"],
                    model["readout"], model["decoder"], model["mode"]) == (64, 5, 5, 6, "vanilla", "qvf", "ae")
            assert (model["field_width"], model["field_layers"]) == {
                "baseline128": (128, 3), "matched50k": (357, 3), "conv50k": (128, 2)}[kind]
            assert (cfg["batch_size"], cfg["pixels_per_image"], cfg["model_lr"], cfg["code_lr"],
                    cfg["code_norm_weight"], cfg["code_init_std"]) == (32, 256, .001, .001, .0001, .01)
            assert result["optimizer_steps"] == cfg["steps"] == 50000
            assert result["training_image_observations"] == 1600000
            assert result["training_coordinate_queries"] == 409600000
            assert result["visits_per_image"] == visits.tolist()
            assert result["parameter_counts"] == {"decoder": protocol["parameters"][kind],
                "circuit": 150, "image_codes": 303616, "encoder": 0}
            manifest = read(directory / "source_manifest.json")
            assert manifest["experiments/qvf_repro/qvf_fast.py"] == protocol["decoder_source_sha256"]
            for filename, expected in manifest.items():
                assert sha256(directory / "source_snapshot" / filename) == expected
            if spec:
                relevant = ["spatial_study.py", "conv_qvf.py", "direct_qvf.py", "models.py", "train.py", "data.py"]
                for filename in relevant:
                    relative = f"experiments/qvf_project_a/{filename}"
                    assert sha256(REPO / relative) == manifest[relative]
                source_checks.append({name: manifest[f"experiments/qvf_project_a/{name}"] for name in relevant})
            for name, digest in result["checkpoint_sha256"].items():
                assert sha256(directory / "fit/checkpoints" / f"{name}.pt") == digest
            initial = torch.load(directory / "fit/checkpoints/step0.pt", map_location="cpu", weights_only=False)
            torch.testing.assert_close(initial["codes"], initial_reference["codes"], atol=0, rtol=0)
            for key in ("field.quantum.weights", "field.quantum.cond"):
                torch.testing.assert_close(initial["decoder_state"][key], initial_reference["decoder_state"][key], atol=0, rtol=0)
            decoder, codes, checkpoint = loader(directory / "fit/checkpoints/last.pt", torch.device("cuda:0"))
            assert checkpoint["step"] == 50000 and checkpoint["dataset_indices"] == ids["train"]
            assert torch.equal(checkpoint["visits"], visits)
            for rng in ("order", "pixels"):
                assert torch.equal(checkpoint["rng"][rng], reference_last["rng"][rng])
            assert codes.shape == (4744, 64) and torch.isfinite(codes).all()
            assert not hasattr(decoder, "encoder")
            assert decoder.field.use_parity and decoder.circuit.n_qubits == decoder.circuit.repetitions == 5
            assert sum(p.numel() for p in decoder.parameters()) == protocol["parameters"][kind]
            for value in decoder.state_dict().values():
                assert torch.isfinite(value).all()
            circuit_changes = {key: (checkpoint["decoder_state"][key] - initial["decoder_state"][key]).abs().max().item()
                               for key in ("field.quantum.weights", "field.quantum.cond")}
            assert min(circuit_changes.values()) > 0
            payload = torch.load(directory / "final/train.pt", map_location="cpu", weights_only=False)
            assert payload["dataset_indices"] == ids["train"]
            torch.testing.assert_close(payload["originals"], originals, atol=0, rtol=0)
            prediction = payload["reconstructions"]
            assert prediction.shape == originals.shape and torch.isfinite(prediction).all()
            metrics = reconstruction_metrics(prediction, originals)
            assert metrics == result["final_fit"] == read(directory / "final/metrics.json")
            separate, per_image = independent_metrics(prediction, originals)
            for key, value in separate.items():
                assert abs(value - metrics[key]) < 1e-7
            with torch.no_grad():
                rebuilt = render_direct(decoder, codes, 32)
            error = (rebuilt - prediction).abs().max().item()
            assert error < 3e-6, (directory, error)
            maximum_reload_error[kind] = max(maximum_reload_error[kind], error)
            history = [json.loads(line) for line in (directory / "fit/history.jsonl").read_text().splitlines()]
            curve = [record for record in history if record["event"] == "fitting_evaluation"]
            assert [record["step"] for record in curve] == sorted({0, 1, 7680, *range(5000, 50001, 5000)})
            assert result["best_fitting_step"] == min(curve, key=lambda event: event["fit"]["mse"])["step"]
            predictions[kind], scores[kind] = prediction, per_image
            row[kind] = {"directory": str(directory),
                "metrics": {k: v for k, v in metrics.items() if k != "per_image_psnr_db"},
                "gallery_psnr_db": per_image[gallery].tolist(),
                "curve": [[r["step"], r["fit"]["psnr_db"], r["fit"]["gradient_mse"]] for r in curve],
                "elapsed_seconds": result["elapsed_seconds"],
                "peak_allocated_gpu_bytes": result["peak_allocated_gpu_bytes"],
                "best_fitting_step": result["best_fitting_step"], "parameter_counts": result["parameter_counts"],
                "max_circuit_angle_change": circuit_changes, "reload_max_rgb_error": error}
            del decoder, codes, checkpoint, rebuilt, initial
            print(f"Verified seed {seed} {kind}: {metrics['psnr_db']:.4f} dB; reload error {error:.3g}", flush=True)
        row["conv_minus_matched_psnr_db"] = row["conv50k"]["metrics"]["psnr_db"] - row["matched50k"]["metrics"]["psnr_db"]
        row["conv_minus_baseline_psnr_db"] = row["conv50k"]["metrics"]["psnr_db"] - row["baseline128"]["metrics"]["psnr_db"]
        row["fraction_training_images_conv_beats_matched_psnr"] = (scores["conv50k"] > scores["matched50k"]).float().mean().item()
        png = output / f"training_comparison_seed{seed}.png"
        save_image(torch.cat([originals[gallery], *[predictions[k][gallery] for k in KINDS]]), png, nrow=16, padding=1)
        webp = png.with_suffix(".webp")
        display = Image.open(png).convert("RGB")
        with webp.open("xb") as handle:
            display.save(handle, format="WEBP", lossless=True, method=6)
        assert Image.open(webp).convert("RGB").tobytes() == display.tobytes()
        data["images"].append({"seed": seed, "path": str(webp.resolve()),
            "training_positions": gallery, "dataset_indices": [ids["train"][i] for i in gallery]})
        data["runs"].append(row)
    assert all(item == source_checks[0] for item in source_checks)
    data["summary"] = {kind: {metric: summary([r[kind]["metrics"][metric] for r in data["runs"]])
                               for metric in ("psnr_db", "mse", "gradient_mse")}
                       for kind in KINDS}
    data["paired_changes"] = {key: summary([r[key] for r in data["runs"]])
                              for key in ("conv_minus_matched_psnr_db", "conv_minus_baseline_psnr_db")}
    data["verification"] = {"status": "passed", "new_runs": len(protocol["runs"]),
        "baseline_runs": len(protocol["baselines"]), "matching_training_ids": True,
        "full_dataset_metrics_independently_recomputed": True,
        "all_training_images_rerendered_from_all_checkpoints": True,
        "maximum_reload_rgb_error": maximum_reload_error,
        "same_initial_codes_and_circuit_angles": True,
        "same_order_and_pixel_rng_states": True, "visits_and_update_query_budgets_checked": True,
        "original_qvf_source_unchanged": True, "baseline_checkpoints_unchanged": True,
        "source_snapshots_and_live_training_sources_checked": True,
        "same_training_source_across_new_runs": True,
        "both_classical_and_quantum_components_trainable": True,
        "no_validation_in_fitting_or_selection": True,
        "gallery_predetermined": True, "lossless_webp_pixels_match_png": True}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_runtime("cuda:0", 0)
    torch.backends.cudnn.allow_tf32 = True
    data = verify(args.protocol, args.output)
    write_json(args.output / "results.json", data)
    write_json(args.output / "verification.json", data["verification"])
    print(json.dumps({"summary": data["summary"], "paired_changes": data["paired_changes"]}, indent=2))


if __name__ == "__main__":
    main()
