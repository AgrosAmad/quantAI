"""Independent metrics, complete head reloads, and fresh-image RAE/QVF checks."""

import argparse
import json
from pathlib import Path
import shutil
import statistics

import numpy as np
import torch

from .rae_assets import sha256, verify_assets, write_json
from .rae_baseline import HERE, configure, load_model, load_split
from .rae_features import capture_features, load_cache
from .rae_qvf import render_readout
from .rae_readout_study import load_head

REPO = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def numpy_metrics(prediction, target):
    error = prediction.numpy().astype(np.float64) - target.numpy().astype(np.float64)
    mse = (error * error).mean(axis=(1, 2, 3))
    psnr = -10 * np.log10(np.maximum(mse, 1e-12))
    gradient = .5 * ((np.diff(error, axis=2) ** 2).mean() + (np.diff(error, axis=3) ** 2).mean())
    return {"mse": float(mse.mean()), "psnr_db": float(psnr.mean()), "gradient_mse": float(gradient),
            "per_image_mse": mse.tolist(), "per_image_psnr_db": psnr.tolist()}


def compare_metrics(actual, expected):
    for name, value in actual.items():
        np.testing.assert_allclose(value, expected[name], rtol=0, atol=1e-10, err_msg=name)


@torch.inference_mode()
def verify(args):
    root = args.study.resolve()
    output = root / "verified"
    output.mkdir(exist_ok=False)
    protocol = read(root / "protocol.json")
    cache_path = Path(protocol["cache"])
    assert sha256(cache_path / "manifest.json") == protocol["cache_manifest_sha256"]
    for name, digest in protocol["source_hashes"].items():
        assert sha256(root / "source_snapshot" / name) == digest
        assert sha256(REPO / name) == digest
    for path, digest in protocol["reference_file_hashes"].items():
        assert sha256(path) == digest
    cache = protocol["cache_manifest"]
    verify_assets(Path(cache["assets_directory"]))
    images, indices = load_split(Path(cache["reference_directory"]), Path(cache["data_root"]))
    assert indices == cache["indices"]
    device = configure(args.device)
    features, loaded_manifest = load_cache(cache_path, device)
    assert cache == loaded_manifest
    assert not set(indices["train"]) & set(indices["validation"])
    for group in ("train", "validation"):
        assert not features[group].requires_grad
        assert sha256(Path(cache["baseline_directory"]) / f"{group}.pt") == cache["groups"][group]["baseline_sha256"]
    # Fresh image -> frozen DINO/body -> cached-feature boundary, without using RGB as input.
    rae = load_model(Path(cache["assets_directory"]), device)
    frozen_parameter_counts = {
        "encoder": sum(p.numel() for p in rae.encoder.parameters()),
        "decoder_body": sum(p.numel() for p in rae.decoder.parameters()) - sum(p.numel() for p in rae.decoder.decoder_pred.parameters()),
        "excluded_published_rgb_projection": sum(p.numel() for p in rae.decoder.decoder_pred.parameters()),
    }
    fresh = []
    for batch in images["validation"][:32].split(cache["batch_size"]):
        values, _, _ = capture_features(rae, batch.to(device))
        fresh.append(values)
    fresh = torch.cat(fresh)
    feature_error = (fresh - features["validation"][:32]).abs().max().item()
    torch.testing.assert_close(fresh, features["validation"][:32], rtol=0, atol=2e-6)
    del rae
    torch.backends.cudnn.allow_tf32 = True
    results = {"protocol": protocol, "frozen_parameter_counts": frozen_parameter_counts,
               "groups": {kind: [] for kind in protocol["variants"]}}
    maximum_reload, maximum_e2e = 0.0, 0.0
    paired = {}
    for seed in protocol["seeds"]:
        initial_states, final_samplers, visits_by_kind = {}, {}, {}
        for kind in protocol["variants"]:
            path = root / f"{kind}_seed{seed}"
            completed = read(path / "COMPLETE.json")
            config = read(path / "config.json")
            assert completed["status"] == "complete" and completed["optimizer_steps"] == protocol["train_config"]["steps"]
            assert config["train"] == {**protocol["train_config"], "seed": seed}
            assert config["model"] == {**protocol["model_config"], "kind": kind}
            cfg = config["train"]
            assert completed["training_image_observations"] == cfg["steps"] * cfg["batch_size"]
            assert completed["training_pixel_queries"] == cfg["steps"] * cfg["batch_size"] * cfg["pixels_per_image"]
            assert completed["trainable_parameters"] == protocol["parameter_counts"][kind]
            assert completed["unique_training_images_seen"] == len(indices["train"])
            history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
            history = [row for row in history if row["event"] == "evaluation"]
            expected_steps = sorted({0, 1, cfg["steps"], *range(cfg["eval_every"], cfg["steps"] + 1, cfg["eval_every"])})
            assert [row["step"] for row in history] == expected_steps
            assert min(history, key=lambda row: row["validation"]["mse"])["step"] == completed["best_step"]
            for name, digest in completed["checkpoints"].items():
                assert sha256(path / "checkpoints" / f"{name}.pt") == digest
            for name, digest in completed["artifacts"].items():
                assert sha256(path / name) == digest
            initial = torch.load(path / "checkpoints/step0.pt", map_location="cpu", weights_only=True)
            initial_states[kind] = initial["model_state"]
            parameter_updates = {}
            for stage, checkpoint_name in (("final", "last"), ("selected", "best")):
                model, checkpoint = load_head(path / "checkpoints" / f"{checkpoint_name}.pt", device)
                assert checkpoint["dataset_indices"] == indices
                assert checkpoint["cache_manifest_sha256"] == protocol["cache_manifest_sha256"]
                assert checkpoint["step"] == (cfg["steps"] if stage == "final" else completed["best_step"])
                assert sum(p.numel() for p in model.parameters()) == completed["trainable_parameters"]
                if kind == "qvf":
                    assert model.quantum.n_qubits == model.quantum.repetitions == 5
                    assert model.embedder.amplitude_output
                    assert not hasattr(model, "rgb")
                else:
                    assert not model.embedder.amplitude_output and not hasattr(model, "quantum")
                if stage == "final":
                    final_samplers[kind] = checkpoint["sampler"]
                    visits_by_kind[kind] = checkpoint["visits"]
                    assert checkpoint["sampler"]["digest"] == completed["batch_stream_digest"]
                    assert checkpoint["visits"].sum().item() == completed["training_image_observations"]
                    for component in ("project", "spatial1", "spatial2", "embedder", "quantum" if kind == "qvf" else "rgb"):
                        change = sum((tensor.double() - initial["model_state"][name].double()).square().sum().item()
                                     for name, tensor in checkpoint["model_state"].items() if name.startswith(component + "."))
                        assert change > 0, (kind, seed, component)
                        parameter_updates[component] = change ** .5
                for group in ("train", "validation"):
                    saved = torch.load(path / stage / f"{group}.pt", map_location="cpu", weights_only=True)
                    assert saved["dataset_indices"] == indices[group]
                    torch.testing.assert_close(saved["originals"], images[group], rtol=0, atol=0)
                    prediction = saved["reconstructions"]
                    assert prediction.shape == images[group].shape and torch.isfinite(prediction).all()
                    calculated = numpy_metrics(prediction, images[group])
                    compare_metrics(calculated, completed["stages"][stage][group])
                    rendered = render_readout(model, features[group], cfg["batch_size"])
                    error = (rendered - prediction).abs().max().item()
                    maximum_reload = max(maximum_reload, error)
                    torch.testing.assert_close(rendered, prediction, rtol=0, atol=2e-6)
                    if group == "validation":
                        from_images = render_readout(model, fresh, cfg["batch_size"])
                        error = (from_images - prediction[:32]).abs().max().item()
                        maximum_e2e = max(maximum_e2e, error)
                        torch.testing.assert_close(from_images, prediction[:32], rtol=0, atol=2e-6)
                print(f"Verified {kind} seed{seed} {stage}: all 5000 images", flush=True)
            results["groups"][kind].append({"seed": seed, "directory": str(path),
                "elapsed_seconds": completed["elapsed_seconds"], "best_step": completed["best_step"],
                "stages": completed["stages"], "history": history, "parameter_update_l2": parameter_updates})
        qstate, cstate = initial_states["qvf"], initial_states["classical"]
        for name, tensor in qstate.items():
            if not name.startswith("quantum."):
                torch.testing.assert_close(tensor, cstate[name], rtol=0, atol=0)
        qsample, csample = final_samplers["qvf"], final_samplers["classical"]
        for name in ("order", "images_rng", "pixels_rng"):
            torch.testing.assert_close(qsample[name], csample[name], rtol=0, atol=0)
        for name in ("cursor", "epoch", "digest"):
            assert qsample[name] == csample[name]
        torch.testing.assert_close(visits_by_kind["qvf"], visits_by_kind["classical"], rtol=0, atol=0)
        paired[str(seed)] = {"identical_common_initial_weights": True, "identical_all_training_batches": True,
                             "batch_stream_digest": qsample["digest"]}
    results["summaries"] = {}
    for kind, rows in results["groups"].items():
        results["summaries"][kind] = {}
        for stage in ("final", "selected"):
            results["summaries"][kind][stage] = {}
            for group in ("train", "validation"):
                results["summaries"][kind][stage][group] = {}
                for metric in ("mse", "psnr_db", "gradient_mse"):
                    values = [row["stages"][stage][group][metric] for row in rows]
                    results["summaries"][kind][stage][group][metric] = {
                        "mean": statistics.mean(values),
                        "sample_sd": statistics.stdev(values) if len(values) > 1 else None, "per_seed": values}
    results["reference"] = read(Path(cache["baseline_directory"]) / "results.json")["groups"]
    results["paired_validation_qvf_minus_classical"] = {}
    for stage in ("final", "selected"):
        results["paired_validation_qvf_minus_classical"][stage] = {}
        for metric in ("mse", "psnr_db", "gradient_mse"):
            differences = [q["stages"][stage]["validation"][metric] - c["stages"][stage]["validation"][metric]
                           for q, c in zip(results["groups"]["qvf"], results["groups"]["classical"])]
            results["paired_validation_qvf_minus_classical"][stage][metric] = {
                "mean": statistics.mean(differences),
                "sample_sd": statistics.stdev(differences) if len(differences) > 1 else None,
                "per_seed": differences}
    audit = {"status": "passed", "all_runs_reached_declared_steps": True,
        "runs": len(protocol["seeds"]) * len(protocol["variants"]),
        "all_final_and_selected_5000_image_metrics_independently_recomputed": True,
        "all_final_and_selected_5000_image_outputs_rerendered": True,
        "max_native_gpu_rgb_reload_error": maximum_reload,
        "fresh_image_to_frozen_feature_max_error": feature_error,
        "fresh_image_end_to_end_max_rgb_error": maximum_e2e,
        "end_to_end_images_per_checkpoint": 32, "paired_checks": paired,
        "source_snapshots_weights_cache_and_reference_hashes_verified": True,
        "same_4744_256_split": True, "official_test_set_used": False,
        "five_qubits_depth_five_original_amplitudes_and_readout": True,
        "verifier_sha256": sha256(Path(__file__))}
    results["verification"] = audit
    shutil.copy2(Path(__file__), output / "verifier_source.py")
    write_json(output / "results.json", results)
    write_json(output / "verification.json", audit)
    print(json.dumps(results["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    verify(parser.parse_args())
