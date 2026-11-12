"""Independent reload, budget and NumPy metric audit for the token pilot."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .direct_qvf import render_direct
from .token_qvf import TokenConfig, build_token_decoder
from .token_study import load_token
from .train import REPO, configure_runtime, sha256, write_json
from .verify_direct_qvf import expected_visits


def metrics_numpy(prediction, target, pixels=None):
    error = prediction.numpy().astype(np.float64) - target.numpy().astype(np.float64)
    selected = error.reshape(len(error), 3, -1)
    if pixels is not None:
        selected = selected[:, :, pixels]
    per_image = np.mean(selected ** 2, axis=(1, 2))
    result = {"mse": float(per_image.mean()),
              "psnr_db": float((-10 * np.log10(np.maximum(per_image, 1e-12))).mean())}
    if pixels is None:
        result["gradient_mse"] = float(.5 * (np.mean(np.diff(error, axis=2) ** 2)
                                                    + np.mean(np.diff(error, axis=3) ** 2)))
    return result


def check_metrics(recomputed, saved):
    for key, value in recomputed.items():
        tolerance = 5e-5 if key == "psnr_db" else 2e-8
        assert abs(value - saved[key]) < tolerance, (key, value, saved[key])


def verify(root, device_name="cuda:0"):
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    device = configure_runtime(device_name, 191)
    output = root / "verified"
    output.mkdir(exist_ok=True)
    for source, digest in protocol["source_sha256"].items():
        assert sha256(REPO / source) == digest
        assert sha256(root / "source_snapshot" / source) == digest
    assert not protocol["official_test_set_used"]
    ids = protocol["indices"]
    assert len(set(ids["train"])) == len(ids["train"])
    assert not set(ids["train"]) & set(ids["validation"])
    completion = protocol["completion"]
    assert set(completion["dataset_indices"]).issubset(ids["validation"])
    assert sorted(completion["observed_pixels"] + completion["unobserved_pixels"]) == list(range(1024))
    dataset = CIFAR10(protocol["data_root"], train=True, download=False)
    originals = torch.from_numpy(dataset.data[ids["train"]].copy()).permute(0, 3, 1, 2).float() / 255
    heldout = torch.from_numpy(dataset.data[completion["dataset_indices"]].copy()).permute(0, 3, 1, 2).float() / 255
    assert all(dataset.targets[i] == 5 for i in ids["train"] + completion["dataset_indices"])
    results, streams, initial_codes, initial_angles, shared_weights = [], {}, {}, {}, {}
    galleries = defaultdict(dict)
    maximum_error = 0.0
    for spec in protocol["runs"]:
        directory = root / spec["directory"]
        assert not (directory / "FAILED.json").exists()
        done = json.loads((directory / "COMPLETE.json").read_text())
        assert done["status"] == "complete" and done["protocol_sha256"] == sha256(root / "protocol.json")
        cfg, modelcfg = spec["config"], TokenConfig(**spec["config"]["model"])
        assert json.loads((directory / "config.json").read_text()) == cfg
        assert json.loads((directory / "split.json").read_text()) == ids
        seed, q = cfg["seed"], modelcfg.qubits
        for name, digest in done["checkpoint_sha256"].items():
            assert sha256(directory / "fit/checkpoints" / f"{name}.pt") == digest
        model, codes, saved = load_token(directory / "fit/checkpoints/last.pt", device)
        initial = torch.load(directory / "fit/checkpoints/step0.pt", weights_only=False)
        assert saved["step"] == cfg["steps"] == done["optimizer_steps"]
        assert saved["dataset_indices"] == ids["train"]
        assert codes.shape == (cfg["n_train"], modelcfg.latent_dim)
        assert sum(p.numel() for p in model.parameters()) == spec["shared_parameters"]
        visits = expected_visits(seed, cfg["n_train"], cfg["batch_size"], cfg["steps"])
        assert torch.equal(visits, saved["visits"]) and visits.tolist() == done["visits_per_image"]
        assert done["training_coordinate_queries"] == cfg["steps"] * cfg["batch_size"] * cfg["pixels_per_image"]
        assert done["parameter_counts"]["encoder"] == 0
        assert done["parameter_counts"]["image_codes"] == cfg["n_train"] * modelcfg.latent_dim
        pixel_rng = torch.Generator(device=device).manual_seed(seed + 12)
        for _ in range(cfg["steps"]):
            torch.randperm(1024, device=device, generator=pixel_rng)
        assert torch.equal(pixel_rng.get_state().cpu(), saved["rng"]["pixels"].cpu())
        if seed in streams:
            for name in ("order", "pixels"):
                assert torch.equal(streams[seed][name], saved["rng"][name])
        streams[seed] = saved["rng"]
        key = seed, modelcfg.latent_dim
        if key in initial_codes:
            assert torch.equal(initial_codes[key], initial["codes"])
        initial_codes[key] = initial["codes"]
        fresh = build_token_decoder(modelcfg, seed)
        for name, value in fresh.state_dict().items():
            assert torch.equal(value, initial["decoder_state"][name]), name
        angle_key = seed, q
        angles = torch.cat((fresh.circuit.weights.flatten(), fresh.circuit.cond.flatten()))
        if angle_key in initial_angles:
            assert torch.equal(angles, initial_angles[angle_key])
        initial_angles[angle_key] = angles
        if modelcfg.retrieval != "global":
            common = {k: v for k, v in fresh.state_dict().items() if k.startswith("values.") or
                      k.startswith(("amplitudes.layers.1.", "amplitudes.layers.2.", "amplitudes.layers.3."))}
            if angle_key in shared_weights:
                for name, value in common.items():
                    assert torch.equal(value, shared_weights[angle_key][name]), name
            shared_weights[angle_key] = common
        for name, value in model.state_dict().items():
            assert torch.isfinite(value).all(), name
        assert not torch.equal(codes.cpu(), initial["codes"])
        circuit_key = "field.quantum.weights" if modelcfg.retrieval == "global" else "quantum.weights"
        assert not torch.equal(saved["decoder_state"][circuit_key], initial["decoder_state"][circuit_key])
        if modelcfg.retrieval == "attention":
            for name in ("queries.0.weight", "keys.0.weight"):
                assert not torch.equal(saved["decoder_state"][name], initial["decoder_state"][name])
        prediction = render_direct(model, codes, cfg["batch_size"])
        payload = torch.load(directory / "final/train.pt", weights_only=False)
        assert torch.equal(payload["originals"], originals)
        assert payload["dataset_indices"] == ids["train"]
        error = float((prediction - payload["reconstructions"]).abs().max())
        assert error == 0, (spec["directory"], error)
        maximum_error = max(maximum_error, error)
        fitting = metrics_numpy(prediction, originals)
        check_metrics(fitting, done["final_fit"])
        new = torch.load(directory / "final/completion.pt", weights_only=False)
        assert torch.equal(new["originals"], heldout)
        assert new["dataset_indices"] == completion["dataset_indices"]
        assert new["observed_pixels"].tolist() == completion["observed_pixels"]
        assert new["unobserved_pixels"].tolist() == completion["unobserved_pixels"]
        assert new["metrics"]["steps"] == completion["steps"] and new["metrics"]["decoder_frozen"]
        # Match the frozen decoder used for completion. CUDA linear kernels can
        # differ by float32 roundoff when parameter requires_grad flags change,
        # even inside no_grad; matching the saved execution mode gives exact RGB.
        model.requires_grad_(False)
        reconstruction = render_direct(model, new["codes"].to(device), 16)
        error = float((reconstruction - new["reconstructions"]).abs().max())
        assert error == 0
        maximum_error = max(maximum_error, error)
        observed = metrics_numpy(reconstruction, heldout, completion["observed_pixels"])
        unseen = metrics_numpy(reconstruction, heldout, completion["unobserved_pixels"])
        check_metrics(observed, done["completion"]["observed"])
        check_metrics(unseen, done["completion"]["unobserved"])
        rows = [json.loads(row) for row in (directory / "fit/history.jsonl").read_text().splitlines()]
        curve = [row for row in rows if row["event"] == "fitting_evaluation"]
        expected_evals = sorted({0, 1, cfg["steps"], *range(cfg["eval_every"], cfg["steps"] + 1, cfg["eval_every"])})
        if cfg["steps"] >= 7680:
            expected_evals = sorted(set(expected_evals) | {7680})
        assert [r["step"] for r in curve] == expected_evals
        assert saved["best_fitting_step"] == min(curve, key=lambda r: r["fit"]["mse"])["step"]
        results.append({"name": spec["name"], "seed": seed, "qubits": q,
            "shared_parameters": spec["shared_parameters"], "code_entries": modelcfg.latent_dim,
            "fit": fitting, "completion_observed": observed, "completion_unobserved": unseen,
            "elapsed_seconds": done["elapsed_seconds"], "curve": curve})
        if seed == protocol["seeds"][0]:
            label = spec["name"].split("_", 1)[1]
            galleries[q][label] = (prediction[:12], reconstruction[:12])
        print(f"Verified {spec['directory']}: exact reload, paired streams, independent metrics", flush=True)
    names = list(galleries[5])
    for q in (5, 7):
        train = torch.stack((originals[:12], *(galleries[q][name][0] for name in names)), 1).flatten(0, 1)
        masked = torch.zeros_like(heldout[:12])
        masked.flatten(2)[:, :, completion["observed_pixels"]] = heldout[:12].flatten(2)[:, :, completion["observed_pixels"]]
        test = torch.stack((heldout[:12], masked, *(galleries[q][name][1] for name in names)), 1).flatten(0, 1)
        save_image(train, output / f"q{q}_training_examples.png", nrow=1 + len(names), padding=2)
        save_image(test, output / f"q{q}_completion_examples.png", nrow=2 + len(names), padding=2)
    write_json(output / "verification.json", {"runs_verified": len(results),
        "max_reload_rgb_error": maximum_error, "source_and_checkpoint_hashes_match": True,
        "paired_initialization_and_streams_match": True, "metrics_recomputed_with_numpy_float64": True,
        "official_test_set_used": False,
        "gallery_columns_training": ["original", *names],
        "gallery_columns_completion": ["original", "observed_only", *names],
        "gallery_images": "First 12 training / completion IDs, protocol order; first seed only. Display clipped; metrics raw."})
    write_json(output / "results.json", {"protocol_sha256": sha256(root / "protocol.json"), "runs": results})
    audit_sources = {}
    for name in ("verify_token_study.py", "token_report.py", "tests/test_token_qvf.py"):
        source = Path(__file__).parent / name
        destination = output / "audit_source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        audit_sources[name] = sha256(destination)
    write_json(output / "audit_source_manifest.json", audit_sources)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(verify(args.root, args.device))


if __name__ == "__main__":
    main()
