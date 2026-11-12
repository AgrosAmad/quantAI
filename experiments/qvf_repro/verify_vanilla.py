"""Reload and independently verify the nine recorded vanilla reconstruction runs."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torchvision

from audit_vanilla import load_repaired_release
from qvf_fast import QVFImage, positional_embedding
from sweep import group_runs
from train import HERE, PRESETS, model_inputs, run_name


def verify(root, official_path):
    torch.set_num_threads(2)
    official, provenance = load_repaired_release(official_path)
    audit = json.loads((root / "audit.json").read_text())
    assert audit["status"] == "passed"
    assert audit["provenance"]["source_sha256"] == provenance["source_sha256"]
    dataset = torchvision.datasets.CIFAR10(str(HERE / "data"), train=True, download=False)
    groups, checks = {}, []
    for group in ("vanilla_single", "vanilla_collection", "vanilla_paper_text"):
        rows = []
        for specification in group_runs(group):
            n = specification["images"] or PRESETS[specification["preset"]]["train_samples_use"]
            name = run_name(specification["preset"], "qvf", "gaussian", "relu", n,
                            specification["seed"], specification["tag"])
            result = json.loads((root / f"{name}.json").read_text())
            checkpoint = torch.load(root / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)
            cfg = checkpoint["config"]
            assert cfg == result["config"]
            assert result["status"] == "complete" and result["output_affine_rescaling"] is False
            assert (cfg["out_dim"], cfg["num_repetations"], cfg["hid_dim"]) == (32, 5, 128)
            assert cfg["activation"] == "relu" and cfg["train_epochs"] == 5000
            assert result["optimizer_steps"] == (5000 if n == 1 else 25000)
            assert result["training_coordinate_queries"] == 5000 * n * 1024
            assert cfg["use_parity"] == (group != "vanilla_paper_text")
            assert cfg["pe_scale"] == PRESETS[specification["preset"]]["pe_scale"]
            assert cfg["latent_reg"] == PRESETS[specification["preset"]]["latent_reg"]
            assert checkpoint["dataset_indices"] == result["dataset_indices"]
            ids = result["dataset_indices"]
            assert all(dataset.targets[i] == 5 for i in ids)
            original = torch.tensor(dataset.data[ids]) / 255.0
            torch.testing.assert_close(original, checkpoint["original_images"], rtol=0, atol=0)
            for filename, expected in result["source_sha256"].items():
                archive = root / "source_snapshot" / name / filename
                assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
            assert hashlib.sha256((HERE / "qvf_fast.py").read_bytes()).hexdigest() == result["source_sha256"]["qvf_fast.py"]
            model = QVFImage(cfg).eval()
            model.load_state_dict(checkpoint["model_state"], strict=True)
            codes, coords = checkpoint["latent_codes"], checkpoint["coordinates"]
            assert codes.shape == (n, 64)
            features = positional_embedding(coords, cfg["L_embed"], scale=cfg["pe_scale"])
            with torch.no_grad():
                reconstruction = torch.cat([model(model_inputs(batch, features)).reshape(len(batch), 32, 32, 3)
                                            for batch in codes.split(10)])
            reload_error = (reconstruction - checkpoint["raw_reconstructions"]).abs().max().item()
            assert reload_error < 2e-5, (name, reload_error)
            mse = (reconstruction - original).square().flatten(1).mean(1)
            psnr = -10 * mse.log10()
            assert abs(psnr.mean().item() - result["psnr_mean"]) < 5e-4
            assert abs(mse.mean().item() - result["mse_mean"]) < 1e-7
            # Compare learned weights with the actual release, including its RGB
            # mapping, on the first three images and 64 fixed pixel coordinates.
            reference = official.ImageRegressor(cfg, torch.device("cpu")).eval()
            reference.load_state_dict(checkpoint["model_state"], strict=True)
            x = model_inputs(codes[:3], features[::16])
            with torch.no_grad():
                official_error = (reference(x) - model(x)).abs().max().item()
            assert official_error < 2e-5, (name, official_error)
            checks.append({"name": name, "checkpoint_reload_max_abs_error": reload_error,
                           "trained_model_vs_release_max_abs_error": official_error,
                           "dataset_and_source_hashes_checked": True})
            result["image_path"] = str((root / f"{name}_all.png").resolve())
            rows.append(result)
            print(f"Verified {name}: {result['psnr_mean']:.3f} dB, release error {official_error:.2e}", flush=True)
        groups[group] = rows
    shared_ids = groups["vanilla_collection"][0]["dataset_indices"]
    assert all(row["dataset_indices"] == shared_ids for key in ("vanilla_collection", "vanilla_paper_text") for row in groups[key])
    summaries = {}
    for key in ("vanilla_collection", "vanilla_paper_text"):
        values = [row["psnr_mean"] for row in groups[key]]
        summaries[key] = {"mean_psnr_db": statistics.mean(values), "sample_sd_psnr_db": statistics.stdev(values),
                          "mean_mse": statistics.mean(row["mse_mean"] for row in groups[key]),
                          "seeds": [row["seed"] for row in groups[key]]}
    single_values = [row["psnr_mean"] for row in groups["vanilla_single"]]
    verification = {"status": "passed", "run_count": len(checks), "runs": checks,
                    "minimum_single_image_psnr_db": min(single_values),
                    "single_image_40_db_check_passed": min(single_values) >= 40,
                    "same_collection_images_across_all_seeds_and_conventions": True,
                    "original_qvf_model_source_unchanged": True,
                    "raw_metrics_recomputed_from_reloaded_checkpoints": True,
                    "learned_weights_match_released_pennylane_model": True,
                    "encoder_used": False, "output_affine_rescaling_used": False,
                    "scope": "Fitted training images; no test of unseen-image generalization or image generation."}
    return {"groups": groups, "summaries": summaries, "verification": verification,
            "paper_table3_gaussian_relu": {"mean_psnr_db": 29.8, "reported_sd": 1.1,
                                           "source": "https://arxiv.org/html/2508.10900v2#S5.T3"},
            "protocol": json.loads((root / "protocol.json").read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    outputs = [Path(str(args.output_stem) + ending) for ending in (".json", "_verification.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Choose a fresh output stem")
    data = verify(args.results.resolve(), args.official)
    for path, contents in zip(outputs, [data, data["verification"]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            json.dump(contents, handle, indent=2, allow_nan=False)
        print(path.resolve())


if __name__ == "__main__":
    main()
