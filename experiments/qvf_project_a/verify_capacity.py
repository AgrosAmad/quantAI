"""Check the duration/capacity study and build data for its interactive report."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .train import DEFAULT_DATA, HERE, REPO, configure_runtime, image_metrics, load_vae, sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def history(path):
    return [row for line in Path(path).read_text().splitlines() if "validation" in (row := json.loads(line))]


def score(row, cfg):
    return row["validation"]["sampled_code"]["mse"] + cfg["kl_weight"] * row["validation"]["kl_nats_per_image"]


def check_prefix(directory, previous):
    old = history(previous / "vae/history.jsonl")
    new = {row["step"]: row for row in history(directory / "vae/history.jsonl")}
    assert len(old) == 25 and old[-1]["step"] == 7680
    for row in old:
        actual = new[row["step"]]
        for key in ("step", "epoch", "pixel_mse", "kl_nats", "kl_weight", "validation"):
            assert actual[key] == row[key], (directory, row["step"], key)
    return len(old)


def brief(result, cfg, rows):
    return {
        "seed": cfg["seed"], "field_width": cfg["model"]["field_width"],
        "decoder_parameters": result["vae"]["parameter_counts"]["decoder"],
        "encoder_parameters": result["vae"]["parameter_counts"]["encoder"],
        "qubits": cfg["model"]["qubits"], "depth": cfg["model"]["circuit_depth"],
        "steps": result["vae"]["optimizer_steps"], "selected_step": result["vae"]["best_step"],
        "elapsed_seconds": result["elapsed_seconds"],
        "training_elapsed_seconds": result["vae"]["elapsed_seconds"],
        "selected_validation_psnr": result["selected"]["validation"]["mean_code"]["psnr_db"],
        "selected_sampled_psnr": result["selected"]["validation"]["sampled_code"]["psnr_db"],
        "final_validation_psnr": result["final"]["validation"]["mean_code"]["psnr_db"],
        "final_train_psnr": result["final"]["train"]["mean_code"]["psnr_db"],
        "selected_gradient_mse": result["selected"]["validation"]["mean_code_gradient_mse"],
        "final_gradient_mse": result["final"]["validation"]["mean_code_gradient_mse"],
        "selected_per_image_psnr": result["selected"]["validation"]["mean_code_per_image_psnr_db"],
        "final_per_image_psnr": result["final"]["validation"]["mean_code_per_image_psnr_db"],
        "selected_train_per_image_psnr": result["selected"]["train"]["mean_code_per_image_psnr_db"][:64],
        "final_train_per_image_psnr": result["final"]["train"]["mean_code_per_image_psnr_db"][:64],
        "curve": [[row["step"], row["validation"]["mean_code"]["psnr_db"],
                   row["validation"]["sampled_code"]["mse"], row["validation"]["kl_nats_per_image"]]
                  for row in rows],
    }


def verify(protocol_path, output):
    protocol = read(protocol_path)
    source = REPO / "experiments/qvf_repro/qvf_fast.py"
    assert sha256(source) == protocol["decoder_source_sha256"]
    assert sha256(Path(protocol["reference_run"]) / "split.json") == protocol["reference_split_sha256"]
    dataset = CIFAR10(str(DEFAULT_DATA), train=True, download=False)
    data = {"groups": {"previous": [], "baseline50k": [], "wide51250k": []},
            "images": [], "protocol": protocol, "verification": {}}
    previous_paths = {item["seed"]: Path(item["directory"]) for item in protocol["previous_runs"]}
    previous_splits = {}
    for previous in protocol["previous_runs"]:
        directory = Path(previous["directory"])
        assert sha256(directory / "vae/checkpoints/best.pt") == previous["best_sha256"]
        assert sha256(directory / "vae/checkpoints/last.pt") == previous["last_sha256"]
        result, cfg = read(directory / "COMPLETE.json"), read(directory / "config.json")
        data["groups"]["previous"].append(brief(result, cfg, history(directory / "vae/history.jsonl")))
        previous_splits[previous["seed"]] = read(directory / "split.json")["indices"]
    max_reload_error, audited, prefixes = 0.0, [], {}
    for run in protocol["runs"]:
        directory, cfg, variant = Path(run["directory"]), run["config"], run["variant"]
        recorded, result = read(directory / "config.json"), read(directory / "COMPLETE.json")
        assert result["status"] == "complete" and not (directory / "FAILED.json").exists()
        assert not (directory / "diffusion").exists()
        assert all(recorded[key] == value for key, value in cfg.items())
        assert (cfg["model"]["qubits"], cfg["model"]["circuit_depth"], cfg["model"]["readout"]) == (5, 5, "vanilla")
        expected_width = 128 if variant == "baseline50k" else 512
        assert cfg["model"]["field_width"] == expected_width
        split = read(directory / "split.json")
        indices = split["indices"]
        assert indices == previous_splits[cfg["seed"]]
        assert len(set(indices["train"])) == 4744 and len(set(indices["validation"])) == 256
        assert not set(indices["train"]) & set(indices["validation"])
        assert not split["official_test_set_used"]
        manifest = read(directory / "source_manifest.json")
        assert manifest["experiments/qvf_repro/qvf_fast.py"] == protocol["decoder_source_sha256"]
        for filename, expected in manifest.items():
            assert sha256(directory / "source_snapshot" / filename) == expected
        training = result["vae"]
        assert training["optimizer_steps"] == 50000
        assert training["training_image_observations"] == 1600000
        assert training["training_coordinate_queries"] == 409600000
        assert training["unique_training_images_seen"] == 4744
        assert training["parameter_counts"] == {"encoder": 427872,
            "decoder": 48950 if expected_width == 128 else 588470, "circuit": 150}
        rows = history(directory / "vae/history.jsonl")
        assert [r["step"] for r in rows] == [1, *range(320, 50001, 320), 50000]
        assert min(rows, key=lambda r: score(r, cfg))["step"] == training["best_step"]
        if variant == "baseline50k":
            prefixes[str(cfg["seed"])] = check_prefix(directory, previous_paths[cfg["seed"]])
        for stage, checkpoint_name in (("selected", "best"), ("final", "last")):
            checkpoint_path = directory / "vae/checkpoints" / f"{checkpoint_name}.pt"
            assert sha256(checkpoint_path) == result["checkpoint_sha256"][checkpoint_name]
            model, checkpoint = load_vae(checkpoint_path, torch.device("cuda:0"))
            assert model.cfg.readout == "vanilla" and model.decoder.field.use_parity
            assert model.decoder.field.quantum.n_qubits == model.decoder.field.quantum.repetitions == 5
            assert sum(p.numel() for p in model.decoder.parameters()) == training["parameter_counts"]["decoder"]
            assert checkpoint["step"] == (training["best_step"] if stage == "selected" else 50000)
            for group in ("train", "validation"):
                payload = torch.load(directory / stage / f"{group}.pt", map_location="cpu", weights_only=False)
                assert payload["dataset_indices"] == indices[group]
                target = torch.from_numpy(dataset.data[indices[group]].copy()).permute(0, 3, 1, 2).float() / 255
                torch.testing.assert_close(payload["originals"], target, atol=0, rtol=0)
                assert all(dataset.targets[i] == 5 for i in indices[group])
                for key, metric in (("mean_reconstructions", "mean_code"), ("sampled_reconstructions", "sampled_code")):
                    prediction = payload[key]
                    assert prediction.shape == target.shape and torch.isfinite(prediction).all()
                    for name, value in image_metrics(prediction, target).items():
                        assert math.isclose(value, result[stage][group][metric][name], rel_tol=0, abs_tol=1e-6)
                error_image = payload["mean_reconstructions"] - target
                per_image_psnr = -10 * error_image.square().flatten(1).mean(1).clamp_min(1e-12).log10()
                torch.testing.assert_close(per_image_psnr, torch.tensor(
                    result[stage][group]["mean_code_per_image_psnr_db"]), atol=1e-6, rtol=0)
                gradient_mse = 0.5 * ((error_image[:, :, 1:] - error_image[:, :, :-1]).square().flatten(1).mean(1)
                    + (error_image[:, :, :, 1:] - error_image[:, :, :, :-1]).square().flatten(1).mean(1)).mean().item()
                assert abs(gradient_mse - result[stage][group]["mean_code_gradient_mse"]) < 1e-8
                checked = target if group == "validation" else target[:32]
                with torch.no_grad():
                    reloaded = torch.cat([model.render(model.encoder(batch.cuda())[0]).cpu() for batch in checked.split(32)])
                error = (reloaded - payload["mean_reconstructions"][:len(checked)]).abs().max().item()
                max_reload_error = max(max_reload_error, error)
                assert error < 3e-6, (directory, stage, group, error)
        item = brief(result, cfg, rows)
        item["directory"] = str(directory)
        data["groups"][variant].append(item)
        audited.append(directory.name)
        print(f"Verified {directory.name}", flush=True)
    # Rendering comparisons expose every held-out image, for every seed, at
    # both the declared selection rule and the fixed final training budget.
    for seed in protocol["seeds"]:
        paths = [previous_paths[seed], *[Path(next(r["directory"] for r in protocol["runs"]
                         if r["variant"] == variant and r["config"]["seed"] == seed))
                         for variant in ("baseline50k", "wide51250k")]]
        for stage in ("selected", "final"):
            for group, count in (("validation", 256), ("train", 64)):
                payloads = [torch.load(p / stage / f"{group}.pt", map_location="cpu", weights_only=False) for p in paths]
                original = payloads[0]["originals"][:count]
                assert all(torch.equal(original, p["originals"][:count]) for p in payloads)
                image_path = output / f"{group}_{stage}_seed{seed}.png"
                save_image(torch.cat([original, *[p["mean_reconstructions"][:count] for p in payloads]]),
                           image_path, nrow=16, padding=1)
                data["images"].append({"seed": seed, "stage": stage, "group": group,
                                       "count": count, "group_rows": count // 16,
                                       "image": str(image_path.resolve()),
                                       "ids": payloads[0]["dataset_indices"][:count]})
    data["summaries"] = {}
    for label, rows in data["groups"].items():
        data["summaries"][label] = {}
        for key in ("selected_validation_psnr", "final_validation_psnr", "final_train_psnr",
                    "selected_sampled_psnr", "selected_gradient_mse", "final_gradient_mse"):
            values = [r[key] for r in rows]
            data["summaries"][label][key] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "per_seed": values}
    data["comparisons"] = {}
    for name, treatment, control in (("longer_training", "baseline50k", "previous"),
                                      ("wider_field", "wide51250k", "baseline50k")):
        data["comparisons"][name] = {}
        for metric in ("selected_validation_psnr", "final_validation_psnr", "final_train_psnr"):
            values = [a[metric] - b[metric] for a, b in zip(data["groups"][treatment], data["groups"][control])]
            data["comparisons"][name][metric] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "per_seed": values}
    data["verification"] = {"status": "passed", "run_count": len(audited), "runs": audited,
        "all_runs_reached_50000_updates": True, "same_training_and_validation_ids": True,
        "previous_baseline_histories_reproduced_exactly": prefixes, "source_snapshots_verified": True,
        "raw_metrics_and_gradients_recomputed": True, "five_qubits_depth_five_and_original_readout_preserved": True,
        "max_native_gpu_reload_rgb_error": max_reload_error, "previous_checkpoints_unchanged": True,
        "diffusion_trained": False, "official_test_set_used": False}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_runtime("cuda:0", 0)
    torch.backends.cudnn.allow_tf32 = True
    try:
        data = verify(args.protocol, args.output)
        write_json(args.output / "results.json", data)
        write_json(args.output / "verification.json", data["verification"])
        print(json.dumps(data["comparisons"], indent=2))
    except BaseException as error:
        write_json(args.output / "FAILED.json", {"error": str(error), "type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
