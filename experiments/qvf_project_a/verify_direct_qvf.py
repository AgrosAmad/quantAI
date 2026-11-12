"""Independently verify direct fitting and compare like-for-like training images."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .direct_qvf import load_direct, reconstruction_metrics
from .models import pixel_coordinates
from .train import DEFAULT_DATA, REPO, configure_runtime, load_vae, sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def expected_visits(seed, count, batch, steps):
    generator = torch.Generator().manual_seed(seed + 11)
    visits = torch.zeros(count, dtype=torch.int64)
    completed = 0
    while completed < steps:
        order = torch.randperm(count, generator=generator)
        batches = min(count // batch, steps - completed)
        visits[order[:batches * batch]] += 1
        completed += batches
    return visits


def verify(protocol_path, output):
    protocol = read(protocol_path)
    assert sha256(REPO / "experiments/qvf_repro/qvf_fast.py") == protocol["decoder_source_sha256"]
    assert sha256(Path(protocol["reference_run"]) / "split.json") == protocol["reference_split_sha256"]
    dataset = CIFAR10(str(DEFAULT_DATA), train=True, download=False)
    gallery = protocol["gallery_training_positions"]
    assert gallery == torch.linspace(0, 4743, 256).round().long().tolist()
    data = {"runs": [], "images": [], "protocol": protocol, "verification": {}}
    max_direct_error, max_vae_error = 0.0, 0.0
    for spec in protocol["runs"]:
        directory, cfg = Path(spec["directory"]), spec["config"]
        seed = cfg["seed"]
        previous = next(r for r in protocol["vae_comparisons"] if r["seed"] == seed)
        vae_dir = Path(previous["directory"])
        result, saved_cfg = read(directory / "COMPLETE.json"), read(directory / "config.json")
        assert result["status"] == "complete" and not (directory / "FAILED.json").exists()
        assert all(saved_cfg[key] == value for key, value in cfg.items())
        assert (cfg["model"]["field_width"], cfg["model"]["qubits"], cfg["model"]["circuit_depth"],
                cfg["model"]["readout"], cfg["model"]["latent_dim"]) == (128, 5, 5, "vanilla", 64)
        assert not (directory / "vae").exists() and not (directory / "diffusion").exists()
        manifest = read(directory / "source_manifest.json")
        assert manifest["experiments/qvf_repro/qvf_fast.py"] == protocol["decoder_source_sha256"]
        for filename, expected in manifest.items():
            assert sha256(directory / "source_snapshot" / filename) == expected
        split = read(directory / "split.json")
        ids = split["indices"]
        assert ids == read(vae_dir / "split.json")["indices"]
        assert len(ids["train"]) == len(set(ids["train"])) == 4744
        assert len(ids["validation"]) == 256 and not set(ids["train"]) & set(ids["validation"])
        assert not split["official_test_set_used"]
        assert not split["validation_images_used_for_optimization_or_selection"]
        assert all(dataset.targets[i] == 5 for i in ids["train"])
        assert result["optimizer_steps"] == cfg["steps"] == 50000
        assert result["training_image_observations"] == 1600000
        assert result["training_coordinate_queries"] == 409600000
        assert result["unique_training_images_seen"] == 4744
        assert result["parameter_counts"] == {"decoder": 48950, "circuit": 150,
                                               "image_codes": 303616, "encoder": 0}
        visits = expected_visits(seed, cfg["n_train"], cfg["batch_size"], cfg["steps"])
        assert visits.tolist() == result["visits_per_image"]
        rows = [json.loads(line) for line in (directory / "fit/history.jsonl").read_text().splitlines()]
        curve = [r for r in rows if r["event"] == "fitting_evaluation"]
        assert [r["step"] for r in curve] == sorted({0, 1, 7680, *range(5000, 50001, 5000)})
        assert result["best_fitting_step"] == min(curve, key=lambda r: r["fit"]["mse"])["step"]
        for name, digest in result["checkpoint_sha256"].items():
            assert sha256(directory / "fit/checkpoints" / f"{name}.pt") == digest
        decoder, codes, checkpoint = load_direct(directory / "fit/checkpoints/last.pt", torch.device("cuda:0"))
        assert checkpoint["step"] == 50000 and checkpoint["dataset_indices"] == ids["train"]
        assert torch.equal(checkpoint["visits"], visits)
        assert not hasattr(decoder, "encoder") and decoder.field.use_parity
        assert decoder.field.quantum.n_qubits == decoder.field.quantum.repetitions == 5
        assert sum(p.numel() for p in decoder.parameters()) == 48950
        assert codes.shape == (4744, 64) and torch.isfinite(codes).all()
        for value in decoder.state_dict().values():
            assert torch.isfinite(value).all()
        payload = torch.load(directory / "final/train.pt", map_location="cpu", weights_only=False)
        originals = torch.from_numpy(dataset.data[ids["train"]].copy()).permute(0, 3, 1, 2).float() / 255
        assert payload["dataset_indices"] == ids["train"]
        torch.testing.assert_close(payload["originals"], originals, atol=0, rtol=0)
        prediction = payload["reconstructions"]
        assert prediction.shape == originals.shape and torch.isfinite(prediction).all()
        metrics = reconstruction_metrics(prediction, originals)
        assert metrics == result["final_fit"] == read(directory / "final/metrics.json")
        with torch.no_grad():
            coordinates = pixel_coordinates(32, 32, device="cuda:0")
            rebuilt = torch.cat([decoder(batch, coordinates).transpose(1, 2).reshape(len(batch), 3, 32, 32).cpu()
                                 for batch in codes.split(32)])
        direct_error = (rebuilt - prediction).abs().max().item()
        assert direct_error < 3e-6, (directory, direct_error)
        max_direct_error = max(max_direct_error, direct_error)

        vae_checkpoint = vae_dir / "vae/checkpoints/last.pt"
        assert sha256(vae_checkpoint) == previous["last_sha256"]
        vae, vae_state = load_vae(vae_checkpoint, torch.device("cuda:0"))
        assert vae_state["step"] == 50000
        vae_payload = torch.load(vae_dir / "final/train.pt", map_location="cpu", weights_only=False)
        assert vae_payload["dataset_indices"] == ids["train"]
        torch.testing.assert_close(vae_payload["originals"], originals, atol=0, rtol=0)
        vae_prediction = vae_payload["mean_reconstructions"]
        vae_metrics = reconstruction_metrics(vae_prediction, originals)
        old = read(vae_dir / "COMPLETE.json")["final"]["train"]
        assert vae_metrics["psnr_db"] == old["mean_code"]["psnr_db"]
        assert vae_metrics["mse"] == old["mean_code"]["mse"]
        assert abs(vae_metrics["gradient_mse"] - old["mean_code_gradient_mse"]) < 1e-8
        with torch.no_grad():
            vae_rebuilt = torch.cat([vae.render(vae.encoder(batch.cuda())[0]).cpu() for batch in originals.split(32)])
        vae_error = (vae_rebuilt - vae_prediction).abs().max().item()
        assert vae_error < 3e-6, (vae_dir, vae_error)
        max_vae_error = max(max_vae_error, vae_error)

        image_path = output / f"training_comparison_seed{seed}.png"
        save_image(torch.cat((originals[gallery], vae_prediction[gallery], prediction[gallery])),
                   image_path, nrow=16, padding=1)
        # The inline report uses lossless WebP copies; raw measurements stay in
        # float32 tensors and the PNG/display conversion does not affect scores.
        webp = image_path.with_suffix(".webp")
        original_display = Image.open(image_path).convert("RGB")
        with webp.open("xb") as handle:
            original_display.save(handle, format="WEBP", lossless=True, method=6)
        assert Image.open(webp).convert("RGB").tobytes() == original_display.tobytes()
        data["images"].append({"seed": seed, "path": str(webp.resolve()),
                                "dataset_indices": [ids["train"][i] for i in gallery],
                                "training_positions": gallery})
        data["runs"].append({"seed": seed, "directory": str(directory), "vae_directory": str(vae_dir),
            "direct": {k: v for k, v in metrics.items() if k != "per_image_psnr_db"},
            "vae": {k: v for k, v in vae_metrics.items() if k != "per_image_psnr_db"},
            "direct_gallery_psnr": [metrics["per_image_psnr_db"][i] for i in gallery],
            "vae_gallery_psnr": [vae_metrics["per_image_psnr_db"][i] for i in gallery],
            "direct_minus_vae_psnr": metrics["psnr_db"] - vae_metrics["psnr_db"],
            "curve": [[r["step"], r["fit"]["psnr_db"], r["fit"]["gradient_mse"]] for r in curve],
            "elapsed_seconds": result["elapsed_seconds"], "best_fitting_step": result["best_fitting_step"],
            "mean_visits_per_code": result["equivalent_dataset_passes"],
            "min_visits_per_code": visits.min().item(), "max_visits_per_code": visits.max().item(),
            "mean_code_norm": codes.detach().norm(dim=1).mean().item()})
        print(f"Verified seed {seed}: VAE train {vae_metrics['psnr_db']:.3f} dB; "
              f"direct QVF train {metrics['psnr_db']:.3f} dB; "
              f"direct/VAE reload error {direct_error:.3g}/{vae_error:.3g}", flush=True)
    data["summary"] = {}
    for kind in ("direct", "vae"):
        data["summary"][kind] = {}
        for metric in ("psnr_db", "mse", "gradient_mse"):
            values = [run[kind][metric] for run in data["runs"]]
            data["summary"][kind][metric] = {"mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values), "per_seed": values}
    differences = [run["direct_minus_vae_psnr"] for run in data["runs"]]
    data["paired_psnr_change"] = {"mean": statistics.mean(differences),
                                 "sample_sd": statistics.stdev(differences), "per_seed": differences}
    data["verification"] = {"status": "passed", "run_count": len(data["runs"]),
        "full_dataset_metrics_recomputed": True, "all_4744_images_reloaded_for_both_models": True,
        "matching_training_ids": True, "no_validation_fitting_or_selection": True,
        "exact_update_query_budgets": True, "per_image_visits_replayed": True,
        "source_snapshots_checked": True, "original_qvf_source_unchanged": True,
        "previous_vae_checkpoints_unchanged": True, "five_qubits_and_original_readout": True,
        "max_direct_reload_rgb_error": max_direct_error, "max_vae_reload_rgb_error": max_vae_error,
        "gallery_predetermined": True, "display_webp_pixels_identical_to_png": True}
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
        print(json.dumps({"summary": data["summary"], "paired": data["paired_psnr_change"]}, indent=2))
    except BaseException as error:
        write_json(args.output / "FAILED.json", {"type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
