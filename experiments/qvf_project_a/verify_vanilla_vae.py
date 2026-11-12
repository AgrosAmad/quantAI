"""Verify completed vanilla-decoder VAE runs and build their comparison data."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .train import HERE, DEFAULT_DATA, REPO, configure_runtime, evaluate_vae, image_metrics, load_vae, sha256, write_json
from .vanilla_vae import VANILLA_ROOT, VANILLA_NAME


def read(path):
    return json.loads(path.read_text())


def verify(root, tag, output):
    protocol = read(root / f"{tag}_protocol.json")
    assert sha256(REPO / "experiments/qvf_repro/qvf_fast.py") == protocol["decoder_source_sha256"]
    assert sha256(VANILLA_ROOT / "checkpoints" / f"{VANILLA_NAME}.pt") == protocol["vanilla_checkpoint_sha256"]
    old0 = root / "20261108_v2_reconstruction_more_data_longer_seed0/vae/checkpoints/best.pt"
    assert sha256(old0) == protocol["old_checkpoint_sha256"]
    dataset = CIFAR10(str(DEFAULT_DATA), train=True, download=False)
    audited, large, fit = [], [], None
    max_reload_error = 0.0
    max_native_reload_error = 0.0
    max_full_precision_error = 0.0
    for member in protocol["runs"]:
        variant, cfg = member["variant"], member["config"]
        seed = cfg["seed"]
        directory = root / f"{tag}_vanilla_vae_{variant}_seed{seed}"
        result, recorded = read(directory / "COMPLETE.json"), read(directory / "config.json")
        assert result["status"] == "complete" and not (directory / "FAILED.json").exists()
        assert not (directory / "diffusion").exists()
        for key, value in cfg.items():
            assert recorded[key] == value, (directory, key)
        split = read(directory / "split.json")
        indices = split["indices"]
        assert not split["official_test_set_used"]
        assert len(indices["train"]) == len(set(indices["train"])) == cfg["n_train"]
        assert len(indices["validation"]) == len(set(indices["validation"])) == cfg["n_val"]
        assert not set(indices["train"]) & set(indices["validation"])
        assert all(dataset.targets[i] == 5 for group in indices.values() for i in group)
        manifest = read(directory / "source_manifest.json")
        assert manifest["experiments/qvf_repro/qvf_fast.py"] == protocol["decoder_source_sha256"]
        for filename, expected in manifest.items():
            assert sha256(directory / "source_snapshot" / filename) == expected
        training = result["vae"]
        assert training["optimizer_steps"] == cfg["vae_max_steps"]
        assert training["training_coordinate_queries"] == cfg["vae_max_steps"] * cfg["batch_size"] * cfg["pixels_per_image"]
        assert training["unique_training_images_seen"] == cfg["n_train"]
        assert training["parameter_counts"]["decoder"] == 48950
        history = [json.loads(line) for line in (directory / "vae/history.jsonl").read_text().splitlines()]
        validation = [row for row in history if "validation" in row]
        selected = min(validation, key=lambda r: r["validation"]["sampled_code"]["mse"]
                       + cfg["kl_weight"] * r["validation"]["kl_nats_per_image"])
        assert selected["step"] == training["best_step"]
        saved = {}
        for stage, checkpoint_name in (("selected", "best"), ("final", "last")):
            checkpoint_path = directory / "vae/checkpoints" / f"{checkpoint_name}.pt"
            assert sha256(checkpoint_path) == result["checkpoint_sha256"][checkpoint_name]
            model, checkpoint = load_vae(checkpoint_path, torch.device("cpu"))
            gpu_model, _ = load_vae(checkpoint_path, torch.device("cuda:0"))
            assert model.cfg.readout == "vanilla" and model.decoder.field.use_parity
            assert checkpoint["step"] == (training["best_step"] if stage == "selected" else cfg["vae_max_steps"])
            for group in ("train", "validation"):
                payload = torch.load(directory / stage / f"{group}.pt", map_location="cpu", weights_only=False)
                assert payload["dataset_indices"] == indices[group]
                targets = torch.from_numpy(dataset.data[indices[group]].copy()).permute(0, 3, 1, 2).float() / 255
                torch.testing.assert_close(payload["originals"], targets, atol=0, rtol=0)
                for key, metric in (("mean_reconstructions", "mean_code"), ("sampled_reconstructions", "sampled_code")):
                    raw = payload[key]
                    assert raw.shape == targets.shape and torch.isfinite(raw).all()
                    recomputed = image_metrics(raw, targets)
                    for name, value in recomputed.items():
                        assert math.isclose(value, result[stage][group][metric][name], rel_tol=0, abs_tol=1e-6)
                # Fresh CPU inference, with every validation image and a fixed
                # training subset. Stochastic CUDA/CPU RNG streams differ, so
                # this reload comparison deliberately uses posterior means.
                check_images = targets if group == "validation" else targets[:cfg["batch_size"]]
                with torch.no_grad():
                    reloaded = torch.cat([model.render(model.encoder(batch)[0]) for batch in check_images.split(cfg["batch_size"])])
                    # Existing VAE runs use deterministic cuDNN convolutions
                    # with TF32 allowed. Match that runtime and batch shape for
                    # the checkpoint-reload check; CPU uses full precision.
                    torch.backends.cudnn.allow_tf32 = True
                    native = torch.cat([gpu_model.render(gpu_model.encoder(batch.cuda())[0]).cpu()
                                        for batch in check_images.split(cfg["batch_size"])])
                    native_error = (native - payload["mean_reconstructions"][:len(check_images)]).abs().max().item()
                    max_native_reload_error = max(max_native_reload_error, native_error)
                    assert native_error < 3e-6, (directory, stage, group, native_error)
                    # Isolate the expected CPU/TF32 difference from an actual
                    # serialization or model-connection error.
                    torch.backends.cudnn.allow_tf32 = False
                    full_precision = torch.cat([gpu_model.render(gpu_model.encoder(batch.cuda())[0]).cpu()
                                                for batch in check_images.split(cfg["batch_size"])])
                    torch.backends.cudnn.allow_tf32 = True
                    full_precision_error = (reloaded - full_precision).abs().max().item()
                    max_full_precision_error = max(max_full_precision_error, full_precision_error)
                    assert full_precision_error < 1e-5, (directory, stage, group, full_precision_error)
                error = (reloaded - payload["mean_reconstructions"][:len(check_images)]).abs().max().item()
                max_reload_error = max(max_reload_error, error)
                saved[(stage, group)] = payload
        item = {"seed": seed, "variant": variant, "selected_step": training["best_step"],
                "final_step": cfg["vae_max_steps"], "directory": str(directory.resolve()),
                "validation_mean_psnr_db": result["selected"]["validation"]["mean_code"]["psnr_db"],
                "validation_sampled_psnr_db": result["selected"]["validation"]["sampled_code"]["psnr_db"],
                "validation_gradient_mse": result["selected"]["validation"]["mean_code_gradient_mse"],
                "train_final_mean_psnr_db": result["final"]["train"]["mean_code"]["psnr_db"],
                "train_final_sampled_psnr_db": result["final"]["train"]["sampled_code"]["psnr_db"],
                "history": [[row["step"], row["validation"]["mean_code"]["psnr_db"]] for row in validation]}
        if variant == "fit50":
            baseline = torch.load(VANILLA_ROOT / "checkpoints" / f"{VANILLA_NAME}.pt", map_location="cpu", weights_only=False)
            assert indices["train"] == baseline["dataset_indices"]
            original = baseline["original_images"].permute(0, 3, 1, 2)
            direct = baseline["raw_reconstructions"].permute(0, 3, 1, 2)
            payload = saved[("final", "train")]
            torch.testing.assert_close(original, payload["originals"], atol=0, rtol=0)
            save_image(torch.cat((original, direct, payload["mean_reconstructions"], payload["sampled_reconstructions"])),
                       output / "fit50.png", nrow=10, padding=1)
            item.update({"image": str((output / "fit50.png").resolve()), "ids": indices["train"],
                         "vanilla_train_psnr_db": image_metrics(direct, original)["psnr_db"],
                         "per_image_vanilla_psnr_db": (-10 * (direct-original).square().flatten(1).mean(1).log10()).tolist(),
                         "per_image_vae_psnr_db": result["final"]["train"]["mean_code_per_image_psnr_db"]})
            fit = item
        else:
            old_dir = root / f"20261108_v2_reconstruction_more_data_longer_seed{seed}"
            old_cfg, old_metrics = read(old_dir / "config.json"), read(old_dir / "vae/metrics.json")
            assert indices == read(old_dir / "split.json")["indices"]
            for key in cfg:
                if key == "model":
                    expected = {**old_cfg[key], "readout": "vanilla"}
                    assert cfg[key] == expected
                else:
                    assert cfg[key] == old_cfg[key], key
            item["legacy_validation_psnr_db"] = old_metrics["validation"]["mean_code"]["psnr_db"]
            item["legacy_sampled_psnr_db"] = old_metrics["validation"]["sampled_code"]["psnr_db"]
            item["paired_gain_db"] = item["validation_mean_psnr_db"] - item["legacy_validation_psnr_db"]
            old_model, _ = load_vae(old_dir / "vae/checkpoints/best.pt", torch.device("cuda:0"))
            assert old_model.cfg.readout == "legacy_affine"
            payload = saved[("selected", "validation")]
            old_result, old_mean, _ = evaluate_vae(old_model, payload["originals"], cfg["batch_size"],
                                                 torch.device("cuda:0"), seed + 1001)
            assert abs(old_result["mean_code"]["psnr_db"] - item["legacy_validation_psnr_db"]) < 1e-4
            save_image(torch.cat((payload["originals"], old_mean, payload["mean_reconstructions"], payload["sampled_reconstructions"])),
                       output / f"heldout_seed{seed}.png", nrow=16, padding=1)
            item.update({"image": str((output / f"heldout_seed{seed}.png").resolve()), "ids": indices["validation"],
                "per_image_vae_psnr_db": result["selected"]["validation"]["mean_code_per_image_psnr_db"],
                "per_image_legacy_psnr_db": (-10 * (old_mean-payload["originals"]).square().flatten(1).mean(1).log10()).tolist()})
            large.append(item)
        audited.append(directory.name)
        print(f"Verified {directory.name}", flush=True)
    assert fit is not None and len(large) == 3
    summary = {}
    for label, key in (("vanilla_vae", "validation_mean_psnr_db"), ("legacy_vae", "legacy_validation_psnr_db"),
                       ("paired_gain", "paired_gain_db"), ("sampled_vanilla_vae", "validation_sampled_psnr_db")):
        values = [item[key] for item in large]
        summary[label] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "per_seed": values}
    verification = {"status": "passed", "runs": audited, "run_count": len(audited),
        "raw_metrics_recomputed": True, "source_snapshots_verified": True,
        "validation_and_training_disjoint": True, "larger_protocol_and_ids_match_legacy": True,
        "fit50_ids_match_vanilla": True, "original_vanilla_source_and_checkpoint_unchanged": True,
        "historical_checkpoint_reload_preserves_affine_mapping": True,
        "max_cpu_reload_rgb_error": max_reload_error,
        "max_native_gpu_reload_rgb_error": max_native_reload_error,
        "max_cpu_vs_full_precision_gpu_rgb_error": max_full_precision_error,
        "cpu_difference_explanation": "The existing VAE runtime allows TF32 cuDNN convolutions. Native GPU reload and CPU versus full-float32 GPU checks are separate.",
        "diffusion_retrained": False, "official_test_set_used": False}
    return {"fit50": fit, "larger": large, "summary": summary, "verification": verification,
            "preflight": protocol["preflight"],
            "uncertainty": "Sample standard deviation across three training seeds on the same development validation split."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--tag", default="20261108_v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_runtime("cuda:0", 0)
    try:
        data = verify(args.root, args.tag, args.output)
        write_json(args.output / "results.json", data)
        write_json(args.output / "verification.json", data["verification"])
        print(json.dumps(data["summary"], indent=2))
    except BaseException as error:
        write_json(args.output / "FAILED.json", {"error": str(error), "type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
