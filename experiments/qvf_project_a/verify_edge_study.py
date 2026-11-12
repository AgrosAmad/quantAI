"""Independent NumPy and checkpoint audit of the V10 edge-loss experiment."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .direct_qvf import render_direct
from .edge_qvf import build_edge_decoder
from .edge_study import load_edge
from .train import REPO, configure_runtime, sha256, write_json
from .verify_token_study import check_metrics, metrics_numpy


def expected_stream(config):
    """Generate pairs independently using integer coordinates, not training code."""
    order = torch.Generator().manual_seed(config["seed"] + 11)
    queries = torch.Generator().manual_seed(config["seed"] + 12)
    visits = torch.zeros(config["n_train"], dtype=torch.int64)
    digest, step = hashlib.sha256(), 0
    while step < config["steps"]:
        for rows in torch.randperm(config["n_train"], generator=order).split(config["batch_size"]):
            if len(rows) < config["batch_size"]:
                continue
            h = torch.randint(32 * 31, (config["pairs_per_orientation"],), generator=queries)
            v = torch.randint(32 * 31, (config["pairs_per_orientation"],), generator=queries)
            left = (h // 31) * 32 + h % 31
            points = torch.cat((torch.stack((left, left + 1), -1), torch.stack((v, v + 32), -1))).flatten()
            digest.update(rows.numpy().tobytes())
            digest.update(points.numpy().tobytes())
            visits[rows] += 1
            step += 1
            if step == config["steps"]:
                break
    return digest.hexdigest(), visits, {"order": order.get_state(), "queries": queries.get_state()}


def masked_edge_numpy(prediction, target, pixels):
    error = prediction.numpy().astype(np.float64) - target.numpy().astype(np.float64)
    mask = np.zeros((32, 32), dtype=bool)
    mask.reshape(-1)[pixels] = True
    horizontal = mask[:, 1:] & mask[:, :-1]
    vertical = mask[1:] & mask[:-1]
    h = np.diff(error, axis=3)[:, :, horizontal]
    v = np.diff(error, axis=2)[:, :, vertical]
    per_image = .5 * ((h ** 2).mean((1, 2)) + (v ** 2).mean((1, 2)))
    return {"gradient_mse": float(per_image.mean()), "horizontal_edges": int(horizontal.sum()),
            "vertical_edges": int(vertical.sum()), "per_image_gradient_mse": per_image.tolist()}


def verify(root, device_name="cuda:0"):
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    device = configure_runtime(device_name, 177)
    output = root / "verified"
    output.mkdir(exist_ok=True)
    for name, digest in protocol["source_sha256"].items():
        assert sha256(REPO / name) == digest, name
        assert sha256(root / "source_snapshot" / name) == digest, name
    ids = protocol["indices"]
    spec_completion = protocol["completion"]
    assert not protocol["official_test_set_used"]
    assert not set(ids["train"]) & set(ids["validation"])
    assert len(set(ids["train"])) == len(ids["train"])
    assert set(spec_completion["dataset_indices"]).issubset(ids["validation"])
    assert sorted(spec_completion["observed_pixels"] + spec_completion["unobserved_pixels"]) == list(range(1024))
    dataset = CIFAR10(protocol["data_root"], train=True, download=False)
    originals = torch.from_numpy(dataset.data[ids["train"]].copy()).permute(0, 3, 1, 2).float() / 255
    excluded = torch.from_numpy(dataset.data[spec_completion["dataset_indices"]].copy()).permute(0, 3, 1, 2).float() / 255
    assert all(dataset.targets[i] == 5 for i in ids["train"] + spec_completion["dataset_indices"])
    streams, initial_codes, initial_weights, completion_codes, initial_scores = {}, {}, {}, {}, {}
    runs, galleries = [], {}
    for spec in protocol["runs"]:
        directory = root / spec["directory"]
        assert not (directory / "FAILED.json").exists()
        done = json.loads((directory / "COMPLETE.json").read_text())
        cfg, seed = spec["config"], spec["config"]["seed"]
        assert done["status"] == "complete"
        assert done["protocol_sha256"] == sha256(root / "protocol.json")
        assert json.loads((directory / "config.json").read_text()) == cfg
        for name, digest in done["checkpoint_sha256"].items():
            assert sha256(directory / "fit/checkpoints" / f"{name}.pt") == digest
        assert sha256(directory / "final/completion.pt") == done["completion_sha256"]
        model, codes, saved = load_edge(directory / "fit/checkpoints/last.pt", device)
        initial = torch.load(directory / "fit/checkpoints/step0.pt", weights_only=False)
        fresh = build_edge_decoder(cfg["architecture"], seed)
        for name, value in fresh.state_dict().items():
            assert torch.equal(value, initial["decoder_state"][name]), name
        common = {k: v for k, v in initial["decoder_state"].items() if not k.startswith("uploads.")}
        if seed in initial_codes:
            assert torch.equal(initial_codes[seed], initial["codes"])
            for name, value in common.items():
                assert torch.equal(value, initial_weights[seed][name])
            # Factoring one matrix into three blocks changes float32 rounding;
            # double-precision function/gradient equivalence is tested separately.
            for key in ("mse", "psnr_db", "gradient_mse"):
                tolerance = 5e-5 if key == "psnr_db" else 2e-6
                assert abs(initial["fit"][key] - initial_scores[seed][key]) < tolerance
        initial_codes[seed], initial_weights[seed], initial_scores[seed] = initial["codes"], common, initial["fit"]
        if cfg["architecture"] == "staged_all":
            for name, value in initial["decoder_state"].items():
                if name.startswith("uploads."):
                    assert torch.count_nonzero(value) == 0
            assert saved["decoder_state"]["uploads.0.weight"].abs().sum() > 0
        assert not torch.equal(saved["decoder_state"]["quantum.weights"], initial["decoder_state"]["quantum.weights"])
        assert not torch.equal(codes.cpu(), initial["codes"])
        assert codes.shape == (cfg["n_train"], 336)
        assert saved["step"] == done["optimizer_steps"] == cfg["steps"]
        assert saved["dataset_indices"] == ids["train"]
        assert done["shared_parameters"] == spec["shared_parameters"] == sum(p.numel() for p in model.parameters())
        assert done["image_code_entries"] == cfg["n_train"] * 336
        assert done["training_coordinate_queries"] == saved["training_coordinate_queries"] == cfg["steps"] * cfg["batch_size"] * cfg["pairs_per_orientation"] * 4
        if seed not in streams:
            streams[seed] = expected_stream(cfg)
        digest, visits, rng = streams[seed]
        assert digest == saved["stream_sha256"] == done["stream_sha256"]
        assert torch.equal(visits, saved["visits"]) and visits.tolist() == done["visits_per_image"]
        for name, state in rng.items():
            assert torch.equal(state, saved["rng"][name])
        prediction = render_direct(model, codes, cfg["batch_size"])
        payload = torch.load(directory / "final/train.pt", weights_only=False)
        assert payload["dataset_indices"] == ids["train"] and torch.equal(payload["originals"], originals)
        assert torch.equal(prediction, payload["reconstructions"])
        fitting = metrics_numpy(prediction, originals)
        check_metrics(fitting, done["final_fit"])
        history = [json.loads(line) for line in (directory / "fit/history.jsonl").read_text().splitlines()]
        expected_steps = sorted({0, 1, 2000, cfg["steps"], *range(cfg["eval_every"], cfg["steps"] + 1, cfg["eval_every"])})
        assert [r["step"] for r in history] == expected_steps
        early_model, early_codes, early_checkpoint = load_edge(directory / "fit/checkpoints/step2000.pt", device)
        early_metrics = metrics_numpy(render_direct(early_model, early_codes, cfg["batch_size"]), originals)
        check_metrics(early_metrics, next(r for r in history if r["step"] == 2000)["fit"])
        assert early_checkpoint["step"] == 2000
        del early_model, early_codes
        new = torch.load(directory / "final/completion.pt", weights_only=False)
        assert new["dataset_indices"] == spec_completion["dataset_indices"]
        assert torch.equal(new["originals"], excluded)
        assert new["observed_pixels"].tolist() == spec_completion["observed_pixels"]
        assert new["unobserved_pixels"].tolist() == spec_completion["unobserved_pixels"]
        if seed in completion_codes:
            assert torch.equal(completion_codes[seed], new["initial_codes"])
        completion_codes[seed] = new["initial_codes"]
        model.requires_grad_(False)  # match completion's CUDA execution mode
        restored = render_direct(model, new["codes"].to(device), 16)
        assert torch.equal(restored, new["reconstructions"])
        unseen = metrics_numpy(restored, excluded, spec_completion["unobserved_pixels"])
        check_metrics(unseen, done["completion"]["unobserved"])
        observed = metrics_numpy(restored, excluded, spec_completion["observed_pixels"])
        check_metrics(observed, done["completion"]["observed"])
        edges = masked_edge_numpy(restored, excluded, spec_completion["unobserved_pixels"])
        stored_edges = done["completion"]["unobserved_edges"]
        assert abs(edges["gradient_mse"] - stored_edges["gradient_mse"]) < 1e-12
        for key in ("horizontal_edges", "vertical_edges"):
            assert edges[key] == stored_edges[key]
        assert np.max(np.abs(np.array(edges["per_image_gradient_mse"]) - np.array(stored_edges["per_image_gradient_mse"]))) < 1e-12
        assert done["completion"]["steps"] == spec_completion["steps"] and done["completion"]["decoder_frozen"]
        runs.append({"name": spec["name"], "seed": seed, "config": cfg, "fit": fitting,
            "step2000": early_metrics, "completion_unobserved": unseen,
            "completion_unobserved_edges": edges, "shared_parameters": done["shared_parameters"],
            "curve": history, "elapsed_seconds": done["elapsed_seconds"]})
        if seed == protocol["seeds"][0]:
            galleries[spec["name"]] = (prediction[:12], restored[:12])
        print(f"Verified {spec['directory']}: paired queries, exact reload, independent pixel/edge metrics", flush=True)
    names = list(galleries)
    train = torch.stack((originals[:12], *(galleries[name][0] for name in names)), 1).flatten(0, 1)
    masked = torch.zeros_like(excluded[:12])
    pixels = spec_completion["observed_pixels"]
    masked.flatten(2)[:, :, pixels] = excluded[:12].flatten(2)[:, :, pixels]
    completion = torch.stack((excluded[:12], masked, *(galleries[name][1] for name in names)), 1).flatten(0, 1)
    save_image(train, output / "training_examples.png", nrow=len(names) + 1, padding=2)
    save_image(completion, output / "completion_examples.png", nrow=len(names) + 2, padding=2)
    write_json(output / "results.json", {"protocol_sha256": sha256(root / "protocol.json"), "runs": runs})
    write_json(output / "verification.json", {"run_count": len(runs), "maximum_reload_rgb_error": 0,
        "source_and_checkpoint_hashes_match": True, "paired_initial_codes_and_shared_weights": True,
        "independent_query_stream_and_visits_match": True, "independent_numpy_metrics_match": True,
        "step2000_metrics_independently_recomputed": True, "official_test_set_used": False,
        "gallery_columns_training": ["original", *names],
        "gallery_columns_completion": ["original", "observed_only", *names],
        "gallery_rows": "First 12 protocol IDs; first seed only. RGB clipped only for display."})
    audit = {}
    for name in ("verify_edge_study.py", "edge_report.py", "tests/test_edge_qvf.py"):
        source = Path(__file__).parent / name
        destination = output / "audit_source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        audit[name] = sha256(destination)
    write_json(output / "audit_source_manifest.json", audit)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(verify(args.root, args.device))
