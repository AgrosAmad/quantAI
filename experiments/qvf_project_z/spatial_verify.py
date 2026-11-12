"""Independent Z4 spatial/qubit checkpoint, sampling, resource and NumPy metric audit."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from experiments.qvf_project_a.direct_qvf import render_direct
from .spatial_models import build_model, resources
from .spatial_study import load_model
from experiments.qvf_project_a.train import REPO, configure_runtime, sha256, write_json
from experiments.qvf_project_a.verify_edge_study import expected_stream, masked_edge_numpy
from experiments.qvf_project_a.verify_token_study import check_metrics, metrics_numpy


def verify(root, device_name="cuda:0"):
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    device = configure_runtime(device_name, 199)
    output = root / "verified"
    output.mkdir(exist_ok=True)
    for name, digest in protocol["source_sha256"].items():
        assert sha256(REPO / name) == digest, name
        assert sha256(root / "source_snapshot" / name) == digest, name
    ids, completion = protocol["indices"], protocol["completion"]
    assert len(set(ids["train"])) == len(ids["train"])
    assert not set(ids["train"]) & set(ids["validation"])
    assert set(completion["dataset_indices"]).issubset(ids["validation"])
    assert sorted(completion["observed_pixels"] + completion["unobserved_pixels"]) == list(range(1024))
    assert not protocol["official_test_set_used"]
    dataset = CIFAR10(protocol["data_root"], train=True, download=False)
    originals = torch.from_numpy(dataset.data[ids["train"]].copy()).permute(0, 3, 1, 2).float() / 255
    excluded = torch.from_numpy(dataset.data[completion["dataset_indices"]].copy()).permute(0, 3, 1, 2).float() / 255
    assert all(dataset.targets[i] == 5 for i in ids["train"] + completion["dataset_indices"])
    streams, initial_codes, initial_modules, completion_codes = {}, {}, {}, {}
    runs, galleries, baseline_checks = [], {}, []
    for spec in protocol["runs"]:
        directory = root / spec["directory"]
        assert not (directory / "FAILED.json").exists()
        done = json.loads((directory / "COMPLETE.json").read_text())
        cfg, seed, name = spec["config"], spec["config"]["seed"], spec["name"]
        assert done["status"] == "complete" and done["stage"] == spec["stage"]
        assert done["protocol_sha256"] == sha256(root / "protocol.json")
        assert json.loads((directory / "config.json").read_text()) == cfg
        for label, digest in done["checkpoint_sha256"].items():
            assert sha256(directory / "fit/checkpoints" / f"{label}.pt") == digest
        assert sha256(directory / "final/completion.pt") == done["completion_sha256"]
        model, codes, saved = load_model(directory / "fit/checkpoints/last.pt", device)
        initial = torch.load(directory / "fit/checkpoints/step0.pt", weights_only=False)
        fresh = build_model(name, seed)
        for key, value in fresh.state_dict().items():
            assert torch.equal(value, initial["decoder_state"][key]), (name, key)
        for module_name, module in fresh.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                shapes = tuple((k, tuple(v.shape)) for k, v in module.state_dict().items())
                key = (seed, module_name, shapes)
                if key in initial_modules:
                    for tensor_name, value in module.state_dict().items():
                        assert torch.equal(value, initial_modules[key][tensor_name]), key
                initial_modules[key] = {k: v.clone() for k, v in module.state_dict().items()}
        if seed in initial_codes:
            assert torch.equal(initial_codes[seed], initial["codes"])
        initial_codes[seed] = initial["codes"]
        assert not torch.equal(codes.cpu(), initial["codes"])
        assert codes.shape == (cfg["n_train"], 336)
        assert saved["step"] == done["optimizer_steps"] == cfg["steps"]
        assert saved["dataset_indices"] == ids["train"]
        assert done["shared_parameters"] == spec["shared_parameters"] == sum(p.numel() for p in model.parameters())
        assert done["image_code_entries"] == cfg["n_train"] * 336
        assert done["resources"] == resources(model, name)
        for key, value in done["resources"].items():
            assert spec[key] == value, key
        if model.circuit is not None:
            assert not torch.equal(saved["decoder_state"]["quantum.weights"], initial["decoder_state"]["quantum.weights"])
        if hasattr(model, "position_weights"):
            assert not torch.equal(saved["decoder_state"]["position_weights"], initial["decoder_state"]["position_weights"])
        rgb_queries = cfg["steps"] * cfg["batch_size"] * cfg["pairs_per_orientation"] * 4
        per_image_quantum_queries = cfg["pairs_per_orientation"] * 4
        quantum_queries = cfg["steps"] * cfg["batch_size"] * per_image_quantum_queries
        assert done["training_coordinate_queries"] == saved["training_coordinate_queries"] == rgb_queries
        assert done["training_quantum_queries"] == saved["training_quantum_queries"] == quantum_queries
        if seed not in streams:
            streams[seed] = expected_stream(cfg)
        digest, visits, rng = streams[seed]
        assert digest == saved["stream_sha256"] == done["stream_sha256"]
        assert torch.equal(visits, saved["visits"]) and visits.tolist() == done["visits_per_image"]
        for key, value in rng.items():
            assert torch.equal(value, saved["rng"][key])
        prediction = render_direct(model, codes, cfg["batch_size"])
        payload = torch.load(directory / "final/train.pt", weights_only=False)
        assert payload["dataset_indices"] == ids["train"] and torch.equal(payload["originals"], originals)
        assert torch.equal(prediction, payload["reconstructions"])
        fitting = metrics_numpy(prediction, originals)
        check_metrics(fitting, done["final_fit"])
        history = [json.loads(line) for line in (directory / "fit/history.jsonl").read_text().splitlines()]
        expected_steps = sorted({0, 1, 2000, cfg["steps"], *range(cfg["eval_every"], cfg["steps"] + 1, cfg["eval_every"])})
        assert [row["step"] for row in history] == expected_steps
        early_model, early_codes, early_saved = load_model(directory / "fit/checkpoints/step2000.pt", device)
        early = metrics_numpy(render_direct(early_model, early_codes, cfg["batch_size"]), originals)
        check_metrics(early, next(row["fit"] for row in history if row["step"] == 2000))
        assert early_saved["step"] == 2000
        del early_model, early_codes
        new = torch.load(directory / "final/completion.pt", weights_only=False)
        assert new["dataset_indices"] == completion["dataset_indices"]
        assert torch.equal(new["originals"], excluded)
        for key in ("observed_pixels", "unobserved_pixels"):
            assert new[key].tolist() == completion[key]
        if seed in completion_codes:
            assert torch.equal(completion_codes[seed], new["initial_codes"])
        completion_codes[seed] = new["initial_codes"]
        model.requires_grad_(False)
        restored = render_direct(model, new["codes"].to(device), 16)
        assert torch.equal(restored, new["reconstructions"])
        unseen = metrics_numpy(restored, excluded, completion["unobserved_pixels"])
        observed = metrics_numpy(restored, excluded, completion["observed_pixels"])
        check_metrics(unseen, done["completion"]["unobserved"])
        check_metrics(observed, done["completion"]["observed"])
        edges = masked_edge_numpy(restored, excluded, completion["unobserved_pixels"])
        stored = done["completion"]["unobserved_edges"]
        assert abs(edges["gradient_mse"] - stored["gradient_mse"]) < 1e-12
        assert np.max(np.abs(np.array(edges["per_image_gradient_mse"]) - np.array(stored["per_image_gradient_mse"]))) < 1e-12
        for key in ("horizontal_edges", "vertical_edges"):
            assert edges[key] == stored[key]
        assert done["completion"]["steps"] == completion["steps"] and done["completion"]["decoder_frozen"]
        if name == "attention_chain_q5":
            previous = REPO / "experiments/qvf_project_z/results/20261110_z3_local_quantum" / f"attention_local_shared_seed{seed}"
            old = torch.load(previous / "fit/checkpoints/last.pt", weights_only=False)
            exact_parameters = all(torch.equal(old["decoder_state"][k], v) for k, v in saved["decoder_state"].items())
            exact_codes = torch.equal(old["codes"], saved["codes"])
            baseline_checks.append({"seed": seed, "parameters_bit_identical_to_z3": exact_parameters,
                                    "codes_bit_identical_to_z3": exact_codes})
        runs.append({"name": name, "stage": spec["stage"], "seed": seed, "config": cfg,
            "fit": fitting, "step2000": early, "completion_unobserved": unseen,
            "completion_unobserved_edges": edges, "resources": done["resources"],
            "elapsed_seconds": done["elapsed_seconds"], "curve": history})
        if seed == protocol["seeds"][0]:
            galleries[name] = (prediction[:12], restored[:12])
        print(f"Verified {spec['directory']}: exact reload, independent metrics, paired streams and resource counts", flush=True)
    columns = {"selected": list(dict.fromkeys(r["name"] for r in protocol["runs"]))}
    masked = torch.zeros_like(excluded[:12])
    observed = completion["observed_pixels"]
    masked.flatten(2)[:, :, observed] = excluded[:12].flatten(2)[:, :, observed]
    for stage, names in columns.items():
        training = torch.stack((originals[:12], *(galleries[n][0] for n in names)), 1).flatten(0, 1)
        completed = torch.stack((excluded[:12], masked, *(galleries[n][1] for n in names)), 1).flatten(0, 1)
        save_image(training, output / f"{stage}_training_examples.png", nrow=len(names) + 1, padding=2)
        save_image(completed, output / f"{stage}_completion_examples.png", nrow=len(names) + 2, padding=2)
    write_json(output / "results.json", {"protocol_sha256": sha256(root / "protocol.json"), "runs": runs})
    write_json(output / "verification.json", {"run_count": len(runs), "maximum_reload_rgb_error": 0,
        "source_and_checkpoint_hashes_match": True, "paired_initial_codes_and_common_modules": True,
        "independent_query_stream_and_visits_match": True, "independent_numpy_metrics_match": True,
        "step2000_metrics_independently_recomputed": True, "quantum_query_counts_checked": True,
        "official_test_set_used": False, "z3_reproduction": baseline_checks,
        "gallery_columns_after_original": columns, "completion_adds_observed_only_after_original": True,
        "gallery_rows": "First 12 protocol IDs; first seed only. RGB clipped only for display."})
    audit = {}
    for name in ("spatial_verify.py", "tests/test_spatial.py"):
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
