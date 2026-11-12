"""Verify vanilla QVF against the pinned release without changing its RGB readout.

The release is imported from an external path. Only its three invalid rotation
indices and its dataset-subset reshape are repaired in memory. The original
PennyLane circuit, embedder, readout and loss otherwise execute unchanged.
"""

import argparse
import hashlib
import json
import math
import subprocess
import types
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from check_equivalence import equivalence, make_fixed_layer
from qvf_fast import QVFImage, parity_decode, positional_embedding
from train import HERE, PRESETS, load_images, make_coords, model_inputs

PINNED_COMMIT = "cb647ef16aa20a8ca90514cef74a7b0927bb60a2"


def load_repaired_release(path):
    path = Path(path).resolve()
    original = path.read_text()
    repairs = {
        "weights[:self.n_qubits, :, r]": "weights[:self.n_qubits, r]",
        "weights[self.n_qubits:, :, r]": "weights[self.n_qubits:, r]",
        "weights[:, :, r]": "weights[:, r]",
        "self.pixels = images.view(num_samples, h * w, 3)":
            "self.pixels = self.images.view(num_samples, h * w, 3)",
    }
    source = original
    for old, new in repairs.items():
        if source.count(old) != 1:
            raise ValueError(f"Expected exactly one known release defect: {old}")
        source = source.replace(old, new)
    module = types.ModuleType("qvf_repaired_release")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, {"source_path": str(path), "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                    "in_memory_repairs": repairs, "repaired_source_sha256": hashlib.sha256(source.encode()).hexdigest()}


def full_model_and_training(official, identity, steps=30):
    torch.manual_seed(41)
    cfg = dict(PRESETS["paper"], use_identity_block=identity, train_samples_use=4)
    device = torch.device("cpu")
    images = load_images(cfg, HERE / "data")
    target = images.reshape(-1, 3)
    dataset = official.CIFARImplicitDataset(images, 3)
    assert torch.equal(dataset.pixels, images[:3].reshape(3, -1, 3))
    coords = make_coords(32, 32)
    features = positional_embedding(coords, cfg["L_embed"])
    reference_features = official.positional_embedding(coords[None].repeat(4, 1, 1), cfg["L_embed"], device)
    torch.testing.assert_close(features[None].expand_as(reference_features), reference_features, rtol=0, atol=0)
    ref, fast = official.ImageRegressor(cfg, device), QVFImage(cfg)
    fast.load_state_dict(ref.state_dict(), strict=True)
    ref_codes = official.generate_latent_codes(4, cfg["latent_size"], device)
    fast_codes = ref_codes.detach().clone().requires_grad_(True)
    opts_ref = [torch.optim.Adam(ref.parameters(), lr=cfg["lr_model"]),
                torch.optim.Adam([ref_codes], lr=cfg["lr_latent"])]
    opts_fast = [torch.optim.Adam(fast.parameters(), lr=cfg["lr_model"]),
                 torch.optim.Adam([fast_codes], lr=cfg["lr_latent"])]
    readouts = []
    for parity in (True, False):
        ref.use_parity = fast.use_parity = parity
        x = model_inputs(fast_codes, features)
        with torch.no_grad():
            expected, actual = ref(x), fast(x)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        readouts.append({"parity": parity, "max_output_difference": (actual - expected).abs().max().item()})
    ref.use_parity = fast.use_parity = True
    gradient_error = None
    losses = []
    for step in range(steps):
        for opt in opts_ref + opts_fast:
            opt.zero_grad()
        x_ref = torch.cat([ref_codes[:, None].expand(-1, 1024, -1), reference_features], dim=-1).reshape(-1, 90)
        x_fast = model_inputs(fast_codes, features)
        expected_loss, expected_mse, _ = official.mse_latent_loss(target, ref(x_ref), x_ref[:, :64], math.sqrt(cfg["latent_reg"]))
        actual_mse = torch.nn.functional.mse_loss(fast(x_fast), target)
        actual_loss = actual_mse + cfg["latent_reg"] * x_fast[:, :64].norm(dim=1).mean()
        expected_loss.backward()
        actual_loss.backward()
        if step == 0:
            gradient_error = {key: (p.grad - dict(ref.named_parameters())[key].grad).abs().max().item()
                              for key, p in fast.named_parameters()}
            gradient_error["latent_codes"] = (fast_codes.grad - ref_codes.grad).abs().max().item()
            assert max(gradient_error.values()) < 2e-5
        for opt in opts_ref + opts_fast:
            opt.step()
        relative = abs(actual_loss.item() - expected_loss.item()) / abs(expected_loss.item())
        losses.append({"step": step + 1, "official_loss": expected_loss.item(),
                       "fast_loss": actual_loss.item(), "relative_difference": relative})
    assert max(row["relative_difference"] for row in losses) < 2e-4
    return {"identity_initialization": identity, "images": 4, "steps": steps,
            "readout_checks": readouts, "first_step_gradient_max_errors": gradient_error, "loss_history": losses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output exists; choose a new filename")
    torch.set_num_threads(2)
    commit = subprocess.check_output(["git", "-C", str(args.official.resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip()
    assert commit == PINNED_COMMIT
    official, provenance = load_repaired_release(args.official)
    # Twenty configurations, including generic angles, compare the matrix shortcut
    # against direct PennyLane gate execution in float64.
    rows = equivalence(make_fixed_layer(official), torch.device("cpu"))
    numeric_keys = [key for key in rows[0] if key.startswith("max_abs") or key == "orthogonality_error"]
    assert all(row[key] < 1e-10 for row in rows for key in numeric_keys)
    print("20 circuit output, gradient and orthogonality checks passed", flush=True)
    full = []
    for identity in (False, True):
        full.append(full_model_and_training(official, identity))
        print(f"Full release model and 30 training steps passed; identity={identity}", flush=True)
    for n in (3, 4, 5, 6):
        x = torch.randn(17, n)
        cfg = dict(PRESETS["paper"], use_identity_block=False)
        ref = official.ImageRegressor(cfg, torch.device("cpu"))
        torch.testing.assert_close(parity_decode(x, 3), ref.parity(x, 3), rtol=0, atol=0)
    report = {"status": "passed", "official_commit": commit, "provenance": provenance,
              "versions": {"torch": torch.__version__, "pennylane": qml.__version__, "numpy": np.__version__},
              "circuit_checks": rows, "full_model_training_checks": full,
              "parity_readout_matches_release": True, "coordinate_encoding_matches_release": True,
              "dataset_subset_checked": True, "output_affine_rescaling": False,
              "scope": "Numerical implementation equivalence, not reproduction of undocumented paper settings."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
