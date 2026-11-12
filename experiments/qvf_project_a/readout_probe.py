"""Training-set diagnostic: fit tiny RGB readouts to frozen V10 quantum states.

This is not an end-to-end architecture comparison or a generalization test.
Original checkpoints and their fitted image codes are never changed. Compare
the released fixed RGB rule with learned affine maps of the same five Z means,
then those five means plus ten ZZ pair correlations. Both measured feature sets
are accessible from the same computational-basis samples on hardware; no finite
shots are simulated here.
"""

import argparse
import itertools
import json
import shutil
import statistics
from pathlib import Path

import torch

from .edge_study import load_edge
from .models import pixel_coordinates
from .train import REPO, configure_runtime, sha256, write_json
from .verify_token_study import metrics_numpy


def diagonal_observables(qubits, dtype=torch.float64):
    """Rows are basis outcomes (wire zero = MSB); columns are Z, then ZZ."""
    bits = (torch.arange(2 ** qubits)[:, None] >> torch.arange(qubits - 1, -1, -1)) & 1
    signs = 1 - 2 * bits.to(dtype)
    pairs = list(itertools.combinations(range(qubits), 2))
    return torch.cat((signs, torch.stack([signs[:, i] * signs[:, j] for i, j in pairs], 1)), 1)


def affine_design(features):
    return torch.cat((features.double(), torch.ones_like(features[..., :1], dtype=torch.float64)), -1)


def fit_affine(features, targets, edge_weight=.5):
    """Exact full-image pixel + edge least squares, with one global affine map.

    Shapes are [images, height, width, features/channels]. The edge convention
    averages the two orientations equally, as in V10. The fitted code penalty
    is constant in this diagnostic and therefore omitted.
    """
    x, y = affine_design(features), targets.double()
    flat, values = x.flatten(0, 2), y.flatten(0, 2)
    gram = flat.T @ flat / len(flat)
    rhs = flat.T @ values / len(flat)
    for axis in (1, 2):
        dx = x.diff(dim=axis).flatten(0, 2)
        dy = y.diff(dim=axis).flatten(0, 2)
        gram += .5 * edge_weight * (dx.T @ dx) / len(dx)
        rhs += .5 * edge_weight * (dx.T @ dy) / len(dx)
    solution = torch.linalg.lstsq(gram, rhs, driver="gelsd", rcond=1e-12).solution
    residual = (gram @ solution - rhs).abs().max().item()
    assert residual < 1e-8, residual
    eigenvalues = torch.linalg.eigvalsh(gram)
    return solution, {"normal_equation_max_residual": residual,
        "gram_min_eigenvalue": eigenvalues.min().item(),
        "gram_max_eigenvalue": eigenvalues.max().item()}


def measure(features, weights, originals):
    rgb = (affine_design(features) @ weights).permute(0, 3, 1, 2).contiguous()
    scores = metrics_numpy(rgb, originals)
    return {k: scores[k] for k in ("mse", "psnr_db", "gradient_mse")}


@torch.no_grad()
def probe(root, output, device_name="cuda:0"):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = configure_runtime(device_name, 311)
    protocol = json.loads((root / "protocol.json").read_text())
    for name, digest in protocol["source_sha256"].items():
        assert sha256(REPO / name) == digest, name
    selected = [r for r in protocol["runs"] if r["name"] == "attention_edge"]
    manifest = {"kind": "frozen_readout_probe", "training_set_only": True,
        "generalization_test": False, "end_to_end_retraining": False,
        "finite_shots": False, "source_protocol_sha256": sha256(root / "protocol.json"),
        "source_protocol": str(root / "protocol.json"), "seeds": protocol["seeds"],
        "edge_weight": .5, "ridge_penalty": 0,
        "readouts": {"original": 0, "affine_z": 18, "affine_z_zz": 48},
        "checkpoint_sha256": {r["directory"]: sha256(root / r["directory"] / "fit/checkpoints/last.pt") for r in selected},
        "probe_source_sha256": sha256(__file__)}
    write_json(output / "protocol.json", manifest)
    shutil.copy2(__file__, output / "readout_probe.py")
    rows = []
    for spec in selected:
        directory = root / spec["directory"]
        path = directory / "fit/checkpoints/last.pt"
        done = json.loads((directory / "COMPLETE.json").read_text())
        assert sha256(path) == done["checkpoint_sha256"]["last"]
        model, codes, saved = load_edge(path, device)
        # No gradients or updates; keep parameter flags matching V10 rendering
        # because they can affect the selected float32 CUDA matrix kernel.
        targets = torch.load(directory / "final/train.pt", weights_only=False)
        assert saved["dataset_indices"] == targets["dataset_indices"] == protocol["indices"]["train"]
        originals = targets["originals"]
        coordinates = pixel_coordinates(32, 32, device=device)
        matrix = model.quantum.circuit_matrix()
        observables = diagonal_observables(5, codes.dtype).to(device)
        measured = []
        norm_error = 0.
        for batch in codes.split(16):
            features, position = model.retrieve(batch, coordinates)
            initial = torch.cat((*features, position[None].expand(len(batch), -1, -1)), -1)
            state = model.amplitudes(initial.flatten(0, 1)) @ matrix
            probabilities = state.square()
            norm_error = max(norm_error, (probabilities.sum(-1) - 1).abs().max().item())
            measured.append((probabilities @ observables).reshape(len(batch), 32, 32, 15).cpu())
        features = torch.cat(measured)
        weights = torch.zeros(6, 3, dtype=torch.float64)
        weights[0, 0] = weights[3, 0] = weights[1, 1] = weights[4, 1] = weights[2, 2] = 1
        original_rgb = (affine_design(features[..., :5]) @ weights).permute(0, 3, 1, 2)
        error = (original_rgb - targets["reconstructions"].double()).abs().max().item()
        assert error < 3e-6, error
        assert norm_error < 3e-6, norm_error
        result = {"seed": spec["config"]["seed"], "n_images": len(codes),
            "original_rgb_max_discrepancy": error, "state_norm_max_error": norm_error,
            "original": measure(features[..., :5], weights, originals)}
        target = originals.permute(0, 2, 3, 1)
        previous_loss = result["original"]["mse"] + .5 * result["original"]["gradient_mse"]
        for name, width in (("affine_z", 5), ("affine_z_zz", 15)):
            learned, checks = fit_affine(features[..., :width], target)
            scores = measure(features[..., :width], learned, originals)
            objective = scores["mse"] + .5 * scores["gradient_mse"]
            assert objective <= previous_loss + 1e-10, (name, objective, previous_loss)
            previous_loss = objective
            result[name] = {**scores, "weights": learned.tolist(), **checks}
        assert sha256(path) == manifest["checkpoint_sha256"][spec["directory"]]
        rows.append(result)
        write_json(output / f"seed{result['seed']}.json", result)
        print(json.dumps({"seed": result["seed"], **{n: {k: result[n][k] for k in ("mse", "psnr_db", "gradient_mse")} for n in manifest["readouts"]}}), flush=True)
    summary = {name: {metric: {"mean": statistics.mean(r[name][metric] for r in rows),
        "sample_sd": statistics.stdev(r[name][metric] for r in rows)}
        for metric in ("mse", "psnr_db", "gradient_mse")} for name in manifest["readouts"]}
    write_json(output / "results.json", {"protocol_sha256": sha256(output / "protocol.json"), "runs": rows, "summary": summary})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    names = list(summary)
    for ax, metric, factor, label in ((axes[0], "gradient_mse", 1000, "Edge error × 1,000; lower is better"),
                                    (axes[1], "psnr_db", 1, "PSNR (dB); higher is better")):
        for row in rows:
            ax.plot(range(3), [row[n][metric] * factor for n in names], "o-", alpha=.35, label=f"Seed {row['seed']}")
        ax.plot(range(3), [summary[n][metric]["mean"] * factor for n in names], "D--", color="black", label="Mean")
        ax.set_xticks(range(3), ["Original\nfixed rule", "Learned\n5 Z means", "Learned\n5 Z + 10 ZZ"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Frozen V10 states: training-set readout diagnostic (no joint retraining)")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    probe(args.root, args.output, args.device)
