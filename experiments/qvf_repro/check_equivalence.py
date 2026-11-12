"""Check the fast simulator against the official PennyLane circuit, and time both.

Usage:
    python check_equivalence.py /path/to/official/src/Images/QVF_for_image.py

The official file is loaded by path (it is not put on sys.path). Its QuantumLayer is
used as is, except for the rotation indexing, which raises an IndexError in the
released code (weights[:, :, r] on a 2D tensor). The fix used here is weights[:, r].
"""

import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import pennylane as qml
import torch
import torch.nn.functional as F

from qvf_fast import FastQuantumLayer, QVFImage


def load_official(path):
    spec = importlib.util.spec_from_file_location("qvf_official", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fixed_layer(official):
    class OfficialQuantumLayerFixed(official.QuantumLayer):
        def _qnode(self, inputs):
            n = self.n_qubits

            @qml.qnode(self.dev, interface="torch")
            def circuit(inputs, weights, cond):
                self._embed(inputs)
                if self.use_identity_block:
                    for r in range(self.repetitions):
                        self._entangle(cond[:n, :, r])
                        self._rotations(weights[:n, r])
                        qml.adjoint(self._rotations)(weights[n:, r])
                        qml.adjoint(self._entangle)(cond[n:, :, r])
                else:
                    for r in range(self.repetitions):
                        self._entangle(cond[:, :, r])
                        self._rotations(weights[:, r])
                return [qml.expval(qml.PauliZ(i)) for i in range(n)]

            return circuit(inputs, self.weights, self.cond)

        def forward(self, x):
            return torch.stack(list(self._qnode(x)), dim=0).reshape(self.n_qubits, -1).T

    return OfficialQuantumLayerFixed


def random_amplitudes(batch, dim, dtype, device, gen):
    logits = torch.randn(batch, dim, dtype=dtype, device=device, generator=gen)
    return torch.sqrt(torch.softmax(logits, dim=-1))


def equivalence(official_layer_cls, device):
    rows = []
    for n, reps in [(2, 1), (3, 2), (4, 2), (5, 5), (6, 3)]:
        for identity in (True, False):
            for trained_like in (False, True):
                gen = torch.Generator(device="cpu").manual_seed(1000 * n + 10 * reps + identity)
                fast = FastQuantumLayer(n, reps, identity, gaussian_std=0.1).double()
                with torch.no_grad():
                    if trained_like:  # generic angles, so the identity block is no longer the identity
                        fast.weights.copy_(torch.randn(fast.weights.shape, generator=gen, dtype=torch.float64) * 2.0)
                        fast.cond.copy_(torch.randn(fast.cond.shape, generator=gen, dtype=torch.float64) * 2.0)
                ref = official_layer_cls(n, reps, 0.0, False, identity, 0.1, torch.device("cpu"))
                ref.weights = torch.nn.Parameter(fast.weights.detach().clone())
                ref.cond = torch.nn.Parameter(fast.cond.detach().clone())

                amps = random_amplitudes(64, 2 ** n, torch.float64, "cpu", gen).requires_grad_(True)
                out_ref = ref(amps)
                out_fast = fast(amps)
                out_slow = fast(amps, per_sample=True)

                # Gradients with respect to the angles must agree too.
                g_ref = torch.autograd.grad(out_ref.pow(2).sum(), [ref.weights, ref.cond])
                g_fast = torch.autograd.grad(out_fast.pow(2).sum(), [fast.weights, fast.cond])

                m = fast.circuit_matrix()
                rows.append({
                    "n_qubits": n, "repetitions": reps, "identity_block": identity, "generic_angles": trained_like,
                    "max_abs_diff_vs_pennylane": (out_fast - out_ref).abs().max().item(),
                    "max_abs_diff_matrix_vs_per_sample": (out_fast - out_slow).abs().max().item(),
                    "max_abs_grad_diff_weights": (g_fast[0] - g_ref[0]).abs().max().item(),
                    "max_abs_grad_diff_cond": (g_fast[1] - g_ref[1]).abs().max().item(),
                    "orthogonality_error": (m @ m.T - torch.eye(2 ** n, dtype=m.dtype)).abs().max().item(),
                })
    return rows


def is_oom(err):
    return "out of memory" in str(err).lower()


def measure(step, repeats, device):
    """Mean seconds per call of step(), and the peak GPU memory in MiB (None on CPU)."""
    on_gpu = device.type == "cuda"
    if on_gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    step()  # warm-up
    if on_gpu:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        step()
    if on_gpu:
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - t0) / repeats
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if on_gpu else None
    return seconds, peak


def measure_largest_batch(make_step, batch, repeats, device, min_batch=100):
    """Time one training step at `batch`. On a GPU out-of-memory error, halve the batch and retry."""
    b = batch
    while b >= min_batch:
        result = None
        try:
            result = measure(make_step(b), repeats, device)
        except RuntimeError as err:
            if not is_oom(err):
                raise
        if result is not None:
            return {"batch": b, "ms": 1e3 * result[0], "peak_gpu_mib": result[1]}
        gc.collect()
        torch.cuda.empty_cache()
        b //= 2
    return None


TIMING_CONFIGS = {
    "demo config (4 qubits, depth 2, width 256)": {"hid_dim": 256, "out_dim": 16, "num_repetations": 2},
    "paper config (5 qubits, depth 5, width 128)": {"hid_dim": 128, "out_dim": 32, "num_repetations": 5},
}


def timing(official, official_layer_cls, device, batch):
    """One full training step (forward, loss, backward, Adam update) of the whole model.

    `batch` is the number of (image, pixel) rows; 51,200 is one batch of 50 CIFAR images.
    The official model is timed at the largest batch that fits in GPU memory, and its
    time is scaled linearly to the full batch when it had to be reduced.
    """
    official.QuantumLayer = official_layer_cls
    rows = []
    for label, override in TIMING_CONFIGS.items():
        cfg = {"latent_size": 64, "L_embed": 6, "num_hidden_layers_energy": 3, "skip_connection": False,
               "use_parity": True, "output_channels": 3, "use_identity_block": True, "gaussian_std": 0.1, **override}

        def make_step_factory(build):
            def make_step(b):
                torch.manual_seed(0)
                model = build().to(device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                x = torch.randn(b, 64 + 26, device=device)
                y = torch.rand(b, 3, device=device)

                def step():
                    opt.zero_grad()
                    loss = F.mse_loss(model(x).float(), y)
                    loss.backward()
                    opt.step()
                return step
            return make_step

        fast = measure_largest_batch(make_step_factory(lambda: QVFImage(cfg)), batch, 20, device)
        ref = measure_largest_batch(make_step_factory(lambda: official.ImageRegressor(cfg, device)), batch, 2, device)
        ref_ms_full = ref["ms"] * batch / ref["batch"]
        rows.append({
            "config": label, "device": str(device), "batch": batch,
            "fast_ms": fast["ms"], "fast_peak_gpu_mib": fast["peak_gpu_mib"],
            "official_batch_that_fits": ref["batch"], "official_ms_at_that_batch": ref["ms"],
            "official_peak_gpu_mib_at_that_batch": ref["peak_gpu_mib"],
            "official_ms_scaled_to_full_batch": ref_ms_full,
            "official_fits_full_batch": ref["batch"] == batch,
            "speedup": ref_ms_full / fast["ms"],
        })
        print(f"  timed {label} on {device}: fast {fast['ms']:.2f} ms, official {ref_ms_full:.0f} ms "
              f"(measured at batch {ref['batch']})", flush=True)
    return rows


def full_model_check(official, official_layer_cls):
    """Load one set of weights into the official ImageRegressor and into QVFImage, compare outputs."""
    cfg = {"latent_size": 64, "L_embed": 6, "hid_dim": 256, "out_dim": 16, "num_hidden_layers_energy": 3,
           "skip_connection": False, "use_parity": True, "output_channels": 3, "use_identity_block": True,
           "gaussian_std": 0.1, "num_repetations": 2}
    torch.manual_seed(0)
    official.QuantumLayer = official_layer_cls
    ref = official.ImageRegressor(cfg, torch.device("cpu")).double()
    mine = QVFImage(cfg).double()
    missing = mine.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(256, 64 + 26, dtype=torch.float64)
    with torch.no_grad():
        diff = (mine(x) - ref(x).double()).abs().max().item()
    return {"state_dict_loads_strict": str(missing), "max_abs_diff_full_model": diff}


if __name__ == "__main__":
    official = load_official(sys.argv[1])
    layer_cls = make_fixed_layer(official)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = Path(__file__).parent / "results" / "equivalence.json"
    out.parent.mkdir(exist_ok=True)

    report = {
        "versions": {"torch": torch.__version__, "pennylane": qml.__version__},
        "equivalence_float64": equivalence(layer_cls, device),
        "full_model": full_model_check(official, layer_cls),
    }
    out.write_text(json.dumps(report, indent=2))
    rows = report["equivalence_float64"]
    print(f"configs checked: {len(rows)}")
    print(f"worst output difference vs PennyLane: {max(r['max_abs_diff_vs_pennylane'] for r in rows):.3e}")
    print(f"worst gradient difference vs PennyLane: "
          f"{max(max(r['max_abs_grad_diff_weights'], r['max_abs_grad_diff_cond']) for r in rows):.3e}")
    print(f"worst single-matrix vs per-sample difference: {max(r['max_abs_diff_matrix_vs_per_sample'] for r in rows):.3e}")
    print(f"full model: {report['full_model']}", flush=True)

    # 51,200 rows is one batch of 50 CIFAR images. The official code is timed on the GPU (the
    # paper's setting) and on the CPU, because its backward pass is much slower on the GPU.
    report["timing_training_step"] = timing(official, layer_cls, device, 51200)
    if device.type == "cuda":
        report["timing_training_step"] += timing(official, layer_cls, torch.device("cpu"), 51200)
    out.write_text(json.dumps(report, indent=2))
    for r in report["timing_training_step"]:
        note = "" if r["official_fits_full_batch"] else f" (official ran out of GPU memory above batch {r['official_batch_that_fits']}; time scaled)"
        print(f"{r['config']}, {r['device']}, batch {r['batch']}: official {r['official_ms_scaled_to_full_batch']:.0f} ms, "
              f"fast {r['fast_ms']:.2f} ms, speedup {r['speedup']:.0f}x{note}")
    print(f"saved {out}")
