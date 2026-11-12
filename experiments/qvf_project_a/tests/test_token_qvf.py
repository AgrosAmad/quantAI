"""Numerical circuit, retrieval and checkpoint tests for the token experiment."""

import tempfile
import unittest
from pathlib import Path

import pennylane as qml
import torch

from experiments.qvf_project_a.direct_qvf import DirectConfig, render_direct, train_direct
from experiments.qvf_project_a.models import pixel_coordinates
from experiments.qvf_project_a.token_qvf import TokenConfig, ScaleQuantumLayer, build_token_decoder
from experiments.qvf_project_a.token_study import fit_completion, load_token
from experiments.qvf_project_a.train import configure_runtime
from experiments.qvf_repro.qvf_fast import FastQuantumLayer, parity_decode


class TokenTests(unittest.TestCase):
    def setUp(self):
        configure_runtime("cpu", 31)

    def test_circuit_and_gradients_match_independent_pennylane(self):
        for n in (5, 7):
            layer = ScaleQuantumLayer(n, 5, False).double()
            amps = torch.nn.functional.normalize(torch.rand(2 ** n, dtype=torch.float64), dim=0).requires_grad_()
            angles = (torch.randn(2, n, dtype=torch.float64) * .2).requires_grad_()
            weights = layer.weights.detach().clone().requires_grad_()
            cond = layer.cond.detach().clone().requires_grad_()
            dev = qml.device("default.qubit", wires=n)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def reference(a, u, w, c):
                qml.QubitStateVector(a, wires=range(n))
                for r in range(5):
                    if r in (2, 4):
                        for wire in range(n):
                            qml.RY(u[r // 2 - 1, wire], wires=wire)
                    for control in range(n):
                        for target in range(n):
                            if control != target:
                                qml.CRY(c[control, target, r], wires=[control, target])
                    for wire in range(n):
                        qml.RY(w[wire, r], wires=wire)
                return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(n))

            actual = layer(amps[None], angles[None])[0]
            expected = torch.stack(reference(amps, angles, weights, cond))
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
            coefficients = torch.arange(1, n + 1, dtype=torch.float64)
            actual_grad = torch.autograd.grad((actual * coefficients).sum(), (amps, angles, layer.weights, layer.cond))
            expected_grad = torch.autograd.grad((expected * coefficients).sum(), (amps, angles, weights, cond))
            for a, b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a, b, atol=5e-11, rtol=5e-11)
            original = FastQuantumLayer(n, 5, False).double()
            original.load_state_dict(layer.state_dict())
            torch.testing.assert_close(layer(amps[None], torch.zeros(1, 2, n, dtype=torch.float64)),
                original(amps[None], per_sample=True), atol=2e-12, rtol=2e-12)

    def test_retrieval_chunking_gradients_and_paired_initialization(self):
        points = pixel_coordinates(32, 32).double()[::43]
        codes = (torch.randn(2, 336, dtype=torch.float64) * .1).requires_grad_()
        previous = None
        for retrieval, encoding in (("bilinear", "single"), ("attention", "single"), ("attention", "staged")):
            model = build_token_decoder(TokenConfig(retrieval=retrieval, encoding=encoding), 11).double()
            whole = model(codes, points)
            chunks = torch.cat([model(codes, p) for p in points.split(5)], 1)
            torch.testing.assert_close(whole, chunks, atol=1e-12, rtol=1e-12)
            gradients = torch.autograd.grad(whole.square().sum(), (codes, *model.parameters()))
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(gradients[0].abs().sum().item(), 0)
            if previous is not None:
                torch.testing.assert_close(model.circuit.weights, previous.circuit.weights, atol=0, rtol=0)
                for a, b in zip(model.values.parameters(), previous.values.parameters()):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            previous = model
        with self.assertRaises(ValueError):
            TokenConfig(latent_dim=64)
        z = torch.arange(7, dtype=torch.float64)[None]
        torch.testing.assert_close(parity_decode(z, 3), torch.tensor([[4.5, 2.5, 3.5]], dtype=torch.float64))

    def test_training_reload_and_completion_uses_only_observed_targets(self):
        cfg = DirectConfig(model=TokenConfig(encoding="staged"), seed=31,
            n_train=4, n_val=2, batch_size=2, pixels_per_image=13, steps=2, eval_every=1)
        images = torch.rand(4, 3, 32, 32)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train_direct(cfg, images, [2, 4, 8, 9], directory, torch.device("cpu"),
                decoder_factory=lambda c: build_token_decoder(c, 31), checkpoint_kind="token_qvf")
            model, codes, checkpoint = load_token(directory / "fit/checkpoints/last.pt", "cpu")
            saved = torch.load(directory / "final/train.pt", weights_only=False)
            torch.testing.assert_close(render_direct(model, codes, 2), saved["reconstructions"], atol=0, rtol=0)
            initial = torch.load(directory / "fit/checkpoints/step0.pt", weights_only=False)
            for key in ("queries.0.weight", "uploads.0.weight", "quantum.weights"):
                self.assertFalse(torch.equal(checkpoint["decoder_state"][key], initial["decoder_state"][key]))
            spec = {"observed_pixels": list(range(128)), "unobserved_pixels": list(range(128, 1024)),
                    "steps": 2, "code_lr": .001, "code_init_std": .01, "code_norm_weight": .0001}
            first, second = directory / "first", directory / "second"
            first.mkdir(); second.mkdir()
            heldout = images[:2].clone()
            fit_completion(model, heldout, [21, 22], spec, 31, first, "cpu")
            heldout.flatten(2)[:, :, 128:] = 1 - heldout.flatten(2)[:, :, 128:]
            fit_completion(model, heldout, [21, 22], spec, 31, second, "cpu")
            a = torch.load(first / "completion.pt", weights_only=False)
            b = torch.load(second / "completion.pt", weights_only=False)
            torch.testing.assert_close(a["codes"], b["codes"], atol=0, rtol=0)
            torch.testing.assert_close(a["reconstructions"], b["reconstructions"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
