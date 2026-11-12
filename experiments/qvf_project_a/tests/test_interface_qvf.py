"""Physical, spatial and experiment-integrity checks for V11."""

import tempfile
import unittest
from pathlib import Path

import pennylane as qml
import torch
import torch.nn.functional as F

from experiments.qvf_project_a.edge_qvf import build_edge_decoder
from experiments.qvf_project_a.interface_qvf import (STAGES, angle_state, build_interface,
    diagonal_observables, patch_features, resources)
from experiments.qvf_project_a.interface_study import InterfaceTrainConfig, load_interface, train_interface
from experiments.qvf_project_a.models import pixel_coordinates
from experiments.qvf_project_a.token_qvf import ScaleQuantumLayer
from experiments.qvf_project_a.token_study import fit_completion
from experiments.qvf_project_a.direct_qvf import render_direct


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_patch_query_and_gradient_against_unfold_grid_sample(self):
        grid = torch.randn(2, 4, 4, 4, dtype=torch.float64, requires_grad=True)
        points = torch.tensor([[0., 0.], [.25, .7], [1., 1.], [.5, .5]], dtype=torch.float64)
        actual = patch_features(grid, points)
        unfolded = F.unfold(F.pad(grid, (1, 1, 1, 1), mode="replicate"), 3)
        unfolded = unfolded.reshape(2, 4, 9, 4, 4).permute(0, 2, 1, 3, 4).reshape(2, 36, 4, 4)
        query = (points.flip(-1) * 2 - 1)[None, None].expand(2, -1, -1, -1)
        expected = F.grid_sample(unfolded, query, mode="bilinear", padding_mode="border", align_corners=True).squeeze(2).transpose(1, 2)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        weight = torch.randn_like(actual)
        ga = torch.autograd.grad((actual * weight).sum(), grid, retain_graph=True)[0]
        ge = torch.autograd.grad((expected * weight).sum(), grid)[0]
        torch.testing.assert_close(ga, ge, atol=1e-11, rtol=1e-11)

    def test_angle_state_outputs_and_gradients_against_pennylane(self):
        quantum = ScaleQuantumLayer(5, 5, False).double()
        angles = (torch.randn(1, 5, 5, dtype=torch.float64) * .4).requires_grad_()
        device = qml.device("default.qubit", wires=5)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def reference(data, weights, controlled):
            for layer in range(5):
                for wire in range(5):
                    qml.RY(data[layer, wire], wires=wire)
                for control in range(5):
                    for target in range(5):
                        if control != target:
                            qml.CRY(controlled[control, target, layer], wires=[control, target])
                for wire in range(5):
                    qml.RY(weights[wire, layer], wires=wire)
            return qml.probs(wires=range(5))

        state = angle_state(angles, quantum)
        actual = state.square()[0] @ diagonal_observables(5)
        expected = reference(angles[0], quantum.weights, quantum.cond) @ diagonal_observables(5)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        objective = torch.randn(15, dtype=torch.float64)
        params = (angles, quantum.weights, quantum.cond)
        ga = torch.autograd.grad((actual * objective).sum(), params, retain_graph=True)
        ge = torch.autograd.grad((expected * objective).sum(), params)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(state.square().sum(1), torch.ones(1, dtype=torch.float64))
        independent = angle_state(angles, None).square() @ diagonal_observables(5)
        torch.testing.assert_close(independent[:, :5], angles.sum(1).cos(), atol=1e-12, rtol=1e-12)

    def test_baseline_and_initialization_controls(self):
        baseline = build_interface("attention_amp", 2).double()
        reference = build_edge_decoder("attention", 2).double()
        codes = torch.randn(2, 336, dtype=torch.float64)
        points = pixel_coordinates(32, 32).double()[::101]
        torch.testing.assert_close(baseline(codes, points), reference(codes, points), atol=0, rtol=0)
        models = {name: build_interface(name, 2) for names in STAGES.values() for name in names}
        self.assertGreaterEqual(resources(models["attention_amp_wide"], "attention_amp_wide")["shared_parameters"],
                                resources(models["patch_amp"], "patch_amp")["shared_parameters"])
        self.assertGreaterEqual(resources(models["patch_classical"], "patch_classical")["shared_parameters"],
                                resources(models["patch_angle"], "patch_angle")["shared_parameters"])
        self.assertGreaterEqual(resources(models["classical_grid_conv"], "classical_grid_conv")["shared_parameters"],
                                resources(models["quantum_grid_conv"], "quantum_grid_conv")["shared_parameters"])
        for key, value in models["quantum_grid_conv"].image_decoder.state_dict().items():
            torch.testing.assert_close(value, models["classical_grid_conv"].image_decoder.state_dict()[key], atol=0, rtol=0)
        for key, value in models["patch_angle_identity"].state_dict().items():
            torch.testing.assert_close(value, models["patch_angle"].state_dict()[key], atol=0, rtol=0)

    def test_all_variants_have_code_gradients_and_chunk_independence(self):
        points = pixel_coordinates(32, 32).double()[::127]
        for names in STAGES.values():
            for name in names:
                with self.subTest(architecture=name):
                    model = build_interface(name, 0).double()
                    codes = (torch.randn(2, 336, dtype=torch.float64) * .1).requires_grad_()
                    full = model(codes, points)
                    chunked = torch.cat([model(codes, p) for p in points.split(3)], 1)
                    torch.testing.assert_close(full, chunked, atol=1e-11, rtol=1e-11)
                    full.square().mean().backward()
                    self.assertGreater(codes.grad.abs().sum().item(), 0)
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                    if hasattr(model, "position_weights"):
                        self.assertGreater(model.position_weights.grad.abs().sum().item(), 0)

    def test_train_reload_and_hidden_target_isolation(self):
        for name in ("patch_amp", "attention_angle", "patch_angle", "quantum_grid_conv", "classical_grid_conv"):
            with self.subTest(architecture=name), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                cfg = InterfaceTrainConfig(architecture=name, n_train=2, batch_size=2,
                    steps=2, eval_every=2, pairs_per_orientation=2)
                images = torch.rand(2, 3, 32, 32)
                train_interface(cfg, images, [10, 11], directory, torch.device("cpu"))
                model, codes, _ = load_interface(directory / "fit/checkpoints/last.pt", torch.device("cpu"))
                saved = torch.load(directory / "final/train.pt", weights_only=False)
                torch.testing.assert_close(render_direct(model, codes, 2), saved["reconstructions"], atol=0, rtol=0)
        # Completion information isolation includes the conv head, whose receptive
        # field spans both observed and withheld output pixels.
        for name in ("patch_angle", "quantum_grid_conv"):
            with self.subTest(completion=name), tempfile.TemporaryDirectory() as tmp:
                images = torch.rand(2, 3, 32, 32)
                changed = images.clone()
                changed.flatten(2)[:, :, 256:] = torch.rand(2, 3, 768)
                spec = {"observed_pixels": list(range(256)), "unobserved_pixels": list(range(256, 1024)),
                        "steps": 2, "code_init_std": .01, "code_lr": .001, "code_norm_weight": .0001}
                payloads = []
                for index, target in enumerate((images, changed)):
                    directory = Path(tmp) / str(index)
                    directory.mkdir()
                    fit_completion(build_interface(name, 0), target, [20, 21], spec, 0, directory, torch.device("cpu"))
                    payloads.append(torch.load(directory / "completion.pt", weights_only=False))
                for key in ("codes", "reconstructions"):
                    torch.testing.assert_close(payloads[0][key], payloads[1][key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
