"""Independent physical measurement and full-objective checks for the probe."""

import unittest

import pennylane as qml
import torch

from experiments.qvf_project_a.readout_probe import affine_design, diagonal_observables, fit_affine


class ReadoutProbeTests(unittest.TestCase):
    def test_pair_observables_against_pennylane(self):
        generator = torch.Generator().manual_seed(108)
        state = torch.randn(8, generator=generator, dtype=torch.float64)
        state /= state.norm()
        device = qml.device("default.qubit", wires=3)

        @qml.qnode(device)
        def reference():
            qml.QubitStateVector(state.numpy(), wires=range(3))
            return [qml.expval(qml.PauliZ(i)) for i in range(3)] + [
                qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in ((0, 1), (0, 2), (1, 2))]

        actual = state.square() @ diagonal_observables(3)
        expected = torch.tensor([float(value) for value in reference()], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        # Equal marginals need not imply equal correlations.
        bell = torch.tensor([.5, 0., 0., .5], dtype=torch.float64) @ diagonal_observables(2)
        uniform = torch.full((4,), .25, dtype=torch.float64) @ diagonal_observables(2)
        torch.testing.assert_close(bell[:2], uniform[:2])
        self.assertEqual(bell[2].item(), 1.)
        self.assertEqual(uniform[2].item(), 0.)

    def test_solution_stationary_for_explicit_pixel_and_edge_objective(self):
        generator = torch.Generator().manual_seed(29)
        features = torch.randn(4, 7, 9, 5, generator=generator, dtype=torch.float64)
        target = torch.randn(4, 7, 9, 3, generator=generator, dtype=torch.float64)
        weights, _ = fit_affine(features, target)
        weights.requires_grad_(True)
        error = affine_design(features) @ weights - target
        loss = error.square().mean() + .25 * (
            error.diff(dim=1).square().mean() + error.diff(dim=2).square().mean())
        loss.backward()
        self.assertLess(weights.grad.abs().max().item(), 1e-12)


if __name__ == "__main__":
    unittest.main()
