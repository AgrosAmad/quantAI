import itertools
import math
import unittest
import numpy as np
import pennylane as qml
import torch
from experiments.qvf_project_a.train import configure_runtime
from experiments.qvf_project_z.spatial_models import build_model, resources
from experiments.qvf_project_z.spatial_gates import (primitive_ops, simulate_density,
    gate_resources, marginal_readout, observable_rgb, inputs_and_probabilities, shot_rgb,
    shot_mse_moments)


class SpatialGatesTests(unittest.TestCase):
    def setUp(self):
        configure_runtime('cpu', 13)

    def test_explicit_preparation_counts_and_execution(self):
        for q in (5, 7):
            for family in ('attention_chain', 'attention_star', 'spatial_star'):
                model = build_model(f'{family}_q{q}', 2).double()
                codes, xy = torch.randn(1, 336, dtype=torch.float64), torch.rand(2, 2, dtype=torch.float64)
                kind, data, p = inputs_and_probabilities(model, codes, xy)
                ops = primitive_ops(kind, data, model.quantum)
                rebuilt = simulate_density(ops, 2, q)
                np.testing.assert_allclose(rebuilt, p, atol=2e-12, rtol=2e-12)
                analytic, actual = resources(model), gate_resources(ops, q)
                for gate in ('RY', 'CNOT'):
                    self.assertEqual(analytic[f'primitive_{gate.lower()}_per_query'], actual['all_to_all_primitives'][gate])
                if kind == 'amplitude':
                    @qml.qnode(qml.device('default.qubit', wires=q))
                    def reference():
                        qml.MottonenStatePreparation(data[0], wires=range(q))
                        for gate, wires, angle in ops[2**(q+1)-3:]:
                            qml.RY(float(angle), wires=wires) if gate == 'RY' else qml.CNOT(wires=wires)
                        return qml.probs(wires=range(q))
                    np.testing.assert_allclose(reference(), p[0], atol=2e-12, rtol=2e-12)

    def test_seven_wire_noise_against_independent_kraus_channel(self):
        ops = [('RY', (0,), np.array(.7)), ('CNOT', (0, 6), None), ('RY', (2,), np.array(.4))]
        p1, p2, readout = .03, .07, .02
        paulis = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1.])]
        matrices = [np.kron(a, b) for a, b in itertools.product(paulis, repeat=2)]
        kraus = [np.sqrt(1-p2)*matrices[0]] + [np.sqrt(p2/15)*x for x in matrices[1:]]
        @qml.qnode(qml.device('default.mixed', wires=7))
        def reference():
            for gate, wires, angle in ops:
                if gate == 'RY':
                    qml.RY(float(angle), wires=wires); qml.DepolarizingChannel(p1, wires=wires)
                else:
                    qml.CNOT(wires=wires); qml.QubitChannel(kraus, wires=wires)
            for w in range(5):
                qml.BitFlip(readout, wires=w)
            return qml.probs(wires=range(7))
        np.testing.assert_allclose(simulate_density(ops, 1, 7, p1=p1, p2=p2, readout=readout)[0], reference(), atol=2e-12, rtol=2e-12)

    def test_marginal_shots_retain_rgb_correlations(self):
        p = np.zeros((12000, 128)); p[:, 0] = .3; p[:, 124:128] = .175
        np.testing.assert_allclose(p @ observable_rgb(7), marginal_readout(p) @ observable_rgb(5), atol=1e-14)
        sample = shot_rgb(marginal_readout(p), 64, 7)
        np.testing.assert_array_equal(sample[:, 0], sample[:, 1])
        np.testing.assert_array_equal(sample[:, 0], 2*sample[:, 2])
        self.assertAlmostEqual(sample[:, 2].mean(), -.4, delta=.004)
        self.assertAlmostEqual(sample[:, 2].var(), .84/64, delta=.0005)

    def test_terminal_work_wire_rotations_are_unobservable(self):
        # Two stored entries at seven qubits are redundant for the fixed RGB
        # measurement. Count these honestly in the audit, without changing the
        # frozen training implementation or its conservative gate execution.
        for family in ('attention_chain', 'attention_star', 'spatial_star'):
            m = build_model(f'{family}_q7', 2).double()
            codes, xy = torch.randn(1, 336, dtype=torch.float64), torch.rand(7, 2, dtype=torch.float64)
            before = m(codes, xy)
            grad = torch.autograd.grad(before.sum(), m.quantum.weights)[0]
            torch.testing.assert_close(grad[5:, -1], torch.zeros(2, dtype=torch.float64), atol=2e-13, rtol=0)
            with torch.no_grad():
                m.quantum.weights[5:, -1] += torch.tensor([.7, -1.1], dtype=torch.float64)
            torch.testing.assert_close(m(codes, xy), before, atol=2e-13, rtol=2e-13)

    def test_exact_mse_variance_against_binomial_enumeration(self):
        p = np.zeros((1, 128)); p[:, 0] = .3; p[:, 127] = .7
        target = np.array([[.2, .8, .4]])
        for shots in (1, 2, 7, 64):
            k = np.arange(shots+1)
            mass = np.array([math.comb(shots, i)*.3**i*.7**(shots-i) for i in k])
            means = (2*k[:, None]/shots-1)*np.array([2., 2., 1.])
            mse = ((means-target)**2).mean(-1)
            exact_mean = mass @ mse
            exact_variance = mass @ (mse-exact_mean)**2
            mean, variance = shot_mse_moments(p, target, shots)
            np.testing.assert_allclose(mean[0], exact_mean, atol=2e-13, rtol=2e-13)
            np.testing.assert_allclose(variance[0], exact_variance, atol=2e-13, rtol=2e-13)


if __name__ == '__main__':
    unittest.main()
