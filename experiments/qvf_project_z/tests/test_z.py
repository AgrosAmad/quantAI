import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pennylane as qml
import torch
from experiments.qvf_project_a.token_qvf import TokenConfig, build_token_decoder
from experiments.qvf_project_a.token_study import fit_completion
from experiments.qvf_project_a.direct_qvf import render_direct
from experiments.qvf_project_a.train import configure_runtime
from experiments.qvf_project_z.models import ALL, LATENT, build_model, resources, LocalQuantum
from experiments.qvf_project_z.feasibility import primitive_ops, simulate_density, depolarize, shot_rgb, observable_rgb
from experiments.qvf_project_z.study import ZTrainConfig, train_model, load_model


class ProjectZTests(unittest.TestCase):
    def setUp(self):
        configure_runtime('cpu', 11)

    def test_initial_functions_and_spatial_scope(self):
        code = torch.randn(2, 336, dtype=torch.float64) * .2
        xy = torch.rand(13, 2, dtype=torch.float64)
        ref = build_token_decoder(TokenConfig(), 2).double()
        for name in LATENT:
            model = build_model(name, 2).double()
            torch.testing.assert_close(model(code, xy), ref(code, xy), atol=0, rtol=0)
        conv, point = build_model('latent_conv', 2), build_model('latent_pointwise', 2)
        self.assertGreater(resources(point, '')['shared_parameters'], resources(conv, '')['shared_parameters'])
        for model, spatial in ((conv, True), (point, False)):
            layer = model.refiners[2]
            torch.nn.init.normal_(layer.last.weight, std=.1)
            grid = torch.zeros(1, 4, 8, 8)
            changed = grid.clone(); changed[0, 0, 3, 3] = 1
            difference = layer(changed) - layer(grid)
            self.assertEqual(bool(difference[0, :, 3, 4].abs().sum() > 0), spatial)

    def test_local_quantum_against_pennylane_and_shared_control(self):
        a = build_model('attention_local_shared', 1).double().quantum
        b = build_model('attention_local_untied', 1).double().quantum
        vector = torch.rand(32, dtype=torch.float64); vector /= vector.norm()
        torch.testing.assert_close(a(vector[None]), b(vector[None]), atol=0, rtol=0)
        @qml.qnode(qml.device('default.qubit', wires=5), interface='torch', diff_method='backprop')
        def reference(weights, filters):
            qml.StatePrep(vector, wires=range(5))
            for layer in range(5):
                for u, v in LocalQuantum.pairs:
                    f = filters[layer, 0]
                    qml.RY(f[0], wires=u); qml.RY(f[1], wires=v)
                    qml.CRY(f[2], wires=[u, v]); qml.CRY(f[3], wires=[v, u])
                for wire in range(5):
                    qml.RY(weights[wire, layer], wires=wire)
            return tuple(qml.expval(qml.PauliZ(w)) for w in range(5))
        actual, expected = a(vector[None])[0], torch.stack(reference(a.weights, a.filters))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        ga = torch.autograd.grad(actual.sum(), (a.weights, a.filters), retain_graph=True)
        ge = torch.autograd.grad(expected.sum(), (a.weights, a.filters))
        for x, y in zip(ga, ge):
            torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-11)

    def test_explicit_preparation_and_gate_simulation(self):
        amplitudes = np.random.default_rng(1).random((3, 32))
        amplitudes[1] = 0; amplitudes[1, 0] = 1
        amplitudes /= np.linalg.norm(amplitudes, axis=-1, keepdims=True)
        for name in ('attention_amp', 'attention_local_shared'):
            model = build_model(name, 2).double()
            ops = primitive_ops('amplitude', amplitudes, model.quantum)
            expected = (torch.from_numpy(amplitudes) @ model.quantum.circuit_matrix()).square().detach().numpy()
            np.testing.assert_allclose(simulate_density(ops, 3), expected, atol=1e-12, rtol=1e-12)
            # Independent PL standard preparation and decomposed processing for one input.
            @qml.qnode(qml.device('default.qubit', wires=5))
            def reference():
                qml.MottonenStatePreparation(amplitudes[0], wires=range(5))
                for gate, wires, angle in ops[61:]:
                    qml.RY(float(angle), wires=wires) if gate == 'RY' else qml.CNOT(wires=wires)
                return qml.probs(wires=range(5))
            np.testing.assert_allclose(reference(), expected[0], atol=1e-12, rtol=1e-12)

    def test_noise_against_kraus_reference_and_parameter_shift(self):
        ops = [('RY', (0,), np.array(.7)), ('CNOT', (0, 4), None), ('RY', (2,), np.array(.4))]
        p1, p2, readout = .03, .07, .02
        paulis = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1.])]
        matrices = [np.kron(a, b) for a, b in itertools.product(paulis, repeat=2)]
        kraus = [np.sqrt(1 - p2) * matrices[0]] + [np.sqrt(p2 / 15) * x for x in matrices[1:]]
        @qml.qnode(qml.device('default.mixed', wires=5))
        def reference():
            for gate, wires, angle in ops:
                if gate == 'RY':
                    qml.RY(float(angle), wires=wires); qml.DepolarizingChannel(p1, wires=wires)
                else:
                    qml.CNOT(wires=wires); qml.QubitChannel(kraus, wires=wires)
            for w in range(5):
                qml.BitFlip(readout, wires=w)
            return qml.probs(wires=range(5))
        np.testing.assert_allclose(simulate_density(ops, 1, p1=p1, p2=p2, readout=readout)[0], reference(), atol=1e-12, rtol=1e-12)
        # Shift one primitive input gate; derivative agrees with independent QNode autograd.
        @qml.qnode(qml.device('default.qubit', wires=5), interface='torch', diff_method='backprop')
        def scalar(angle):
            qml.RY(angle, wires=0); qml.CNOT(wires=[0, 4])
            return qml.expval(qml.PauliZ(0))
        theta = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(scalar(theta), theta)[0].item()
        shifted = []
        for shift in (np.pi / 2, -np.pi / 2):
            p = simulate_density([('RY', (0,), np.array(.7 + shift)), ('CNOT', (0, 4), None)], 1)
            shifted.append(float(p @ (1 - 2 * ((np.arange(32) >> 4) & 1))))
        self.assertAlmostEqual((shifted[0] - shifted[1]) / 2, gradient, places=12)

    def test_joint_shots_moments(self):
        p = np.zeros((12000, 32)); p[:, 0] = .3; p[:, 31] = .7
        sample = shot_rgb(p, 64, 7)
        # All wires are perfectly correlated in this distribution.
        np.testing.assert_array_equal(sample[:, 0], sample[:, 1])
        np.testing.assert_array_equal(sample[:, 0], 2 * sample[:, 2])
        self.assertAlmostEqual(sample[:, 2].mean(), -.4, delta=.004)
        self.assertAlmostEqual(sample[:, 2].var(), .84 / 64, delta=.0005)

    def test_training_reload_and_completion_isolation(self):
        for name in ('latent_conv', 'latent_pointwise', 'attention_local_shared'):
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                cfg = ZTrainConfig(architecture=name, n_train=2, batch_size=2, steps=2, eval_every=2, pairs_per_orientation=2)
                originals = torch.rand(2, 3, 32, 32)
                train_model(cfg, originals, [1, 2], folder, torch.device('cpu'))
                model, codes, _ = load_model(folder / 'fit/checkpoints/last.pt', torch.device('cpu'))
                saved = torch.load(folder / 'final/train.pt', weights_only=False)
                torch.testing.assert_close(render_direct(model, codes, 2), saved['reconstructions'], atol=0, rtol=0)
        for name in ('latent_conv', 'attention_local_shared'):
            with tempfile.TemporaryDirectory() as tmp:
                target = torch.rand(2, 3, 32, 32)
                altered = target.clone(); altered.flatten(2)[:, :, 256:] = torch.rand(2, 3, 768)
                spec = {'observed_pixels': list(range(256)), 'unobserved_pixels': list(range(256, 1024)),
                        'steps': 2, 'code_init_std': .01, 'code_lr': .001, 'code_norm_weight': .0001}
                outputs = []
                for i, images in enumerate((target, altered)):
                    folder = Path(tmp) / str(i); folder.mkdir()
                    fit_completion(build_model(name, 1), images, [1, 2], spec, 1, folder, torch.device('cpu'))
                    outputs.append(torch.load(folder / 'completion.pt', weights_only=False))
                torch.testing.assert_close(outputs[0]['codes'], outputs[1]['codes'], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
