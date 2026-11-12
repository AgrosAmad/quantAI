"""Independent circuit and spatial semantics checks for Z4."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pennylane as qml
import torch
from torch.nn import functional as F
from experiments.qvf_project_a.direct_qvf import render_direct
from experiments.qvf_project_a.token_study import fit_completion
from experiments.qvf_project_a.train import configure_runtime
from experiments.qvf_project_z.models import build_model as old_model
from experiments.qvf_project_z.spatial_models import build_model, model_specs, resources
from experiments.qvf_project_z.spatial_study import ZTrainConfig, train_model, load_model


class SpatialTests(unittest.TestCase):
    def setUp(self):
        configure_runtime('cpu', 13)

    def test_old_five_qubit_chain_is_exact(self):
        a, b = old_model('attention_local_shared', 2).double(), build_model('attention_chain_q5', 2).double()
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
        code, xy = torch.randn(2, 336, dtype=torch.float64), torch.rand(9, 2, dtype=torch.float64)
        pa, pb = a(code, xy), b(code, xy)
        torch.testing.assert_close(pa, pb, atol=0, rtol=0)
        ga = torch.autograd.grad(pa.square().sum(), tuple(a.parameters()))
        gb = torch.autograd.grad(pb.square().sum(), tuple(b.parameters()))
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_slot_sampling_against_grid_sample_and_isolation(self):
        code = torch.randn(2, 336, dtype=torch.float64)
        xy = torch.tensor([[0., 0.], [.37, .61], [1., 1.]], dtype=torch.float64)
        for family in ('spatial_star', 'center_star'):
            m = build_model(f'{family}_q5').double()
            expected, start = [], 0
            for k, size in enumerate((2, 4, 8)):
                count = 4 * size * size
                grid = code[:, start:start+count].reshape(2, 4, size, size)
                grid = m.values[k](grid.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(grid)
                offset = m.offsets.double() if family == 'spatial_star' else torch.zeros_like(m.offsets).double()
                locations = (xy[:, None] + offset[None] / (size - 1)).flip(-1) * 2 - 1
                sampled = F.grid_sample(grid, locations[None].expand(2, -1, -1, -1),
                    mode='bilinear', padding_mode='border', align_corners=True).permute(0, 2, 3, 1)
                expected.append(sampled); start += count
            torch.testing.assert_close(m.slot_features(code, xy), torch.cat(expected, -1), atol=1e-14, rtol=1e-14)
        m = build_model('spatial_star_q7').double()
        code = torch.zeros(1, 336, dtype=torch.float64)
        changed = code.clone()
        changed[0, 80 + 3 * 8 + 2] = 1.  # fine-scale left neighbour of (3,3)
        xy = torch.tensor([[3/7, 3/7]], dtype=torch.float64)
        difference = (m.encoded(changed, xy) - m.encoded(code, xy)).abs().sum((0, 1))
        self.assertGreater(difference[0].item(), 0)
        torch.testing.assert_close(difference[1:], torch.zeros(4, dtype=torch.float64), atol=1e-14, rtol=0)

    def test_circuits_and_gradients_against_pennylane(self):
        for q in (5, 7):
            for family in ('attention_chain', 'attention_star', 'spatial_star'):
                m = build_model(f'{family}_q{q}', 1).double()
                c = m.quantum
                slot = family == 'spatial_star'
                data = torch.randn(5, 5, dtype=torch.float64) * .2 if slot else torch.rand(2**q, dtype=torch.float64)
                if not slot:
                    data /= data.norm()
                data.requires_grad_()
                @qml.qnode(qml.device('default.qubit', wires=q), interface='torch', diff_method='backprop')
                def reference(data, weights, filters):
                    if slot:
                        for wire in range(5):
                            qml.RY(np.pi / 2, wires=wire)
                    else:
                        qml.StatePrep(data, wires=range(q))
                    for stage in range(5):
                        if slot:
                            for wire in range(5):
                                qml.RY(data[stage, wire], wires=wire)
                        for a, b in c.pairs:
                            f = filters[stage, 0]
                            qml.RY(f[0], wires=a); qml.RY(f[1], wires=b)
                            qml.CRY(f[2], wires=[a, b]); qml.CRY(f[3], wires=[b, a])
                        for wire in range(q):
                            qml.RY(weights[wire, stage], wires=wire)
                    return qml.probs(wires=range(q))
                actual = c.slot_state(data[None]).square()[0] if slot else (data @ c.circuit_matrix()).square()
                expected = reference(data, c.weights, c.filters)
                torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
                coefficients = torch.arange(2**q, dtype=torch.float64).sin()
                ga = torch.autograd.grad(actual @ coefficients, (data, c.weights, c.filters), retain_graph=True)
                ge = torch.autograd.grad(expected @ coefficients, (data, c.weights, c.filters))
                for x, y in zip(ga, ge):
                    torch.testing.assert_close(x, y, atol=2e-12, rtol=2e-12)
                if slot and q == 7:
                    self.assertGreater(ga[1][5:].abs().sum().item(), 0.)

    def test_control_initialization_and_readout(self):
        for q in (5, 7):
            a, b = build_model(f'spatial_star_q{q}', 2), build_model(f'center_star_q{q}', 2)
            for k, v in a.state_dict().items():
                torch.testing.assert_close(v, b.state_dict()[k], atol=0, rtol=0)
            code, xy = torch.randn(2, 336) * .2, torch.rand(7, 2)
            probs = a.probabilities(code, xy)
            z = probs @ a.quantum.signs.float()
            reference = torch.stack((z[..., 0] + z[..., 3], z[..., 1] + z[..., 4], z[..., 2]), -1)
            torch.testing.assert_close(a(code, xy), reference, atol=0, rtol=0)
        a, b = build_model('spatial_star_q5', 2), build_model('spatial_star_q7', 2)
        for k, v in a.state_dict().items():
            if not k.startswith('quantum.'):
                torch.testing.assert_close(v, b.state_dict()[k], atol=0, rtol=0)
        torch.testing.assert_close(a.quantum.weights, b.quantum.weights[:5], atol=0, rtol=0)
        wide, seven = resources(build_model('attention_chain_wide_q5')), resources(build_model('attention_chain_q7'))
        self.assertGreaterEqual(wide['classical_parameters'], seven['classical_parameters'])
        self.assertGreaterEqual(wide['shared_parameters'], seven['shared_parameters'])

    def test_train_reload_and_hidden_target_isolation(self):
        for name in ('attention_chain_q7', 'spatial_star_q7', 'center_star_q5'):
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                cfg = ZTrainConfig(architecture=name, n_train=2, batch_size=2, steps=2, eval_every=2, pairs_per_orientation=2)
                target = torch.rand(2, 3, 32, 32)
                train_model(cfg, target, [1, 2], folder, torch.device('cpu'))
                model, codes, _ = load_model(folder / 'fit/checkpoints/last.pt', torch.device('cpu'))
                saved = torch.load(folder / 'final/train.pt', weights_only=False)
                torch.testing.assert_close(render_direct(model, codes, 2), saved['reconstructions'], atol=0, rtol=0)
        for name in ('attention_chain_q7', 'spatial_star_q7'):
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
