"""Code-size control: original field equivalence, nested64 model and native gates."""
import unittest
import numpy as np
import torch

from .code_budget_models import CodeBudgetConfig, CodeBudgetField, build_model, resources
from .coverage_models import build_model as original_model, vanilla_native, vanilla_primitives
from .hires_gates import density, readout_numpy
from .hires_models import probabilities


class CodeBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_original_field_outputs_gradients_and_cache(self):
        for frequencies in (6, 9):
            model = build_model('vanilla_qvf256', 1, image_size=32,
                                frequencies=frequencies, width=128).double()
            codes = (torch.randn(2, 256, dtype=torch.float64) * .01).requires_grad_()
            xy = torch.rand(7, 2, dtype=torch.float64, requires_grad=True)
            actual = model(codes, xy)
            reference = model.field(model.inputs(codes, xy), per_sample=True).reshape(2, 7, 3)
            self.assertLess(float((actual - reference).abs().max()), 1e-12)
            parameters = [codes, xy, *model.parameters()]
            aa = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
            bb = torch.autograd.grad(reference.square().sum(), parameters)
            self.assertLess(max(float((a-b).abs().max()) for a, b in zip(aa, bb)), 1e-11)
            self.assertTrue(all(torch.isfinite(g).all() for g in aa))
            self.assertGreater(float(aa[0][:, 64:].abs().max()), 0.)
            model.cache_quantum()
            self.assertLess(float((model(codes, xy) - actual).abs().max()), 1e-12)
            _, probs, rgb = model.trace(codes, xy)
            decoded = readout_numpy(model, [p.detach().numpy() for p in probs])
            self.assertLess(float(np.max(np.abs(decoded - rgb.detach().flatten(0, 1).numpy()))), 1e-12)
            with self.assertRaises(ValueError):
                model(torch.zeros(1, 64, dtype=torch.float64), xy)

    def test_64_compatibility_and_exact_embedding_in_256(self):
        # The generic wrapper at64 must reproduce the untouched Z8 constructor.
        original = original_model('vanilla_qvf', 2, image_size=32, frequencies=6, width=128).double()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2)
            generic = CodeBudgetField(CodeBudgetConfig(latent_dim=64, image_size=32, frequencies=6)).double()
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, generic.state_dict()[name]), name)
        # A known64 solution must fit inside256 without any circuit/readout change.
        wide = build_model('vanilla_qvf256', 2, image_size=32, frequencies=6, width=128).double()
        state = wide.state_dict()
        for name, value in original.state_dict().items():
            if name == 'field.embedder.layers.0.weight':
                state[name].zero_()
                state[name][:, :64].copy_(value[:, :64])
                state[name][:, 256:].copy_(value[:, 64:])
            else:
                state[name].copy_(value)
        wide.load_state_dict(state)
        codes = torch.randn(2, 64, dtype=torch.float64) * .01
        padded = torch.cat((codes, torch.zeros(2, 192, dtype=torch.float64)), dim=1)
        xy = torch.rand(7, 2, dtype=torch.float64)
        self.assertLess(float((wide(padded, xy) - original(codes, xy)).abs().max()), 1e-12)
        info = resources(wide, 'vanilla_qvf256')
        self.assertEqual(info['image_code_entries'], 256)
        self.assertEqual(info['stored_shared_parameters'] - sum(p.numel() for p in original.parameters()), 24576)
        self.assertEqual(info['extra_first_layer_weights_vs_64'], 24576)

    def test_original_native_circuit_and_compiled_preparation(self):
        model = build_model('vanilla_qvf256', 0, image_size=32, frequencies=6, width=128).double()
        x = torch.randn(32, dtype=torch.float64).softmax(-1).sqrt().requires_grad_()
        normalized = x / x.norm()
        fast = normalized @ model.field.quantum.circuit_matrix()
        native = vanilla_native(model, normalized)
        self.assertLess(float((fast - native).abs().max()), 1e-12)
        obs = torch.arange(32, dtype=torch.float64).sin()
        parameters = [x, *model.field.quantum.parameters()]
        aa = torch.autograd.grad((fast.square() * obs).sum(), parameters, retain_graph=True)
        bb = torch.autograd.grad((probabilities(native) * obs).sum(), parameters)
        self.assertLess(max(float((a-b).abs().max()) for a, b in zip(aa, bb)), 1e-11)
        compiled = density(vanilla_primitives(model, normalized.detach().numpy()[None]), 1, 5)
        self.assertLess(float(np.max(np.abs(compiled - fast.detach().square().numpy()[None]))), 1e-12)
        info = resources(model, 'vanilla_qvf256')
        self.assertEqual(info['stored_quantum_parameters'], 150)
        self.assertEqual(info['output_relevant_quantum_parameters'], 125)
        self.assertEqual(info['primitive_gates_per_coordinate'], {'RY': 256, 'CNOT': 230})


if __name__ == '__main__':
    unittest.main(verbosity=2)
