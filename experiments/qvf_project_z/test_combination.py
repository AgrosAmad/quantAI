"""Independent attention math, branch isolation, fusion and quantum interfaces."""
import unittest
import torch

from experiments.qvf_repro.qvf_fast import positional_embedding
from .combination_models import NAMES, ALIASES, build_model, WINDOW_RADIUS, resources, trace, readout
from .coverage_audit import native_check

CONFIG = dict(image_size=8, grid_sizes=(2, 4, 8), channels=4, frequencies=3, width=128)


class CombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_window_attention_against_dense_masked_reference_and_gradients(self):
        model = build_model(NAMES[0], 4, **CONFIG).double()
        xy = torch.tensor([[0., 0.], [.37, .61], [1., 1.]], dtype=torch.float64, requires_grad=True)
        for scale, size in enumerate(CONFIG['grid_sizes']):
            tokens = torch.randn(2, size * size, 4, dtype=torch.float64, requires_grad=True)
            actual = model.scale_features(tokens, xy, scale)
            positions = torch.stack(torch.meshgrid(torch.linspace(0, 1, size, dtype=torch.float64),
                                   torch.linspace(0, 1, size, dtype=torch.float64), indexing='ij'), -1).reshape(-1, 2)
            locations = (xy[:, None] + model.offsets[None] / (size - 1)).clamp(0, 1).flatten(0, 1)
            queries = model.queries[scale](positional_embedding(locations, model.cfg.frequencies))
            keys = model.keys[scale](torch.cat((tokens, positions[None].expand(2, -1, -1)), -1))
            scores = torch.einsum('qd,bnd->bqn', queries, keys) / 4.
            distance = ((locations[:, None] - positions[None]) * (size - 1)).square().sum(-1)
            centers = (locations * (size - 1)).floor().long()
            grid_indices = (positions * (size - 1)).round().long()
            valid = (grid_indices[None] - centers[:, None]).abs().le(WINDOW_RADIUS).all(-1)
            weights = (scores - 2 * distance).masked_fill(~valid, -torch.inf).softmax(-1)
            expected = (weights @ model.values[scale](tokens)).reshape_as(actual)
            self.assertLess(float((actual - expected).abs().max()), 1e-12)
            parameters = [tokens, xy, *model.queries[scale].parameters(), *model.keys[scale].parameters(),
                          *model.values[scale].parameters()]
            ga = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
            gb = torch.autograd.grad(expected.square().sum(), parameters)
            self.assertLess(max(float((a - b).abs().max()) for a, b in zip(ga, gb)), 1e-10)

    def test_scale_isolation_and_matched_fusion_initialization(self):
        mean, fusion, coherent = [build_model(n, 3, **CONFIG).double() for n in NAMES]
        codes = torch.randn(2, mean.cfg.latent_dim, dtype=torch.float64) * .2
        xy = torch.rand(4, 2, dtype=torch.float64)
        initial = mean.encoded(codes, xy)
        changed = codes.clone(); changed[:, :16] += .4
        altered = mean.encoded(changed, xy)
        self.assertGreater(float((initial[0] - altered[0]).abs().max()), 1e-8)
        for a, b in zip(initial[1:], altered[1:]): self.assertTrue(torch.equal(a, b))
        for other in (fusion.base, coherent.base):
            for key, value in mean.state_dict().items(): self.assertTrue(torch.equal(value, other.state_dict()[key]), key)
        with torch.no_grad():
            fusion.content.zero_(); fusion.position.zero_(); fusion.bias.zero_()
            coherent.routing_angles.zero_(); coherent.phase_angles.zero_()
        expected = mean(codes, xy)
        for model in (fusion, coherent):
            self.assertLess(float((model(codes, xy) - expected).abs().max()), 2e-12)

    def test_differentiability_cache_readout_and_native_quantum(self):
        for name in NAMES:
            model = build_model(name, 2, **CONFIG).double()
            codes = (torch.randn(2, model.cfg.latent_dim, dtype=torch.float64) * .3).requires_grad_()
            xy = torch.rand(3, 2, dtype=torch.float64)
            inputs, probabilities, output = trace(model, name, codes, xy)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(codes.grad).all(), name)
            for key, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (name, key))
                self.assertTrue(torch.isfinite(parameter.grad).all(), (name, key))
            base = model if name == NAMES[0] else model.base
            self.assertGreater(float(base.queries[0].weight.grad.abs().max()), 1e-12)
            independent = readout(model, name, [p.detach().numpy() for p in probabilities], xy.numpy(), 2)
            self.assertLess(float(abs(independent - output.detach().flatten(0, 1).numpy()).max()), 1e-12)
            model.cache_quantum()
            self.assertLess(float((model(codes, xy) - output).abs().max()), 1e-12)
            model.clear_cache()
            native_check(model, ALIASES[name], [x.detach().numpy() for x in inputs])
            info = resources(model, name)
            self.assertEqual(info['logical_circuits_per_coordinate'], 1 if name == NAMES[2] else 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
