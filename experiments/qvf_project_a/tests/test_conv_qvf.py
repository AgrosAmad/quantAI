"""Numerical checks for the untrained spatial-feature decoder proposal."""

import unittest

import torch
from torch.nn import functional as F

from experiments.qvf_project_a.conv_qvf import LocalFeatureQVFDecoder, local_features


class ConvolutionalQVFTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(91)

    def test_local_sampling_matches_grid_sample_outputs_and_feature_gradients(self):
        features = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
        other = features.detach().clone().requires_grad_(True)
        coordinates = torch.cat((torch.rand(23, 2, dtype=torch.float64),
                                 torch.tensor([[0., 0.], [1., 1.], [-.2, 1.2]], dtype=torch.float64)))
        grid = (2 * coordinates.flip(-1) - 1)[None, None].expand(2, 1, -1, -1)
        expected = F.grid_sample(other, grid, mode="bilinear", padding_mode="border", align_corners=True)[:, :, 0].transpose(1, 2)
        actual = local_features(features, coordinates)
        torch.testing.assert_close(actual, expected, atol=3e-15, rtol=3e-15)
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        torch.testing.assert_close(features.grad, other.grad, atol=3e-15, rtol=3e-15)

    def test_quantum_head_and_feature_generator_receive_reconstruction_gradients(self):
        model = LocalFeatureQVFDecoder()
        code = torch.randn(2, 64, requires_grad=True)
        coordinates = torch.rand(17, 2)
        image = model(code, coordinates)
        self.assertEqual(tuple(image.shape), (2, 17, 3))
        F.mse_loss(image, torch.rand_like(image)).backward()
        for parameter in (code, model.generator[0].weight, model.generator[4].weight,
                          model.field.embedder.layers[0].weight, model.field.quantum.weights,
                          model.field.quantum.cond):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual(model.field.quantum.n_qubits, 5)
        self.assertEqual(model.field.quantum.repetitions, 5)
        self.assertTrue(model.field.use_parity)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 299894)

    def test_feature_reuse_and_chunked_queries_preserve_the_same_image_field(self):
        model = LocalFeatureQVFDecoder().eval()
        code, coordinates = torch.randn(2, 64), torch.rand(65, 2)
        with torch.no_grad():
            features = model.prepare_features(code)
            full = model(code, coordinates)
            chunked = torch.cat([model.query_features(features, chunk) for chunk in coordinates.split(11)], 1)
        self.assertEqual(tuple(features.shape), (2, 32, 16, 16))
        torch.testing.assert_close(full, chunked, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
