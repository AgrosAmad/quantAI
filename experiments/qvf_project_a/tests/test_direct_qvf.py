"""Check direct-code fitting against explicit vanilla-QVF calculations."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.qvf_project_a.direct_qvf import (DirectConfig, direct_objective,
    evaluation_steps, load_direct, reconstruction_metrics, render_direct, train_direct)
from experiments.qvf_project_a.models import CoordinateDecoder, ModelConfig, pixel_coordinates
from experiments.qvf_project_a.train import configure_runtime
from experiments.qvf_repro.qvf_fast import positional_embedding


class DirectQVFTests(unittest.TestCase):
    def setUp(self):
        configure_runtime("cpu", 71)
        self.cfg = ModelConfig(mode="ae", field_width=16, field_layers=2, qubits=3, circuit_depth=1)

    def test_indexed_objective_and_updates_match_explicit_original_field(self):
        decoder = CoordinateDecoder(self.cfg).double()
        original = copy.deepcopy(decoder.field)
        codes = torch.nn.Parameter(torch.randn(7, 64, dtype=torch.float64) * .01)
        other_codes = torch.nn.Parameter(codes.detach().clone())
        rows = torch.tensor([5, 1, 6])
        coordinates = torch.rand(11, 2, dtype=torch.float64)
        target = torch.rand(3, 11, 3, dtype=torch.float64)
        positions = positional_embedding(coordinates, 6)
        inputs = torch.cat((other_codes[rows, None].expand(-1, 11, -1),
                            positions[None].expand(3, -1, -1)), -1).flatten(0, 1)
        expected = F.mse_loss(original(inputs).reshape(3, 11, 3), target)
        expected = expected + 1e-4 * inputs[:, :64].norm(dim=1).mean()
        actual, _, _ = direct_objective(decoder, codes, rows, coordinates, target, 1e-4)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(codes.grad, other_codes.grad, atol=1e-13, rtol=1e-12)
        self.assertEqual(codes.grad[[0, 2, 3, 4]].abs().sum().item(), 0)
        for a, b in zip(decoder.field.parameters(), original.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-13, rtol=1e-12)
        for parameters in [decoder.parameters(), original.parameters(), [codes], [other_codes]]:
            torch.optim.Adam(parameters, lr=.001).step()
        torch.testing.assert_close(codes, other_codes, atol=1e-13, rtol=1e-12)
        for a, b in zip(decoder.field.parameters(), original.parameters()):
            torch.testing.assert_close(a, b, atol=1e-13, rtol=1e-12)

    def test_render_is_independent_of_batch_and_coordinate_chunking(self):
        decoder = CoordinateDecoder(self.cfg)
        codes = torch.randn(5, 64) * .01
        whole = render_direct(decoder, codes, 5, 1024)
        chunked = render_direct(decoder, codes, 2, 73)
        torch.testing.assert_close(whole, chunked, atol=2e-6, rtol=1e-6)
        self.assertEqual(whole.shape, (5, 3, 32, 32))

    def test_small_training_budget_ids_and_checkpoint_round_trip(self):
        cfg = DirectConfig(model=self.cfg, n_train=5, n_val=2, batch_size=2,
                           pixels_per_image=19, steps=5, eval_every=2, seed=71)
        images = torch.rand(5, 3, 32, 32)
        ids = [31, 19, 4, 88, 12]
        with tempfile.TemporaryDirectory() as directory:
            result = train_direct(cfg, images, ids, directory, torch.device("cpu"))
            self.assertEqual(result["optimizer_steps"], 5)
            self.assertEqual(result["training_image_observations"], 10)
            self.assertEqual(result["training_coordinate_queries"], 190)
            self.assertEqual(sum(result["visits_per_image"]), 10)
            self.assertEqual(result["parameter_counts"]["encoder"], 0)
            decoder, codes, checkpoint = load_direct(Path(directory) / "fit/checkpoints/last.pt", torch.device("cpu"))
            self.assertEqual(checkpoint["dataset_indices"], ids)
            self.assertEqual(checkpoint["step"], 5)
            saved = torch.load(Path(directory) / "final/train.pt", weights_only=False)
            rebuilt = render_direct(decoder, codes, cfg.batch_size)
            torch.testing.assert_close(rebuilt, saved["reconstructions"], atol=0, rtol=0)
            self.assertEqual(reconstruction_metrics(rebuilt, images), result["final_fit"])
            self.assertEqual(evaluation_steps(cfg), [0, 1, 2, 4, 5])
            self.assertTrue(torch.isfinite(codes).all())


if __name__ == "__main__":
    unittest.main()
