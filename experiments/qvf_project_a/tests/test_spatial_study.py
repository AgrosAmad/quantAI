"""Test the spatial study's training and checkpoint boundary, not its score."""

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.qvf_project_a.direct_qvf import render_direct, train_direct
from experiments.qvf_project_a.models import CoordinateDecoder, ModelConfig
from experiments.qvf_project_a.spatial_study import (SpatialConfig, build_decoder,
    load_spatial, spatial_configs)
from experiments.qvf_project_a.train import configure_runtime


class SpatialStudyTests(unittest.TestCase):
    def setUp(self):
        configure_runtime("cpu", 91)

    def test_matched_control_is_original_field_and_preserves_parameter_budget(self):
        configs = dict(spatial_configs(0))
        spatial = build_decoder(configs["conv50k"])
        matched = build_decoder(configs["matched50k"])
        self.assertIsInstance(matched, CoordinateDecoder)
        self.assertEqual(sum(p.numel() for p in spatial.parameters()), 299894)
        self.assertEqual(sum(p.numel() for p in matched.parameters()), 299705)
        matched.circuit.load_state_dict(spatial.circuit.state_dict())
        for a, b in zip(spatial.circuit.parameters(), matched.circuit.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            build_decoder(SpatialConfig(model=ModelConfig(mode="ae", field_layers=3)))

    def test_spatial_training_reloads_codes_images_and_updates_both_components(self):
        cfg = SpatialConfig(n_train=5, n_val=2, batch_size=2, pixels_per_image=19,
                            steps=3, eval_every=2, seed=91)
        images, ids = torch.rand(5, 3, 32, 32), [17, 3, 28, 91, 8]
        with tempfile.TemporaryDirectory() as directory:
            result = train_direct(cfg, images, ids, directory, torch.device("cpu"),
                decoder_factory=lambda _: build_decoder(cfg), checkpoint_kind="spatial_qvf")
            decoder, codes, checkpoint = load_spatial(Path(directory) / "fit/checkpoints/last.pt", "cpu")
            initial = torch.load(Path(directory) / "fit/checkpoints/step0.pt", weights_only=False)
            saved = torch.load(Path(directory) / "final/train.pt", weights_only=False)
            torch.testing.assert_close(render_direct(decoder, codes, 2), saved["reconstructions"], atol=0, rtol=0)
            self.assertEqual(checkpoint["dataset_indices"], ids)
            self.assertEqual(result["training_coordinate_queries"], 3 * 2 * 19)
            self.assertEqual(result["parameter_counts"]["circuit"], 150)
            self.assertEqual(result["parameter_counts"]["encoder"], 0)
            for key in ("generator.0.weight", "generator.4.weight", "field.quantum.weights"):
                self.assertGreater((checkpoint["decoder_state"][key] - initial["decoder_state"][key]).abs().sum().item(), 0)
            self.assertGreater((codes.detach() - initial["codes"]).abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
