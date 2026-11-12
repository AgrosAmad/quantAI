"""Spatial indexing, original-QVF equivalence and matched-training contracts."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.qvf_repro.qvf_fast import QVFImage
from experiments.qvf_project_a.rae_qvf import ReadoutConfig, RAEFeatureReadout, pixel_layout, render_readout
from experiments.qvf_project_a.rae_readout_study import BatchStream, TrainConfig, learning_rate, load_head, train_readout


class RAEReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(18)
        self.cfg = ReadoutConfig(input_channels=6, adapter_channels=8, grid_size=2,
                                 image_size=4, field_width=12, field_layers=2)

    def test_pixel_patch_and_local_offset_mapping(self):
        patches, positions = pixel_layout(4, 2, 6)
        self.assertEqual(patches.tolist(), [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3])
        torch.testing.assert_close(positions[[0, 1, 4, 5], :2], torch.tensor([
            [-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]]), rtol=0, atol=0)
        model = RAEFeatureReadout(self.cfg)
        adapted = torch.arange(4.).reshape(1, 1, 2, 2).expand(1, 8, 2, 2)
        actual = model.query_inputs(adapted, torch.arange(16))
        torch.testing.assert_close(actual[0, :, 0], patches.float(), rtol=0, atol=0)

    def test_exact_original_quantum_readout_and_gradient_path(self):
        model = RAEFeatureReadout(self.cfg).double()
        reference = QVFImage(model.field_config()).double()
        reference.embedder.load_state_dict(model.embedder.state_dict())
        reference.quantum.load_state_dict(model.quantum.state_dict())
        features = torch.randn(2, 6, 2, 2, dtype=torch.float64)
        points = torch.tensor([0, 3, 7, 9, 15])
        adapted = model.prepare_features(features)
        inputs = model.query_inputs(adapted, points)
        actual = model(features, points)
        expected = reference(inputs.flatten(0, 1)).reshape(2, 5, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
        F.mse_loss(actual, torch.rand_like(actual)).backward()
        for parameter in (model.project.weight, model.spatial1.weight, model.spatial2.weight,
                          model.embedder.layers[0].weight, model.quantum.weights, model.quantum.cond):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual((model.quantum.n_qubits, model.quantum.repetitions), (5, 5))

    def test_identical_common_initialization_and_close_parameter_budgets(self):
        torch.manual_seed(91)
        quantum = RAEFeatureReadout()
        torch.manual_seed(91)
        classical = RAEFeatureReadout(ReadoutConfig(kind="classical"))
        for name, value in quantum.state_dict().items():
            if not name.startswith("quantum."):
                torch.testing.assert_close(value, classical.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in quantum.parameters()), 500150)
        self.assertEqual(sum(p.numel() for p in classical.parameters()), 500099)

    def test_rendering_chunk_and_image_batch_consistency(self):
        for kind in ("qvf", "classical"):
            model = RAEFeatureReadout(replace(self.cfg, kind=kind)).eval()
            features = torch.randn(3, 6, 2, 2)
            with torch.no_grad():
                full = model(features, torch.arange(16))
                adapted = model.prepare_features(features)
                chunks = torch.cat([model.query(adapted, points) for points in torch.arange(16).split(5)], 1)
                images = render_readout(model, features, batch_size=1)
            torch.testing.assert_close(full, chunks, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(full.transpose(1, 2).reshape(3, 3, 4, 4), images, rtol=1e-5, atol=1e-6)

    def test_sampler_resume_and_schedule(self):
        cfg = TrainConfig(steps=10, batch_size=3, pixels_per_image=5, warmup_steps=2)
        a = BatchStream(8, 4, cfg)
        for _ in range(3):
            a.next()
        b = BatchStream(8, 4, cfg)
        b.load_state_dict(a.state_dict())
        for _ in range(5):
            for x, y in zip(a.next(), b.next()):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertEqual(a.digest, b.digest)
        self.assertAlmostEqual(learning_rate(1, cfg), cfg.learning_rate / 2)
        self.assertAlmostEqual(learning_rate(2, cfg), cfg.learning_rate)
        self.assertAlmostEqual(learning_rate(10, cfg), cfg.final_learning_rate)

    def test_training_budget_artifact_and_reload_for_both_heads(self):
        features = {"train": torch.randn(6, 6, 2, 2), "validation": torch.randn(3, 6, 2, 2)}
        images = {"train": torch.rand(6, 3, 4, 4), "validation": torch.rand(3, 3, 4, 4)}
        ids = {"train": [12, 9, 4, 70, 2, 19], "validation": [8, 1, 44]}
        cfg = TrainConfig(steps=3, batch_size=2, pixels_per_image=8, warmup_steps=0, eval_every=2)
        digests = []
        with tempfile.TemporaryDirectory() as folder:
            for kind in ("qvf", "classical"):
                path = Path(folder) / kind
                result = train_readout(cfg, replace(self.cfg, kind=kind), features, images, ids, path, "test-cache")
                self.assertEqual(result["optimizer_steps"], 3)
                self.assertEqual(result["training_image_observations"], 6)
                self.assertEqual(result["training_pixel_queries"], 48)
                model, checkpoint = load_head(path / "checkpoints/last.pt", torch.device("cpu"))
                self.assertEqual(checkpoint["dataset_indices"], ids)
                payload = torch.load(path / "final/validation.pt", weights_only=True)
                torch.testing.assert_close(render_readout(model, features["validation"], 2),
                                           payload["reconstructions"], rtol=0, atol=0)
                digests.append(result["batch_stream_digest"])
            self.assertEqual(digests[0], digests[1])

    def test_interrupted_resume_matches_uninterrupted_updates(self):
        features = {"train": torch.randn(6, 6, 2, 2), "validation": torch.randn(3, 6, 2, 2)}
        images = {"train": torch.rand(6, 3, 4, 4), "validation": torch.rand(3, 3, 4, 4)}
        ids = {"train": [12, 9, 4, 70, 2, 19], "validation": [8, 1, 44]}
        cfg = TrainConfig(steps=6, batch_size=2, pixels_per_image=8, warmup_steps=0, eval_every=3)
        with tempfile.TemporaryDirectory() as folder:
            full, resumed = Path(folder) / "full", Path(folder) / "resumed"
            uninterrupted = train_readout(cfg, self.cfg, features, images, ids, full, "test-cache")
            original_step = torch.optim.Adam.step
            calls = 0

            def interrupted_step(optimizer, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("simulated interruption")
                return original_step(optimizer, *args, **kwargs)

            with patch.object(torch.optim.Adam, "step", interrupted_step):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train_readout(cfg, self.cfg, features, images, ids, resumed, "test-cache")
            continued = train_readout(cfg, self.cfg, features, images, ids, resumed, "test-cache", resume=True)
            a, ca = load_head(full / "checkpoints/last.pt", torch.device("cpu"))
            b, cb = load_head(resumed / "checkpoints/last.pt", torch.device("cpu"))
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0)
            torch.testing.assert_close(ca["visits"], cb["visits"], rtol=0, atol=0)
            self.assertEqual(uninterrupted["batch_stream_digest"], continued["batch_stream_digest"])
            self.assertEqual(uninterrupted["stages"], continued["stages"])


if __name__ == "__main__":
    unittest.main()
