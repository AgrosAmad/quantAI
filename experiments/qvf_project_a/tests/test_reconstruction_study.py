import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from experiments.qvf_project_a.data import extend_split_indices
from experiments.qvf_project_a.models import ModelConfig
from experiments.qvf_project_a.reconstruction import study_configs
from experiments.qvf_project_a.refine import refine_codes
from experiments.qvf_project_a.train import RunConfig, train_vae


class ToyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encoder(self, images):
        return images.new_zeros((len(images), 3)), None

    def render(self, code):
        return (self.scale * code).sigmoid()[:, :, None, None].expand(-1, -1, 4, 4)


class ReconstructionStudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_more_data_preserves_validation_ids_and_training_prefix(self):
        targets = [5] * 20 + [0] * 5
        reference = {"train": [2, 7, 4, 0], "validation": [5, 8, 1]}
        expanded = extend_split_indices(targets, 5, reference, 17, 9)
        self.assertEqual(expanded["validation"].tolist(), reference["validation"])
        self.assertEqual(expanded["train"][:4].tolist(), reference["train"])
        self.assertEqual(set(expanded["train"].tolist()), set(range(20)) - set(reference["validation"]))
        same = extend_split_indices(targets, 5, reference, 17, 9)
        torch.testing.assert_close(same["train"], expanded["train"])
        with self.assertRaises(ValueError):
            extend_split_indices(targets, 5, reference, 18, 9)
        with self.assertRaises(ValueError):
            extend_split_indices(targets, 5, {"train": [1, 2], "validation": [2, 3]}, 4, 9)

    def test_study_matches_optimizer_and_kl_budgets_across_data_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(json.dumps(asdict(RunConfig())))
            variants = dict(study_configs(2, root))
        for name, cfg in variants.items():
            self.assertEqual(cfg.seed, 2)
            self.assertEqual((cfg.model.qubits, cfg.model.circuit_depth), (5, 5))
            self.assertEqual(cfg.kl_warmup_steps, 320)
            self.assertEqual(cfg.vae_eval_every_steps, 320)
            self.assertTrue(cfg.vae_drop_last)
        self.assertEqual(variants["baseline"].vae_max_steps, 2560)
        self.assertEqual(variants["more_data"].vae_max_steps, 2560)
        self.assertEqual(variants["longer"].vae_max_steps, 7680)
        self.assertEqual(variants["more_data_longer"].vae_max_steps, 7680)
        self.assertEqual(variants["more_data"].n_train, 4744)
        self.assertEqual(variants["latent128"].model.latent_dim, 128)
        self.assertEqual(variants["wider"].model.field_width, 256)

    def test_exact_step_count_and_warmup_independent_of_dataset_size(self):
        validation = torch.rand(4, 3, 32, 32)
        pool = torch.rand(20, 3, 32, 32)
        for count in (8, 20):
            torch.manual_seed(11)
            cfg = RunConfig(model=ModelConfig(latent_dim=4, encoder_width=4, field_width=8,
                                              field_layers=1, frequencies=1, qubits=3, circuit_depth=1),
                            n_train=count, n_val=4, batch_size=4, pixels_per_image=16,
                            vae_max_steps=5, vae_eval_every_steps=2, kl_warmup_steps=4, vae_drop_last=True)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _, metrics = train_vae(cfg, {"train": pool[:count], "validation": validation}, root, torch.device("cpu"))
                history = [json.loads(line) for line in (root / "vae/history.jsonl").read_text().splitlines()]
                evaluations = [row for row in history if "validation" in row]
                self.assertEqual([row["step"] for row in evaluations], [1, 2, 4, 5])
                self.assertEqual([row["kl_weight"] / cfg.kl_weight for row in evaluations], [0.25, 0.5, 1, 1])
                self.assertEqual(metrics["optimizer_steps"], 5)
                self.assertEqual(metrics["training_image_observations"], 20)
                self.assertEqual(metrics["training_coordinate_queries"], 320)

    def test_refinement_changes_codes_and_keeps_network_frozen(self):
        model = ToyAutoencoder()
        colours = torch.tensor([[0.2, 0.6, 0.8], [0.7, 0.3, 0.1], [0.4, 0.9, 0.6]])
        images = colours[:, :, None, None].expand(-1, -1, 4, 4)
        result = refine_codes(model, images, batch_size=2, steps=100, lr=0.1, log_every=20)
        self.assertTrue(result["model_weights_unchanged"])
        self.assertEqual(model.scale.item(), 1.0)
        self.assertIsNone(model.scale.grad)
        self.assertTrue((result["refined_mse"] <= result["initial_mse"]).all())
        self.assertLess(result["refined_mse"].mean().item(), result["initial_mse"].mean().item() / 10)
        self.assertGreater(result["refined_codes"].norm().item(), 0)
        self.assertEqual([row["step"] for row in result["history"]], [0, 20, 40, 60, 80, 100])


if __name__ == "__main__":
    unittest.main()
