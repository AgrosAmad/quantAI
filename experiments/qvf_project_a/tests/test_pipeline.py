"""Numerical and integration checks for the scientific changes in Project A."""

import tempfile
import unittest
from unittest import mock
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.qvf_project_a.data import posterior_normalization, split_indices
from experiments.qvf_project_a.diffusion import LatentDDPM, LatentDenoiser
from experiments.qvf_project_a.models import ModelConfig, QVFAutoencoder, kl_per_image, sample_posterior
from experiments.qvf_project_a.train import (RunConfig, cache_latents, configure_runtime, generate,
                                            load_vae, save_checkpoint, sha256, train_diffusion, train_vae)
from experiments.qvf_repro.qvf_fast import QVFImage, positional_embedding


class ProjectATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.cfg = ModelConfig(latent_dim=4, encoder_width=4, field_width=16,
                               field_layers=2, qubits=3, circuit_depth=1, frequencies=2,
                               readout="legacy_affine")

    def test_decoder_matches_reproduction_with_declared_rgb_mapping(self):
        model = QVFAutoencoder(self.cfg)
        reference = QVFImage({"L_embed": 2, "latent_size": 4, "hid_dim": 16, "out_dim": 8,
                              "num_hidden_layers_energy": 2, "skip_connection": False,
                              "num_repetations": 1, "use_identity_block": False,
                              "gaussian_std": 0.1, "output_channels": 3, "use_parity": False})
        reference.embedder.load_state_dict(model.decoder.network.state_dict())
        reference.quantum.load_state_dict(model.decoder.quantum.state_dict())
        code = torch.randn(2, 4, requires_grad=True)
        coords = model.coordinates[:7]
        inputs = torch.cat((code[:, None].expand(-1, 7, -1),
                            positional_embedding(coords, 2)[None].expand(2, -1, -1)), dim=-1).reshape(14, -1)
        expected = 0.5 * (reference(inputs).reshape(2, 7, 3) + 1)
        actual = model.decoder(code, coords)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        expected_grad = torch.autograd.grad(expected.square().sum(), code)[0]
        actual_grad = torch.autograd.grad(actual.square().sum(), code)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-6)

    def test_reconstruction_gradients_reach_both_encoder_heads_and_circuit(self):
        model = QVFAutoencoder(self.cfg)
        images = torch.rand(3, 3, 32, 32)
        indices = torch.arange(0, 1024, 64)
        predictions, _, _ = model(images, indices)
        F.mse_loss(predictions, images.flatten(2).transpose(1, 2)[:, indices]).backward()
        for parameter in [model.encoder.mean.weight, model.encoder.log_variance.weight,
                          model.decoder.network.layers[0].weight, model.decoder.quantum.weights,
                          model.decoder.quantum.cond]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_gaussian_kl_against_torch_distributions(self):
        mean, logvar = torch.randn(5, 4), torch.randn(5, 4)
        posterior = torch.distributions.Normal(mean, (0.5 * logvar).exp())
        prior = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
        expected = torch.distributions.kl_divergence(posterior, prior).sum(-1)
        torch.testing.assert_close(kl_per_image(mean, logvar), expected)
        torch.testing.assert_close(kl_per_image(torch.zeros(5, 4), torch.zeros(5, 4)), torch.zeros(5))

    def test_disjoint_reproducible_split_and_oversized_request(self):
        targets = [5] * 20 + [0] * 10
        split = split_indices(targets, 5, 12, 5, 7)
        other = split_indices(targets, 5, 12, 5, 7)
        self.assertFalse(set(split["train"].tolist()) & set(split["validation"].tolist()))
        for name in split:
            torch.testing.assert_close(split[name], other[name])
            self.assertTrue(all(targets[index] == 5 for index in split[name]))
        with self.assertRaises(ValueError):
            split_indices(targets, 5, 18, 5, 7)

    def test_posterior_normalization_includes_sampling_variance(self):
        train = {"mean": torch.tensor([[1., 4.], [3., 4.]]), "logvar": torch.zeros(2, 2)}
        normalizer = posterior_normalization(train)
        torch.testing.assert_close(normalizer["mean"], torch.tensor([2., 4.]))
        torch.testing.assert_close(normalizer["std"], torch.tensor([2., 1.]).sqrt())
        # Verify the formula against independent Monte Carlo draws.
        repeated_means = train["mean"].repeat(20000, 1)
        samples = sample_posterior(repeated_means, torch.zeros_like(repeated_means))
        normalized = (samples - normalizer["mean"]) / normalizer["std"]
        torch.testing.assert_close(normalized.mean(0), torch.zeros(2), atol=0.025, rtol=0)
        torch.testing.assert_close(normalized.var(0), torch.ones(2), atol=0.025, rtol=0)

    def test_velocity_oracle_recovers_clean_at_all_timesteps(self):
        diffusion = LatentDDPM(100)
        timesteps = torch.arange(100)
        clean, noise = torch.randn(100, 4), torch.randn(100, 4)
        noisy = diffusion.q_sample(clean, timesteps, noise)
        velocity = diffusion.target(clean, timesteps, noise)
        torch.testing.assert_close(diffusion.clean_from_velocity(noisy, timesteps, velocity), clean)
        # At the final reverse step no additional noise is injected.
        torch.testing.assert_close(diffusion.reverse_step(noisy[:1], timesteps[:1], velocity[:1],
                                                         torch.full((1, 4), 1000.)), clean[:1])
        self.assertLess(diffusion.signal[-1].item(), 0.001)
        self.assertTrue((diffusion.signal[1:] < diffusion.signal[:-1]).all())

    def test_ddpm_posterior_matches_independent_gaussian_conditioning(self):
        diffusion = LatentDDPM(20)
        for step in (1, 8, 19):
            beta = diffusion.betas[step].double()
            alpha = 1 - beta
            previous = torch.prod(1 - diffusion.betas[:step].double())
            # Conditional Gaussian: Var(x_previous | x_current), computed by
            # covariance conditioning instead of the sampler's coefficient formula.
            var_previous = 1 - previous
            var_current = alpha * var_previous + beta
            covariance = alpha.sqrt() * var_previous
            conditional_variance = var_previous - covariance.square() / var_current
            torch.testing.assert_close(diffusion.posterior_variance[step].double(), conditional_variance, atol=2e-7, rtol=2e-6)

    def test_deterministic_and_decoder_controls(self):
        for decoder in ("qvf", "no_circuit", "mlp"):
            cfg = ModelConfig(**{**asdict(self.cfg), "mode": "ae", "decoder": decoder})
            model = QVFAutoencoder(cfg).eval()
            images = torch.rand(2, 3, 32, 32)
            first, mean, logvar = model(images, torch.arange(8))
            second, _, _ = model(images, torch.arange(8))
            self.assertIsNone(logvar)
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            self.assertTrue(((first >= -1e-6) & (first <= 1 + 1e-6)).all())

    def test_saved_model_rendering_and_pixel_chunking(self):
        model = QVFAutoencoder(self.cfg).eval()
        code = torch.randn(2, 4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vae.pt"
            save_checkpoint(path, {"format_version": 1, "kind": "qvf_vae", "model_config": asdict(self.cfg),
                                   "model_state": model.state_dict()})
            restored, _ = load_vae(path, torch.device("cpu"))
            torch.testing.assert_close(restored.render(code, pixel_chunk=127), model.render(code), atol=2e-6, rtol=2e-6)

    def test_cpu_pipeline_freezes_vae_and_checkpoint_sampling_is_repeatable(self):
        self.cfg.readout = "vanilla"
        cfg = RunConfig(model=self.cfg, n_train=8, n_val=4, batch_size=4, pixels_per_image=16,
                        vae_epochs=1, vae_eval_every=1, diffusion_steps=3, diffusion_batch_size=4,
                        diffusion_width=16, diffusion_layers=1, diffusion_timesteps=4, diffusion_eval_every=3)
        images = {"train": torch.rand(8, 3, 32, 32), "validation": torch.rand(4, 3, 32, 32)}
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vae_path, _ = train_vae(cfg, images, root, device)
            before_hash = sha256(vae_path)
            cache = cache_latents(cfg, images, vae_path, root, device)
            expected = posterior_normalization(cache["posteriors"]["train"])
            torch.testing.assert_close(cache["normalization"]["mean"], expected["mean"])
            with mock.patch("experiments.qvf_project_a.models.CoordinateDecoder.forward",
                            side_effect=AssertionError("Diffusion training must not call QVF")), \
                 mock.patch("experiments.qvf_project_a.models.ImageEncoder.forward",
                            side_effect=AssertionError("Diffusion training must use cached posteriors")):
                path, metrics = train_diffusion(cfg, cache, vae_path, root, device)
            self.assertEqual(sha256(vae_path), before_hash)
            self.assertEqual(metrics["qvf_calls_during_diffusion_training"], 0)
            a = generate(path, root / "sample_a", device, count=2, seed=99)
            b = generate(path, root / "sample_b", device, count=2, seed=99)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(a).all())
            self.assertEqual(a.shape, (2, 3, 32, 32))
            with self.assertRaises(FileExistsError):
                generate(path, root / "sample_a", device, count=2, seed=99)
            cache["vae_sha256"] = "wrong"
            with self.assertRaises(ValueError):
                train_diffusion(cfg, cache, vae_path, root, device)

    def test_guard_against_other_gpu(self):
        with self.assertRaises(ValueError):
            configure_runtime("cuda:1", 0)


if __name__ == "__main__":
    unittest.main()
