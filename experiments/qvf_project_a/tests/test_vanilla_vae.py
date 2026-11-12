"""Guard the scientific contract: the VAE must not silently change vanilla QVF."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.qvf_project_a.models import ModelConfig, QVFAutoencoder
from experiments.qvf_project_a.train import load_vae, save_checkpoint
from experiments.qvf_repro.qvf_fast import QVFImage, positional_embedding


class VanillaVAETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(81)

    def test_five_qubit_adapter_matches_vanilla_outputs_and_all_gradients(self):
        cfg = ModelConfig(latent_dim=4, encoder_width=4, field_width=16,
                          field_layers=2, frequencies=2, qubits=5, circuit_depth=5)
        model = QVFAutoencoder(cfg).double()
        reference = QVFImage({"L_embed": 2, "latent_size": 4, "hid_dim": 16, "out_dim": 32,
            "num_hidden_layers_energy": 2, "skip_connection": False, "num_repetations": 5,
            "use_identity_block": False, "gaussian_std": 0.1, "output_channels": 3,
            "use_parity": True}).double()
        self.assertIsInstance(model.decoder.field, QVFImage)
        reference.load_state_dict(model.decoder.field.state_dict())
        code = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        other_code = code.detach().clone().requires_grad_(True)
        coords = model.coordinates[torch.tensor([0, 17, 213, 300, 782, 1023])]
        inputs = torch.cat((other_code[:, None].expand(-1, 6, -1),
            positional_embedding(coords, 2)[None].expand(3, -1, -1)), dim=-1).reshape(18, -1)
        expected = reference(inputs).reshape(3, 6, 3)
        actual = model.decoder(code, coords)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        probes = torch.randn_like(actual)
        (actual * probes).sum().backward()
        (expected * probes).sum().backward()
        torch.testing.assert_close(code.grad, other_code.grad, atol=0, rtol=0)
        for ours, theirs in zip(model.decoder.field.parameters(), reference.parameters()):
            torch.testing.assert_close(ours.grad, theirs.grad, atol=0, rtol=0)
        # Explicit grouping guards against accidentally testing first-three
        # expectations on a three-qubit circuit, where the maps coincide.
        with torch.no_grad():
            z = reference.quantum(reference.embedder(inputs))
            grouped = torch.stack((z[:, 0] + z[:, 3], z[:, 1] + z[:, 4], z[:, 2]), -1)
            torch.testing.assert_close(actual.reshape(-1, 3), grouped, atol=0, rtol=0)

    def test_vanilla_reconstruction_trains_both_encoder_heads_and_circuit(self):
        model = QVFAutoencoder(ModelConfig(latent_dim=4, encoder_width=4, field_width=16,
                                          field_layers=2, frequencies=2, circuit_depth=2))
        images = torch.rand(3, 3, 32, 32)
        pixels = torch.arange(0, 1024, 64)
        predictions, _, _ = model(images, pixels)
        F.mse_loss(predictions, images.flatten(2).transpose(1, 2)[:, pixels]).backward()
        for parameter in (model.encoder.mean.weight, model.encoder.log_variance.weight,
                          model.decoder.field.embedder.layers[0].weight,
                          model.decoder.field.quantum.weights, model.decoder.field.quantum.cond):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_pre_versioned_checkpoint_keeps_legacy_rgb_and_new_checkpoint_keeps_vanilla(self):
        for readout in ("legacy_affine", "vanilla"):
            cfg = ModelConfig(latent_dim=4, encoder_width=4, field_width=16,
                              field_layers=2, frequencies=2, circuit_depth=1, readout=readout)
            model = QVFAutoencoder(cfg).eval()
            code = torch.randn(2, 4)
            values = asdict(cfg)
            if readout == "legacy_affine":
                del values["readout"]
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "checkpoint.pt"
                save_checkpoint(path, {"kind": "qvf_vae", "format_version": 1,
                    "model_config": values, "model_state": model.state_dict()})
                loaded, _ = load_vae(path, torch.device("cpu"))
                self.assertEqual(loaded.cfg.readout, readout)
                torch.testing.assert_close(loaded.render(code), model.render(code), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
