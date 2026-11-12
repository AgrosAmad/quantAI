"""V10 equivalence, unbiased edge sampling, masked scoring and training boundary."""

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.qvf_project_a.direct_qvf import render_direct
from experiments.qvf_project_a.edge_qvf import (build_edge_decoder, edge_objective,
    masked_gradient_metrics, neighbour_pairs, rgb_importance_weights, sample_pairs)
from experiments.qvf_project_a.edge_study import EdgeConfig, load_edge, train_edge
from experiments.qvf_project_a.models import pixel_coordinates
from experiments.qvf_project_a.token_study import fit_completion
from experiments.qvf_project_a.train import configure_runtime


class EdgeTests(unittest.TestCase):
    def setUp(self):
        configure_runtime("cpu", 37)

    def test_zero_uploads_recover_attention_outputs_and_shared_gradients(self):
        a = build_edge_decoder("attention", 3).double()
        b = build_edge_decoder("staged_all", 3).double()
        codes = torch.randn(2, 336, dtype=torch.float64, requires_grad=True) * .1
        points = pixel_coordinates(32, 32).double()[::79]
        pa, pb = a(codes, points), b(codes, points)
        torch.testing.assert_close(pa, pb, atol=1e-12, rtol=1e-12)
        shared = dict(b.named_parameters())
        ga = torch.autograd.grad(pa.square().sum(), (codes, *a.parameters()), retain_graph=True)
        gb = torch.autograd.grad(pb.square().sum(), (codes, *[shared[n] for n, _ in a.named_parameters()]), retain_graph=True)
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-11)
        extra = torch.autograd.grad(pb.square().sum(), tuple(b.uploads.parameters()))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in extra))
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in b.parameters()) - sum(p.numel() for p in a.parameters()), 50)

    def test_edge_sampling_has_no_wrap_and_pixel_weights_are_unbiased(self):
        size = 4
        h, v = neighbour_pairs(size)
        self.assertTrue(torch.equal(h[:, 0] // size, h[:, 1] // size))
        self.assertTrue(torch.equal(v[:, 0] % size, v[:, 1] % size))
        self.assertTrue(((h[:, 1] - h[:, 0]) == 1).all())
        self.assertTrue(((v[:, 1] - v[:, 0]) == size).all())
        image = torch.randn(2, 3, size, size, dtype=torch.float64, requires_grad=True)
        target = torch.randn_like(image)
        points = torch.cat((h, v)).flatten()
        predictions = image.flatten(2).transpose(1, 2)[:, points]
        truth = target.flatten(2).transpose(1, 2)[:, points]
        _, pixel, edge, _ = edge_objective(predictions, truth, rgb_importance_weights(size)[points],
                                           torch.zeros(2, 1), .5, 0)
        error = image - target
        full_pixel = error.square().mean()
        full_edge = .5 * (error.diff(dim=2).square().mean() + error.diff(dim=3).square().mean())
        torch.testing.assert_close(pixel, full_pixel, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(edge, full_edge, atol=1e-12, rtol=1e-12)
        g1 = torch.autograd.grad(pixel + .5 * edge, image, retain_graph=True)[0]
        g2 = torch.autograd.grad(full_pixel + .5 * full_edge, image)[0]
        torch.testing.assert_close(g1, g2, atol=1e-7, rtol=1e-7)
        one = sample_pairs(torch.Generator().manual_seed(9), 10, size).reshape(-1, 2)
        two = sample_pairs(torch.Generator().manual_seed(9), 10, size).reshape(-1, 2)
        self.assertTrue(torch.equal(one, two))
        self.assertTrue(((one[:10, 1] - one[:10, 0]) == 1).all())
        self.assertTrue(((one[10:, 1] - one[10:, 0]) == size).all())

    def test_edge_metric_uses_both_endpoints_and_signed_changes(self):
        target = torch.zeros(1, 3, 4, 4, dtype=torch.float64)
        prediction = target.clone()
        prediction[:, :, 0] = 100
        mask = list(range(4, 16))
        values = masked_gradient_metrics(prediction, target, mask)
        self.assertEqual(values["gradient_mse"], 0)
        self.assertEqual((values["horizontal_edges"], values["vertical_edges"]), (9, 8))
        # Equal-magnitude but opposite gradients must not be considered correct.
        ramp = torch.arange(4, dtype=torch.float64).repeat(1, 3, 4, 1)
        values = masked_gradient_metrics(-ramp, ramp, list(range(16)))
        self.assertEqual(values["gradient_mse"], 2)
        values = masked_gradient_metrics(ramp + 1, ramp, list(range(16)))
        self.assertEqual(values["gradient_mse"], 0)

    def test_training_checkpoint_and_completion_mask_isolation(self):
        cfg = EdgeConfig(architecture="staged_all", edge_weight=.5, seed=37,
            n_train=4, steps=3, batch_size=2, pairs_per_orientation=3, eval_every=1)
        images = torch.rand(4, 3, 32, 32)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = train_edge(cfg, images, [3, 7, 10, 11], directory, torch.device("cpu"))
            model, codes, saved = load_edge(directory / "fit/checkpoints/last.pt", "cpu")
            initial = torch.load(directory / "fit/checkpoints/step0.pt", weights_only=False)
            rendered = torch.load(directory / "final/train.pt", weights_only=False)
            torch.testing.assert_close(render_direct(model, codes, 2), rendered["reconstructions"], atol=0, rtol=0)
            self.assertEqual(result["training_coordinate_queries"], 3 * 2 * 12)
            for key in ("uploads.0.weight", "queries.0.weight", "quantum.weights"):
                self.assertFalse(torch.equal(saved["decoder_state"][key], initial["decoder_state"][key]))
            spec = {"observed_pixels": list(range(128)), "unobserved_pixels": list(range(128, 1024)),
                    "steps": 2, "code_lr": .001, "code_init_std": .01, "code_norm_weight": .0001}
            first, second = directory / "first", directory / "second"
            first.mkdir()
            second.mkdir()
            heldout = images[:2].clone()
            fit_completion(model, heldout, [21, 22], spec, 37, first, "cpu")
            heldout.flatten(2)[:, :, 128:] = 1 - heldout.flatten(2)[:, :, 128:]
            fit_completion(model, heldout, [21, 22], spec, 37, second, "cpu")
            a = torch.load(first / "completion.pt", weights_only=False)
            b = torch.load(second / "completion.pt", weights_only=False)
            torch.testing.assert_close(a["codes"], b["codes"], atol=0, rtol=0)
            torch.testing.assert_close(a["reconstructions"], b["reconstructions"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
