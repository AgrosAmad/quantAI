"""Guard native-resolution scoring and the published RAE integration contract."""

import math
import unittest

import torch

from experiments.qvf_project_a.rae_baseline import metrics, native_reconstruction


class TestRAEScoring(unittest.TestCase):
    def test_psnr_is_mean_of_per_image_scores(self):
        target = torch.zeros(2, 3, 32, 32)
        prediction = target.clone()
        prediction[0] = .1
        prediction[1] = .5
        score = metrics(prediction, target)
        self.assertAlmostEqual(score["mse"], .13, places=7)
        self.assertAlmostEqual(score["psnr_db"], (20 - 10 * math.log10(.25)) / 2, places=6)
        self.assertEqual(score["gradient_mse"], 0)

    def test_native_resize_does_not_clip_or_round(self):
        high_resolution = torch.full((2, 3, 256, 256), 1.2)
        native = native_reconstruction(high_resolution)
        self.assertEqual(tuple(native.shape), (2, 3, 32, 32))
        torch.testing.assert_close(native, torch.full_like(native, 1.2), rtol=0, atol=1e-6)

    def test_invalid_or_nonfinite_images_rejected(self):
        with self.assertRaises(ValueError):
            native_reconstruction(torch.zeros(2, 3, 224, 224))
        with self.assertRaises(ValueError):
            metrics(torch.zeros(2, 3, 32, 32), torch.zeros(1, 3, 32, 32))
        with self.assertRaises(ValueError):
            metrics(torch.full((2, 3, 32, 32), float("nan")), torch.zeros(2, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
