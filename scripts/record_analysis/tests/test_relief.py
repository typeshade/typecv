"""Tests of the depth-relief statistics."""
from __future__ import annotations

import unittest

import numpy as np

from record_analysis.relief import (
    least_squares_slope,
    model_relief,
    robust_scatter,
    theil_sen_slope,
    third_bounds,
    third_masks,
)
from record_analysis.scene import reconstruction_from_json
from record_analysis.tests import synthetic


class SlopeTest(unittest.TestCase):
    def test_power_law_depths(self):
        rng = np.random.default_rng(5)
        reference = rng.uniform(3, 60, size=3000)
        measured = 4.0 * reference ** -0.35
        x, y = np.log(reference), np.log(measured)
        self.assertAlmostEqual(theil_sen_slope(x, y), -0.35, places=9)
        self.assertAlmostEqual(least_squares_slope(x, y), -0.35, places=9)

    def test_theil_sen_ignores_outliers_and_float32_depths(self):
        rng = np.random.default_rng(6)
        x = np.log(rng.uniform(3, 60, size=2000))
        y = (0.5 * x + rng.normal(scale=0.01, size=2000)).astype(np.float32)
        y[:200] += rng.uniform(1, 3, size=200).astype(np.float32)
        self.assertAlmostEqual(theil_sen_slope(x, y), 0.5, delta=0.02)
        self.assertGreater(abs(least_squares_slope(x, y) - 0.5), abs(theil_sen_slope(x, y) - 0.5))

    def test_subsample_is_reproducible(self):
        rng = np.random.default_rng(7)
        x, y = rng.normal(size=4000), rng.normal(size=4000)
        self.assertEqual(theil_sen_slope(x, y, seed=0), theil_sen_slope(x, y, seed=0))
        self.assertNotEqual(theil_sen_slope(x, y, seed=0), theil_sen_slope(x, y, seed=1))

    def test_robust_scatter_of_a_normal_sample(self):
        sample = np.random.default_rng(8).normal(scale=0.2, size=100000)
        self.assertAlmostEqual(robust_scatter(sample), 0.2, delta=0.005)


class ThirdsTest(unittest.TestCase):
    def test_lab_image_bounds(self):
        self.assertEqual(third_bounds(1280), [(0, 427), (427, 853), (853, 1280)])
        self.assertEqual(third_bounds(720), [(0, 240), (240, 480), (480, 720)])

    def test_masks_are_half_open(self):
        masks = third_masks(np.array([0.0, 239.9, 240.0, 479.0, 480.0, 719.9, 720.0, -0.5]), 720)
        self.assertEqual([m.nonzero()[0].tolist() for m in masks], [[0, 1], [2, 3], [4, 5]])


class ModelReliefTest(unittest.TestCase):
    def test_plane_relief(self):
        model = reconstruction_from_json(synthetic.export_json())
        relief = model_relief('plane', model, [0, 5], 1280, 720)
        self.assertEqual(relief.unregistered, (5,))
        self.assertAlmostEqual(relief.turn_deg, 4.0, places=9)
        self.assertAlmostEqual(relief.ratio, 50 / 64)
        view = relief.relief[0]
        self.assertEqual(view.landmarks, 130)
        # z = 5 + 0.3 x + 0.2 y grows down the image and to the right.
        self.assertLess(view.rows[0], view.rows[2])
        self.assertLess(view.columns[0], view.columns[2])
        self.assertGreater(view.spread, 1)


if __name__ == '__main__':
    unittest.main()
