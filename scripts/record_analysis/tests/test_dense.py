"""Tests of the dense-stage analyses on the synthetic plane record."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from record_analysis.archive import DepthArchive
from record_analysis.dense import (
    block_planes,
    dense_normal_angles,
    filtered_surface_normals,
    seed_agreement,
    stage_normals,
)
from record_analysis.geometry import line_angle_deg
from record_analysis.ply import read_dense_ply
from record_analysis.scene import Observations, Reconstruction, load_reconstruction
from record_analysis.tests import synthetic


class DenseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        paths = synthetic.write_record(Path(cls.folder.name))
        cls.reconstruction = load_reconstruction(paths['export'])
        cls.archive = DepthArchive(paths['archive'])
        cls.cloud = read_dense_ply(paths['ply'])

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_seeds_agree_on_an_exact_plane(self):
        r = seed_agreement(self.reconstruction, self.archive, 0)
        landmarks, pixels = self.reconstruction.observed_by(0)
        invalid = int((np.round(pixels[:, 0]) < synthetic.INVALID_COLUMNS).sum())
        self.assertEqual((r.observed, r.without_depth), (len(landmarks), invalid))
        self.assertLess(abs(r.median_relative), 1e-6)
        self.assertEqual(r.within, ((0.02, 1.0), (0.1, 1.0)))

    def test_seeds_refuse_a_distorted_camera(self):
        from dataclasses import replace

        from record_analysis.scene import Reconstruction
        view = self.reconstruction.view(0)
        distorted = replace(view, camera=replace(view.camera, k1=0.01))
        rec = Reconstruction((distorted,), self.reconstruction.xyz, self.reconstruction.observations)
        with self.assertRaises(ValueError):
            seed_agreement(rec, self.archive, 0)

    def test_seeds_of_a_view_that_observes_no_landmark(self):
        observations = self.reconstruction.observations
        kept = observations.view != 1
        without = Observations(observations.landmark[kept], observations.view[kept], observations.pixel[kept])
        rec = Reconstruction(self.reconstruction.views, self.reconstruction.xyz, without)
        r = seed_agreement(rec, self.archive, 1)
        self.assertEqual((r.observed, r.without_depth), (0, 0))
        self.assertTrue(np.isnan([r.median_relative, r.abs_relative_p50, r.abs_relative_p90]).all())
        self.assertEqual([t for t, _ in r.within], [0.02, 0.1])
        self.assertTrue(np.isnan([share for _, share in r.within]).all())

    def test_filtered_normals_leave_out_zero_medians(self):
        # A plane with a patch where two of three pixels are invalid: the 5 x 5 median is zero
        # there, and a zero depth would put a point at the camera centre.
        pose = synthetic.poses()[0]
        v, u = np.mgrid[0:synthetic.HEIGHT, 0:synthetic.WIDTH]
        valid = ~((v >= 12) & (v < 36) & (u >= 16) & (u < 48) & ((u + v) % 3 != 0))
        depth = np.where(valid, synthetic.plane_depth(pose), 0)
        normals, mask = filtered_surface_normals(depth, valid, synthetic.CAMERA, 6, 5)
        self.assertGreater(mask.sum(), 400)
        self.assertFalse(mask[20, 30])
        error = line_angle_deg(normals, pose.R @ synthetic.plane_normal())[mask]
        self.assertLess(error.max(), 10)

    def test_stored_normals_match_the_depth_surface(self):
        s, images = stage_normals(self.archive, 0, 'geometric-filtered', render=True)
        self.assertLess(s.stored_vs_surface_median_deg, 0.05)
        self.assertAlmostEqual(s.surface_raw.median_deg, s.stored.median_deg, delta=0.05)
        self.assertAlmostEqual(sum(s.surface_raw.shares), 1.0)
        self.assertAlmostEqual(s.valid_share, 1 - synthetic.INVALID_COLUMNS / synthetic.WIDTH)
        self.assertEqual(images.inverse_depth.shape, (synthetic.HEIGHT, synthetic.WIDTH))
        self.assertEqual(images.surface_angle.dtype, np.uint8)

    def test_block_planes_are_parallel_on_a_plane(self):
        r = block_planes(self.reconstruction, self.archive, self.cloud, 0, grid=(2, 2), depth_stride=1,
                         pairs=[((0, 0), (1, 1))])
        self.assertEqual(len(r.blocks), 4)
        self.assertEqual(r.ply_consistent, r.ply_in_view)
        self.assertGreater(r.ply_in_view, 0.9 * len(self.cloud.positions))
        pair = r.pairs[0]
        for angle in (pair.sparse_deg, pair.depth_deg, pair.ply_deg):
            self.assertLess(angle, 0.01)
        truth = synthetic.plane_normal()
        for block in r.blocks:
            for plane in (block.sparse, block.depth, block.ply):
                self.assertGreater(abs(float(plane.normal @ truth)), 1 - 1e-8)

    def test_dense_normal_angles(self):
        r = dense_normal_angles(self.reconstruction, self.cloud)
        self.assertEqual(r.points, len(self.cloud.positions))
        self.assertEqual([v.view_id for v in r.views], [0, 1])
        self.assertEqual(r.views[0].points, r.points)
        rays = self.cloud.positions / np.linalg.norm(self.cloud.positions, axis=1, keepdims=True)
        expected = np.degrees(np.arccos(np.abs(rays @ synthetic.plane_normal())))
        self.assertAlmostEqual(r.views[0].median_deg, float(np.median(expected)), places=4)
        self.assertEqual(sum(r.nearest_camera_histogram), r.points)


if __name__ == '__main__':
    unittest.main()
