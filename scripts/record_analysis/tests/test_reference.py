"""Tests of the COLMAP comparisons on a synthetic reference (no pycolmap needed): the synthetic
TypeCV record seen through a known similarity, with COLMAP's +0.5 pixel convention."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from record_analysis.archive import DepthArchive
from record_analysis.colmap import ColmapCamera, ColmapImage, ColmapModel, view_id_from_name
from record_analysis.geometry import Camera, Pose
from record_analysis.reference import common_pixel_relief, depth_spans, pose_agreement, reference_blocks
from record_analysis.scene import load_reconstruction
from record_analysis.tests import synthetic

Q = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
SCALE, SHIFT = 3.0, np.array([1.0, 2.0, -4.0])


def similar_model(reconstruction) -> ColmapModel:
    """COLMAP's world = SCALE * Q * TypeCV's world + SHIFT; pixels in COLMAP's convention."""
    camera = synthetic.CAMERA
    colmap_camera = ColmapCamera(1, 'SIMPLE_PINHOLE', (camera.fx, camera.cx + 0.5, camera.cy + 0.5),
                                 synthetic.WIDTH, synthetic.HEIGHT,
                                 Camera(camera.fx, camera.fy, camera.cx + 0.5, camera.cy + 0.5))
    world = SCALE * reconstruction.xyz @ Q.T + SHIFT
    images = {}
    for view in reconstruction.views:
        R = view.pose.R @ Q.T
        center = SCALE * Q @ view.pose.center + SHIFT
        landmarks, pixels = reconstruction.observed_by(view.id)
        images[view.id] = ColmapImage(view.id, f'view-{view.id:03d}.jpg', 1, Pose(R, -R @ center),
                                      pixels + 0.5, world[landmarks])
    return ColmapModel({1: colmap_camera}, images, world)


class ReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        paths = synthetic.write_record(Path(cls.folder.name))
        cls.reconstruction = load_reconstruction(paths['export'])
        cls.archive = DepthArchive(paths['archive'])
        cls.model = similar_model(cls.reconstruction)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_image_names(self):
        self.assertEqual(view_id_from_name('view-007.jpg'), 7)
        with self.assertRaises(ValueError):
            view_id_from_name('frame-7.jpg')

    def test_pose_agreement_after_a_similarity(self):
        r = pose_agreement(self.reconstruction, self.model)
        self.assertEqual(r.common_views, 2)
        self.assertAlmostEqual(r.scale, SCALE, places=9)
        self.assertLess(r.rotation_max_deg, 1e-5)
        self.assertLess(r.center_max, 1e-9)
        self.assertAlmostEqual(r.typecv_turn_deg, r.colmap_turn_deg, places=9)

    def test_depth_spans_are_scale_free(self):
        span = depth_spans(self.reconstruction, self.model, [0])[0]
        self.assertAlmostEqual(span.typecv, span.colmap, places=9)

    def test_plane_angles_are_frame_free(self):
        r = reference_blocks(self.reconstruction, self.model, 0, grid=(2, 2), pairs=[((0, 0), (1, 0))])
        self.assertAlmostEqual(r.pairs[0].typecv_deg, r.pairs[0].colmap_deg, places=5)
        half_pixel_deg = np.degrees(np.arctan(np.hypot(0.5, 0.5) / synthetic.CAMERA.fx))
        for block in r.blocks:
            self.assertEqual(block.typecv_points, block.colmap_points)
            # Each system's blocks and central rays use its own pixel convention.
            self.assertLess(abs(block.typecv_ray_deg - block.colmap_ray_deg), half_pixel_deg)

    def test_relief_on_common_pixels(self):
        r = common_pixel_relief(self.archive, self.model, 0, sample=50, min_third_pixels=5)
        landmarks, pixels = self.reconstruction.observed_by(0)
        self.assertEqual(r.pixels, int((np.round(pixels[:, 0]) >= synthetic.INVALID_COLUMNS).sum()))
        self.assertAlmostEqual(r.theil_sen_slope, 1.0, places=5)
        self.assertAlmostEqual(r.least_squares_slope, 1.0, places=5)
        self.assertAlmostEqual(r.colmap_spread, r.typecv_spread, places=5)
        self.assertLess(r.residual_scatter, 1e-5)
        self.assertEqual([t.name for t in r.rows], ['top third', 'middle', 'bottom third'])
        for third in r.rows + r.columns:
            self.assertAlmostEqual(third.colmap, third.typecv, places=5)


if __name__ == '__main__':
    unittest.main()
