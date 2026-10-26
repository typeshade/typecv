"""Tests of the shared geometry: octahedral decode, robust plane, poses, blocks, finite differences."""
from __future__ import annotations

import unittest

import numpy as np

from record_analysis.geometry import (
    Block,
    Camera,
    Pose,
    align_cameras,
    backproject,
    block_grid,
    camera_center,
    decode_octahedral,
    direction_angle_deg,
    finite_difference_normals,
    line_angle_deg,
    plane_angle_deg,
    project,
    robust_plane,
    rotation_angle_deg,
    unit,
)
from record_analysis.tests.synthetic import encode_octahedral, rotation_y


def rotation_z(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q * np.sign(np.linalg.det(q))


class OctahedralTest(unittest.TestCase):
    def test_round_trip_against_the_reference_encoder(self):
        rng = np.random.default_rng(0)
        normals = unit(rng.normal(size=(20000, 3)))
        axes = np.vstack([np.eye(3), -np.eye(3), unit(np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1]]))])
        normals = np.vstack([normals, axes])
        decoded = decode_octahedral(encode_octahedral(normals))
        self.assertEqual(decoded.shape, normals.shape)
        np.testing.assert_allclose(np.linalg.norm(decoded, axis=1), 1, atol=1e-12)
        error = np.degrees(np.arccos(np.clip(np.sum(decoded * normals, axis=1), -1, 1)))
        self.assertLess(error.max(), 0.01)

    def test_bit_layout(self):
        self.assertTrue(np.allclose(decode_octahedral(np.array([0], np.uint32)), [[0, 0, 1]]))
        x_only = decode_octahedral(np.array([0x4000], np.uint32))[0]
        self.assertGreater(x_only[0], 0)
        self.assertAlmostEqual(x_only[1], 0)
        y_only = decode_octahedral(np.array([0x4000 << 16], np.uint32))[0]
        self.assertGreater(y_only[1], 0)
        self.assertAlmostEqual(y_only[0], 0)

    def test_most_negative_snorm_reads_as_minus_one(self):
        np.testing.assert_allclose(decode_octahedral(np.array([0x8000], np.uint32))[0], [-1, 0, 0], atol=1e-12)

    def test_keeps_the_grid_shape(self):
        words = encode_octahedral(np.tile([0.0, 0.6, -0.8], (4, 5, 1)))
        self.assertEqual(decode_octahedral(words).shape, (4, 5, 3))


class RobustPlaneTest(unittest.TestCase):
    def test_noisy_plane_with_outliers(self):
        rng = np.random.default_rng(1)
        xy = rng.uniform(-1, 1, size=(400, 2))
        z = 5 + 0.3 * xy[:, 0] - 0.2 * xy[:, 1] + rng.normal(scale=0.005, size=400)
        points = np.column_stack([xy, z])
        points[:80, 2] += rng.uniform(0.5, 2.0, size=80)
        fit = robust_plane(points)
        truth = unit(np.array([0.3, -0.2, -1.0]))
        self.assertLess(line_angle_deg(fit.normal, truth), 0.5)
        self.assertLessEqual(fit.normal[2], 0)
        self.assertGreater(fit.inlier_share, 0.75)
        self.assertLess(fit.inlier_share, 0.85)
        self.assertLess(fit.rms_over_depth, 0.005)

    def test_too_few_points(self):
        self.assertIsNone(robust_plane(np.zeros((11, 3))))

    def test_plane_angle(self):
        a = robust_plane(np.column_stack([np.random.default_rng(2).uniform(-1, 1, (50, 2)), np.full(50, 4.0)]))
        self.assertAlmostEqual(plane_angle_deg(a, a), 0.0, places=6)
        self.assertIsNone(plane_angle_deg(a, None))


class PoseTest(unittest.TestCase):
    def setUp(self):
        self.R = rotation_y(30) @ rotation_z(10)
        self.center = np.array([1.0, -2.0, 0.5])
        self.pose = Pose(self.R, -self.R @ self.center)

    def test_center_maps_to_the_camera_origin(self):
        np.testing.assert_allclose(self.pose.center, self.center, atol=1e-12)
        np.testing.assert_allclose(camera_center(self.pose.R, self.pose.t), self.center, atol=1e-12)
        np.testing.assert_allclose(self.pose.to_camera(self.center[None])[0], 0, atol=1e-12)

    def test_forward_axis_is_row_three(self):
        ahead = self.center + 3 * self.pose.forward
        np.testing.assert_allclose(self.pose.to_camera(ahead[None])[0], [0, 0, 3], atol=1e-12)

    def test_from_json_reads_row_major(self):
        pose = Pose.from_json({'R': self.R.reshape(-1).tolist(), 't': self.pose.t.tolist()})
        np.testing.assert_allclose(pose.R, self.R)

    def test_projection(self):
        camera = Camera(100, 80, 10, 20)
        u, v, front = project(np.array([[1.0, 2.0, 4.0], [1.0, 1.0, -1.0]]), camera)
        self.assertEqual((u[0], v[0]), (35.0, 60.0))
        self.assertEqual(front.tolist(), [True, False])

    def test_angles(self):
        self.assertAlmostEqual(rotation_angle_deg(rotation_z(30), np.eye(3)), 30, places=9)
        self.assertAlmostEqual(float(direction_angle_deg(np.array([2.0, 0, 0]), np.array([0, 0, 5.0]))), 90, places=9)
        self.assertAlmostEqual(float(direction_angle_deg(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]))), 180, places=6)
        self.assertAlmostEqual(float(line_angle_deg(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]))), 0, places=6)

    def test_similarity_alignment_recovers_a_known_transform(self):
        rng = np.random.default_rng(3)
        rotations = [random_rotation(rng) for _ in range(6)]
        centers = rng.normal(size=(6, 3))
        Q, scale, shift = random_rotation(rng), 2.5, np.array([4.0, -1.0, 2.0])
        reference_rotations = [r @ Q.T for r in rotations]
        reference_centers = scale * centers @ Q.T + shift
        similarity = align_cameras(rotations, centers, reference_rotations, reference_centers)
        np.testing.assert_allclose(similarity.rotation, Q, atol=1e-9)
        self.assertAlmostEqual(similarity.scale, scale, places=9)
        np.testing.assert_allclose(similarity.apply(centers), reference_centers, atol=1e-9)


class BlockGridTest(unittest.TestCase):
    def test_lab_grid(self):
        blocks = block_grid(1280, 720, 4, 3)
        self.assertEqual(len(blocks), 12)
        self.assertEqual(blocks[5], Block(1, 1, 320, 640, 240, 480))
        self.assertTrue(all(b.x1 - b.x0 == 320 and b.y1 - b.y0 == 240 for b in blocks))

    def test_uneven_grid_tiles_the_image(self):
        coverage = np.zeros((7, 10), int)
        for b in block_grid(10, 7, 3, 2):
            coverage[b.y0:b.y1, b.x0:b.x1] += 1
        self.assertTrue((coverage == 1).all())

    def test_contains_is_half_open(self):
        block = Block(0, 0, 0, 10, 0, 5)
        inside = block.contains(np.array([[0.0, 0.0], [9.99, 4.99], [10.0, 1.0], [1.0, 5.0], [-0.01, 1.0]]))
        self.assertEqual(inside.tolist(), [True, True, False, False, False])

    def test_central_ray(self):
        camera = Camera(100, 100, 50, 30)
        np.testing.assert_allclose(Block(0, 0, 0, 100, 0, 60).ray(camera), [0, 0, 1])
        ray = Block(1, 0, 100, 200, 0, 60).ray(camera)
        self.assertAlmostEqual(float(np.degrees(np.arctan2(ray[0], ray[2]))), 45, places=9)


class DepthSurfaceTest(unittest.TestCase):
    def test_backprojection_of_the_principal_point(self):
        camera = Camera(50, 50, 2, 1)
        points = backproject(np.full((3, 5), 4.0, np.float32), camera)
        np.testing.assert_allclose(points[1, 2], [0, 0, 4])
        np.testing.assert_allclose(points[1, 4], [4 * 2 / 50, 0, 4])

    def test_finite_differences_of_a_plane(self):
        camera = Camera(40, 40, 15.5, 11.5)
        v, u = np.mgrid[0:24, 0:32]
        rays = np.stack([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, np.ones(u.shape)], -1)
        plane = np.array([-0.3, 0.4, 1.0])
        depth = 6 / (rays @ plane)
        valid = np.ones(depth.shape, bool)
        valid[10, 10] = False
        normals, mask = finite_difference_normals(backproject(depth, camera), valid, 2)
        self.assertFalse(mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any())
        self.assertFalse(mask[10, 8] or mask[10, 12] or mask[8, 10] or mask[12, 10] or mask[10, 10])
        self.assertTrue(mask[5, 5])
        # arccos near 1 resolves about 1e-6 degrees in float64.
        self.assertLess(line_angle_deg(normals[mask], unit(plane)).max(), 1e-5)


if __name__ == '__main__':
    unittest.main()
