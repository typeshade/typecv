"""Tests of the dense PLY reader and writer."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from record_analysis.ply import DenseCloud, read_dense_ply, write_dense_ply


def cloud(count: int, normals: bool) -> DenseCloud:
    rng = np.random.default_rng(4)
    n = rng.normal(size=(count, 3))
    return DenseCloud(
        positions=rng.normal(scale=10, size=(count, 3)),
        colors=rng.integers(0, 256, size=(count, 3), dtype=np.uint8),
        normals=n / np.linalg.norm(n, axis=1, keepdims=True) if normals else None,
    )


class PlyTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / 'dense.ply'

    def tearDown(self):
        self.folder.cleanup()

    def test_round_trip_with_normals(self):
        original = cloud(257, normals=True)
        write_dense_ply(self.path, original)
        read = read_dense_ply(self.path)
        np.testing.assert_array_equal(read.positions, original.positions.astype(np.float32))
        np.testing.assert_array_equal(read.normals, original.normals.astype(np.float32))
        np.testing.assert_array_equal(read.colors, original.colors)
        self.assertEqual(read.positions.dtype, np.float64)

    def test_round_trip_without_normals(self):
        original = cloud(10, normals=False)
        write_dense_ply(self.path, original)
        read = read_dense_ply(self.path)
        self.assertIsNone(read.normals)
        np.testing.assert_array_equal(read.colors, original.colors)

    def test_layout_of_the_lab_export(self):
        write_dense_ply(self.path, cloud(3, normals=True))
        data = self.path.read_bytes()
        header, _, body = data.partition(b'end_header\n')
        self.assertEqual(header.decode().splitlines(), [
            'ply', 'format binary_little_endian 1.0', 'comment typecv dense MVS', 'element vertex 3',
            'property float x', 'property float y', 'property float z',
            'property float nx', 'property float ny', 'property float nz',
            'property uchar red', 'property uchar green', 'property uchar blue',
        ])
        self.assertEqual(len(body), 3 * 27)
        write_dense_ply(self.path, cloud(3, normals=False))
        self.assertEqual(len(self.path.read_bytes().partition(b'end_header\n')[2]), 3 * 15)

    def test_refuses_ascii_and_truncated_files(self):
        self.path.write_bytes(b'ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n')
        with self.assertRaises(ValueError):
            read_dense_ply(self.path)
        write_dense_ply(self.path, cloud(4, normals=True))
        self.path.write_bytes(self.path.read_bytes()[:-5])
        with self.assertRaises(ValueError):
            read_dense_ply(self.path)


if __name__ == '__main__':
    unittest.main()
