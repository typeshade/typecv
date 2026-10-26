"""Tests of the reconstruction export loader and the sparse statistics."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from record_analysis.scene import load_reconstruction, reconstruction_from_json
from record_analysis.sparse import shared_landmarks, sparse_summary, track_spans, visibility


def view(view_id: int, time: float, angle: float = 0.0) -> dict:
    a = np.radians(angle)
    R = [np.cos(a), 0, np.sin(a), 0, 1, 0, -np.sin(a), 0, np.cos(a)]
    return {'id': view_id, 'time': time, 'width': 100, 'height': 80,
            'camera': {'fx': 60, 'fy': 60, 'cx': 49.5, 'cy': 39.5}, 'pose': {'R': R, 't': [0, 0, 0]}}


def point(xyz: list[float], view_ids: list[int]) -> dict:
    return {'xyz': xyz, 'observations': [{'viewId': v, 'featureId': i, 'pixel': [10.0 + i, 20.0]}
                                         for i, v in enumerate(view_ids)]}


def model() -> dict:
    """Four views listed out of time order (ids 0, 3, 2, 1 at times 0, 3, 2, 1)."""
    views = [view(0, 0.0), view(3, 3.0, 30.0), view(2, 2.0, 20.0), view(1, 1.0, 10.0)]
    landmarks = [point([0, 0, 5], [0, 1]), point([1, 0, 6], [0, 3]), point([0, 1, 7], [1, 2, 3]),
                 point([0, 0, -5], [2, 3])]
    landmarks += [point([0.1 * i, 0, 5], [1, 2]) for i in range(16)]
    return {'format': 'typecv-reconstruction-v1', 'views': views, 'landmarks': landmarks,
            'runRecord': {'result': {'focalSearch': {'ratio': 0.6, 'gridRatio': 0.3, 'views': 5}}}}


class SceneTest(unittest.TestCase):
    def test_load_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'r.json'
            path.write_text(json.dumps(model()))
            rec = load_reconstruction(path)
        self.assertEqual([v.id for v in rec.views], [0, 3, 2, 1])
        self.assertEqual([v.id for v in rec.time_order()], [0, 1, 2, 3])
        self.assertEqual(rec.view(2).time, 2.0)
        self.assertEqual(rec.track_lengths().tolist(), [2, 2, 3, 2] + [2] * 16)

    def test_refuses_another_format(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'r.json'
            path.write_text(json.dumps({**model(), 'format': 'typecv-depth-archive-v1'}))
            with self.assertRaises(ValueError):
                load_reconstruction(path)

    def test_observed_by_keeps_the_first_observation(self):
        data = model()
        data['landmarks'][0]['observations'].append({'viewId': 0, 'featureId': 9, 'pixel': [99.0, 98.0]})
        rec = reconstruction_from_json(data)
        landmarks, pixels = rec.observed_by(0)
        self.assertEqual(landmarks.tolist(), [0, 1])
        self.assertEqual(pixels[0].tolist(), [10.0, 20.0])

    def test_sweep_model_without_sizes(self):
        data = {'ratio': 0.97, 'views': [{'id': 5, 'camera': {'fx': 1, 'fy': 1, 'cx': 0, 'cy': 0},
                                          'pose': {'R': np.eye(3).reshape(-1).tolist(), 't': [0, 0, 0]}}],
                'landmarks': []}
        rec = reconstruction_from_json(data)
        self.assertEqual(rec.ratio, 0.97)
        self.assertIsNone(rec.views[0].width)
        self.assertEqual(rec.xyz.shape, (0, 3))


class SparseTest(unittest.TestCase):
    def test_keyframe_indices_follow_time(self):
        rec = reconstruction_from_json(model())
        order = {v.id: i for i, v in enumerate(rec.time_order())}
        seen = visibility(rec, order)
        self.assertEqual(track_spans(seen)[:4].tolist(), [1, 3, 2, 1])
        shared = shared_landmarks(seen)
        self.assertEqual(shared[1, 2], 17)
        self.assertEqual(shared[2, 1], 17)
        self.assertEqual(np.diag(shared).tolist(), [0, 0, 0, 0])

    def test_summary(self):
        s = sparse_summary(reconstruction_from_json(model()), min_shared=16, time_window=(0.5, 2.5))
        self.assertEqual((s.views, s.landmarks, s.observations), (4, 20, 41))
        self.assertAlmostEqual(s.turn_deg, 30.0, places=9)
        self.assertEqual(s.connected_pairs, 1)
        self.assertEqual(dict(s.pair_gap_histogram), {1: 1})
        self.assertEqual(s.views_in_window, ((1, 1.0), (2, 2.0)))
        self.assertEqual(s.focal_search['ratio'], 0.6)
        self.assertAlmostEqual(s.focal_ratio, 0.6)
        self.assertEqual(s.first_view_depth[1], 5.0)
        self.assertEqual(s.track_length.median, 2.0)


if __name__ == '__main__':
    unittest.main()
