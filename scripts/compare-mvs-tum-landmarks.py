"""Sparse landmarks for scripts/compare-mvs-tum.mjs (docs/dense-mvs.md 7.3).

Triangulates SIFT matches with the sample's motion-capture poses, so the landmarks come from the
images and the poses and never from the ground-truth depth maps. Writes `{views, landmarks}` JSON.

  python scripts/compare-mvs-tum-landmarks.py website/public/sample/tum-fr3-structure-texture-far OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

RATIO = 0.8
EPIPOLAR_PX = 2.0
REPROJECTION_PX = 1.5


def main() -> None:
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    sample = json.loads((folder / 'sample.json').read_text())
    frames = sample['frames']
    sift = cv2.SIFT_create(4000)
    features = []
    for f in frames:
        gray = cv2.imread(str(folder / f['file']), cv2.IMREAD_GRAYSCALE)
        points, descriptors = sift.detectAndCompute(gray, None)
        features.append((np.array([p.pt for p in points]), descriptors))
    k = [np.array([[f['camera']['fx'], 0, f['camera']['cx']], [0, f['camera']['fy'], f['camera']['cy']], [0, 0, 1]]) for f in frames]
    r = [np.array(f['reference']['R']).reshape(3, 3) for f in frames]
    t = [np.array(f['reference']['t']) for f in frames]
    proj = [k[i] @ np.hstack([r[i], t[i].reshape(3, 1)]) for i in range(len(frames))]
    matcher = cv2.BFMatcher()
    landmarks = []
    for i in range(len(frames)):
        found: dict[int, list[tuple[int, np.ndarray]]] = {}
        for j in range(len(frames)):
            if j == i:
                continue
            # Fundamental matrix of the pair from the poses: x_j^T F x_i = 0.
            rel_r = r[j] @ r[i].T
            rel_t = t[j] - rel_r @ t[i]
            tx = np.array([[0, -rel_t[2], rel_t[1]], [rel_t[2], 0, -rel_t[0]], [-rel_t[1], rel_t[0], 0]])
            fm = np.linalg.inv(k[j]).T @ tx @ rel_r @ np.linalg.inv(k[i])
            pairs = matcher.knnMatch(features[i][1], features[j][1], k=2)
            for m in (p for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance):
                a = np.append(features[i][0][m[0].queryIdx], 1)
                b = np.append(features[j][0][m[0].trainIdx], 1)
                line = fm @ a
                if abs(b @ line) / np.hypot(line[0], line[1]) < EPIPOLAR_PX:
                    found.setdefault(m[0].queryIdx, []).append((j, b[:2]))
        for q, observations in found.items():
            views = [(i, features[i][0][q])] + observations
            if len(views) < 3:
                continue
            rows = []
            for v, px in views:
                rows += [px[0] * proj[v][2] - proj[v][0], px[1] * proj[v][2] - proj[v][1]]
            x = np.linalg.svd(np.array(rows))[2][-1]
            if abs(x[3]) < 1e-12:
                continue
            x = x[:3] / x[3]
            good = []
            for v, px in views:
                c = r[v] @ x + t[v]
                if c[2] <= 0:
                    continue
                uv = (k[v] @ c)[:2] / c[2]
                if np.hypot(*(uv - px)) < REPROJECTION_PX:
                    good.append({'viewId': v, 'pixel': [float(px[0]), float(px[1])]})
            if len(good) >= 3:
                landmarks.append({'xyz': x.tolist(), 'observations': good})
    out.write_text(json.dumps({'landmarks': landmarks}))
    print(json.dumps({'landmarks': len(landmarks), 'meanObservations': float(np.mean([len(l['observations']) for l in landmarks]))}))


if __name__ == '__main__':
    main()
