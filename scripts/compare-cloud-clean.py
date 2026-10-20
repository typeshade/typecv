"""Reference values of the dense cloud cleanup from scikit-learn (docs/dense-mvs.md, 3.12).

Usage: python -I scripts/compare-cloud-clean.py cloud.f32 out.json k eps minPoints voxel
`cloud.f32` holds x, y, z float32 per point. The script prints nothing; it writes
- `meanDistances`: the mean distance to the k nearest neighbours, the point itself left out
  (NearestNeighbors with k + 1 neighbours), in float64;
- `labels`: DBSCAN(eps, min_samples=minPoints) labels on the float64 positions;
- `voxelCount`: the occupied voxels of side `voxel`, from `floor((p - min) / voxel)`.
Needs numpy and scikit-learn (SKLEARN_PATH may name an extra import directory).
"""
import json
import os
import sys

if os.environ.get("SKLEARN_PATH"):
    sys.path.insert(0, os.environ["SKLEARN_PATH"])
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

path, out, k, eps, min_points, voxel = sys.argv[1:7]
k, min_points, eps, voxel = int(k), int(min_points), float(eps), float(voxel)
points = np.fromfile(path, dtype=np.float32).reshape(-1, 3).astype(np.float64)
distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points)
labels = DBSCAN(eps=eps, min_samples=min_points).fit(points).labels_
cells = np.floor((points - points.min(axis=0)) / voxel).astype(np.int64)
with open(out, "w") as f:
    json.dump(
        {
            "sklearn": __import__("sklearn").__version__,
            "meanDistances": distances[:, 1:].mean(axis=1).tolist(),
            "labels": labels.tolist(),
            "voxelCount": int(len(np.unique(cells, axis=0))),
        },
        f,
    )
