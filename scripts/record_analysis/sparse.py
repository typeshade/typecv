"""Sparse-stage statistics of a reconstruction export: focal, poses, track lengths, the views that
share landmarks, and the first-to-last camera turn.

Keyframe indices count the views in time order (by time, then id), not in the file's order: the
export of record 58532c69, for example, lists its first five keyframes as 0, 17, 13, 9, 5.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .geometry import direction_angle_deg
from .scene import Reconstruction

DIAGNOSTIC_KEYS = ('historyRefinement', 'initialPair', 'meanReprojectionErrorPx', 'focalRefinement', 'sharedFocal')


@dataclass(frozen=True)
class Distribution:
    """Mean, median, 90th percentile and maximum of a sample."""

    mean: float
    median: float
    p90: float
    max: float

    @classmethod
    def of(cls, values: np.ndarray) -> Distribution:
        """The distribution of a non-empty sample."""
        return cls(float(np.mean(values)), float(np.median(values)), float(np.percentile(values, 90)),
                   float(np.max(values)))


@dataclass(frozen=True)
class ViewConnectivity:
    """The views that share at least the threshold of landmarks with the view at `index` (time
    order), the largest index gap among them, and the shared counts with the three views before
    and the three after it."""

    index: int
    view_id: int
    time: float | None
    connected: int
    max_index_gap: int
    neighbours: tuple[int, ...]


@dataclass(frozen=True)
class SparseSummary:
    """The sparse statistics of one export.

    Depths are camera-axis z in model units. `path_extent` is the largest distance of a camera
    centre from their mean, `mean_step` the mean distance between consecutive centres (time
    order). `turn_deg` is the angle between the first and the last view's optical axes.
    `baseline_over_depth` is the median of the consecutive steps over the median landmark depth
    from the first view. `focal_search` comes from the export's diagnostics or, when absent
    there, from the embedded run record's result.
    """

    format: str | None
    source: str | None
    assumed_focal_ratio: float | None
    views: int
    landmarks: int
    observations: int
    width: int | None
    height: int | None
    first_camera: Mapping[str, float]
    fx_min: float
    fx_max: float
    focal_ratio: float | None
    focal_search: Mapping[str, Any] | None
    diagnostics: Mapping[str, Any]
    view_times: tuple[tuple[int, float | None], ...]
    turn_deg: float
    max_turn_from_first_deg: float
    path_extent: float
    mean_step: float
    first_view_depth: tuple[float, float, float]
    track_length: Distribution
    track_length_histogram: Mapping[int, int]
    track_span: Distribution
    min_shared: int
    connected_pairs: int
    pair_gap_histogram: Mapping[int, int]
    connectivity: tuple[ViewConnectivity, ...]
    time_window: tuple[float, float]
    views_in_window: tuple[tuple[int, float], ...]
    baseline_over_depth: float


def visibility(reconstruction: Reconstruction, order: Mapping[int, int]) -> np.ndarray:
    """The 0/1 matrix (landmarks, views) of which view observes which landmark; the columns follow
    `order`, a map from view id to column."""
    columns = np.array([order[v] for v in reconstruction.observations.view], dtype=np.int64)
    matrix = np.zeros((len(reconstruction.xyz), len(order)), dtype=np.int64)
    matrix[reconstruction.observations.landmark, columns] = 1
    return matrix


def shared_landmarks(seen: np.ndarray) -> np.ndarray:
    """The number of landmarks each pair of views shares (V, V), zero on the diagonal."""
    shared = seen.T @ seen
    np.fill_diagonal(shared, 0)
    return shared


def track_spans(seen: np.ndarray) -> np.ndarray:
    """The last minus the first column (keyframe index) that observes each landmark."""
    first = np.argmax(seen, axis=1)
    last = seen.shape[1] - 1 - np.argmax(seen[:, ::-1], axis=1)
    return last - first


def _connectivity(
    shared: np.ndarray, index: int, view_id: int, time: float | None, min_shared: int,
) -> ViewConnectivity:
    count = len(shared)
    connected = np.flatnonzero(shared[index] >= min_shared)
    gap = int(np.max(np.abs(connected - index))) if len(connected) else 0
    window = [j for j in range(max(0, index - 3), min(count, index + 4)) if j != index]
    return ViewConnectivity(index, view_id, time, len(connected), gap, tuple(int(shared[index, j]) for j in window))


def _focal_search(reconstruction: Reconstruction) -> Mapping[str, Any] | None:
    return reconstruction.diagnostics.get('focalSearch') or reconstruction.run_result.get('focalSearch')


def sparse_summary(
    reconstruction: Reconstruction,
    min_shared: int = 16,
    time_window: tuple[float, float] = (28.0, 32.0),
) -> SparseSummary:
    """The sparse statistics of the reconstruction."""
    views = reconstruction.time_order()
    first = views[0]
    order = {v.id: i for i, v in enumerate(views)}
    fx = np.array([v.camera.fx for v in views])
    widths = np.array([v.width or np.nan for v in views], float)
    axes = np.array([v.pose.forward for v in views])
    centers = np.array([v.pose.center for v in views])
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    depth = first.pose.to_camera(reconstruction.xyz)[:, 2]
    depth = depth[depth > 0]
    lengths = reconstruction.track_lengths()
    seen = visibility(reconstruction, order)
    shared = shared_landmarks(seen)
    upper = np.triu(shared, 1) >= min_shared
    gaps = np.diff(np.argwhere(upper), axis=1)[:, 0]
    count = len(views)
    sample = [0, count // 4, count // 2, 3 * count // 4, count - 1]
    window = tuple((v.id, v.time) for v in views if v.time is not None and time_window[0] <= v.time <= time_window[1])
    return SparseSummary(
        format=reconstruction.format,
        source=reconstruction.source,
        assumed_focal_ratio=reconstruction.assumed_focal_ratio,
        views=count,
        landmarks=len(reconstruction.xyz),
        observations=len(reconstruction.observations.view),
        width=first.width,
        height=first.height,
        first_camera=asdict(first.camera),
        fx_min=float(fx.min()),
        fx_max=float(fx.max()),
        focal_ratio=None if np.isnan(widths).any() else float(np.mean(fx / widths)),
        focal_search=_focal_search(reconstruction),
        diagnostics={k: reconstruction.diagnostics[k] for k in DIAGNOSTIC_KEYS if k in reconstruction.diagnostics},
        view_times=tuple((v.id, v.time) for v in views),
        turn_deg=float(direction_angle_deg(axes[0], axes[-1])),
        max_turn_from_first_deg=float(np.max(direction_angle_deg(axes[0], axes))),
        path_extent=float(np.linalg.norm(centers - centers.mean(0), axis=1).max()),
        mean_step=float(np.mean(steps)),
        first_view_depth=tuple(float(p) for p in np.percentile(depth, [1, 50, 99])),
        track_length=Distribution.of(lengths),
        track_length_histogram={int(n): int(c) for n, c in enumerate(np.bincount(lengths)) if c},
        track_span=Distribution.of(track_spans(seen)),
        min_shared=min_shared,
        connected_pairs=int(upper.sum()),
        pair_gap_histogram={int(g): int(c) for g, c in zip(*np.unique(gaps, return_counts=True), strict=True)},
        connectivity=tuple(_connectivity(shared, i, views[i].id, views[i].time, min_shared) for i in sample),
        time_window=time_window,
        views_in_window=window,
        baseline_over_depth=float(np.median(steps / np.median(depth))),
    )
