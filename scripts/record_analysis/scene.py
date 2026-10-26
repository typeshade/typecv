"""The reconstruction export `typecv-reconstruction-v1` (website/src/lib/result-export.ts).

The export holds views with an id, a time, an image size, a camera and a world-to-camera pose,
and landmarks with a world position and observations (view id, feature id, pixel). The loader
also reads the sparse models of the fixed-focal sweep, which carry a `ratio` and views without a
time or an image size. Model units are the reconstruction's own, not metres. Pixels are in the
view's analysis image, x right and y down.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import Camera, Pose

RECONSTRUCTION_FORMAT = 'typecv-reconstruction-v1'


@dataclass(frozen=True)
class View:
    """A registered view: its camera and pose, and the optional time (s) and image size (px)."""

    id: int
    camera: Camera
    pose: Pose
    time: float | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class Observations:
    """Every observation of the file, in file order, as parallel arrays.

    `landmark` is the index of the observed landmark, `view` the view id and `pixel` the (x, y)
    pixel in the view's image.
    """

    landmark: np.ndarray
    view: np.ndarray
    pixel: np.ndarray


@dataclass(frozen=True)
class Reconstruction:
    """A sparse reconstruction: views in file order, landmark positions (M, 3), observations."""

    views: tuple[View, ...]
    xyz: np.ndarray
    observations: Observations
    format: str | None = None
    source: str | None = None
    assumed_focal_ratio: float | None = None
    ratio: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    run_result: Mapping[str, Any] = field(default_factory=dict)

    def view(self, view_id: int) -> View:
        """The view with this id, or KeyError."""
        for view in self.views:
            if view.id == view_id:
                return view
        raise KeyError(f'view {view_id} is not in the reconstruction')

    def has_view(self, view_id: int) -> bool:
        """True when a view has this id."""
        return any(view.id == view_id for view in self.views)

    def time_order(self) -> list[View]:
        """The views by time, then by id. Views without a time come last, by id."""
        return sorted(self.views, key=lambda v: (math.inf if v.time is None else v.time, v.id))

    def track_lengths(self) -> np.ndarray:
        """The number of observations of each landmark (M,)."""
        return np.bincount(self.observations.landmark, minlength=len(self.xyz))

    def observed_by(self, view_id: int) -> tuple[np.ndarray, np.ndarray]:
        """The landmarks that the view observes, in landmark order, and their pixels (n, 2).

        A landmark with more than one observation in the view counts once, with its first one.
        """
        selected = np.flatnonzero(self.observations.view == view_id)
        _, first = np.unique(self.observations.landmark[selected], return_index=True)
        chosen = selected[first]
        return self.observations.landmark[chosen], self.observations.pixel[chosen]


def _view(data: Mapping[str, Any]) -> View:
    """The view of an export's `views` entry."""
    return View(
        id=int(data['id']),
        camera=Camera.from_json(data['camera']),
        pose=Pose.from_json(data['pose']),
        time=None if data.get('time') is None else float(data['time']),
        width=None if data.get('width') is None else int(data['width']),
        height=None if data.get('height') is None else int(data['height']),
    )


def _observations(landmarks: list[Mapping[str, Any]]) -> Observations:
    """The flat observation arrays of an export's `landmarks` list."""
    rows = [
        (index, o['viewId'], o['pixel'][0], o['pixel'][1])
        for index, point in enumerate(landmarks)
        for o in point['observations']
    ]
    table = np.array(rows, dtype=float).reshape(-1, 4)
    return Observations(
        landmark=table[:, 0].astype(np.int64),
        view=table[:, 1].astype(np.int64),
        pixel=table[:, 2:4].copy(),
    )


def reconstruction_from_json(data: Mapping[str, Any]) -> Reconstruction:
    """The reconstruction of a parsed export or sweep model."""
    landmarks = data['landmarks']
    run_record = data.get('runRecord') or {}
    diagnostics = data.get('diagnostics')
    return Reconstruction(
        views=tuple(_view(v) for v in data['views']),
        xyz=np.array([p['xyz'] for p in landmarks], float).reshape(-1, 3),
        observations=_observations(landmarks),
        format=data.get('format'),
        source=data.get('source'),
        assumed_focal_ratio=data.get('assumedFocalRatio'),
        ratio=data.get('ratio'),
        diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
        run_result=run_record.get('result') or {},
    )


def load_reconstruction(path: Path) -> Reconstruction:
    """The reconstruction of an export file (`typecv-reconstruction.json`) or a sweep model."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('format') not in (None, RECONSTRUCTION_FORMAT):
        raise ValueError(f'{path}: format {data.get("format")!r} is not {RECONSTRUCTION_FORMAT}')
    return reconstruction_from_json(data)
