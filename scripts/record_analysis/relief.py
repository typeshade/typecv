"""Depth-relief statistics.

The spread of depths in a view, the log-log slope of one system's depths against another's at
the same pixels, and the median depth of each image third. A measurement is a fact about the
files. A reading of it is an inference: for example, a log-log slope below 1 means that the
measured relief is compressed relative to the reference (inference), and a negative slope that
the depth order across the image is reversed (inference).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import direction_angle_deg
from .scene import Reconstruction

ROW_THIRDS = ('top third', 'middle', 'bottom third')
COLUMN_THIRDS = ('left third', 'centre', 'right third')


def third_bounds(length: int) -> list[tuple[int, int]]:
    """The pixel ranges [0, a), [a, b), [b, length) with a = round(length / 3) and b =
    round(2 * length / 3): 0/427/853/1280 and 0/240/480/720 for a 1280 x 720 image."""
    a, b = round(length / 3), round(2 * length / 3)
    return [(0, a), (a, b), (b, length)]


def third_masks(coordinates: np.ndarray, length: int) -> list[np.ndarray]:
    """The masks of the coordinates (pixel x or y) in each of the three ranges of `third_bounds`."""
    return [(coordinates >= low) & (coordinates < high) for low, high in third_bounds(length)]


def median_or_nan(values: np.ndarray) -> float:
    """The median of the values, NaN when there are none."""
    return float(np.median(values)) if len(values) else float('nan')


def theil_sen_slope(x: np.ndarray, y: np.ndarray, sample: int = 1500, seed: int = 0, min_dx: float = 0.05) -> float:
    """The Theil-Sen slope of y against x: the median of the pairwise slopes.

    The pairs are all pairs of a random subset of at most `sample` points (numpy's default_rng
    with `seed`, without replacement), and only pairs with |dx| > `min_dx` count.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(len(x), sample), replace=False)
    i, j = np.triu_indices(len(idx), 1)
    dx = x[idx][j] - x[idx][i]
    dy = y[idx][j] - y[idx][i]
    good = np.abs(dx) > min_dx
    return float(np.median(dy[good] / dx[good]))


def least_squares_slope(x: np.ndarray, y: np.ndarray) -> float:
    """The slope of the least-squares line y = a x + b."""
    design = np.vstack([x, np.ones_like(x)]).T
    return float(np.linalg.lstsq(design, y, rcond=None)[0][0])


def robust_scatter(residual: np.ndarray) -> float:
    """1.4826 times the median absolute deviation of the residuals from their median."""
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


@dataclass(frozen=True)
class ViewRelief:
    """The landmark depths of one view of a sparse model (model units, camera axis).

    `rows` and `columns` are the median depth of the landmarks in each image third over the
    median depth of all of them; `spread` is the 95th over the 5th percentile.
    """

    view_id: int
    landmarks: int
    spread: float
    rows: tuple[float, float, float]
    columns: tuple[float, float, float]


@dataclass(frozen=True)
class ModelRelief:
    """The depth relief of chosen views of one sparse model, and its first-to-last camera turn
    (the angle between the optical axes of the lowest and the highest view id)."""

    name: str
    ratio: float
    views: int
    landmarks: int
    turn_deg: float
    relief: tuple[ViewRelief, ...]
    unregistered: tuple[int, ...]


def view_relief(model: Reconstruction, view_id: int, width: int, height: int) -> ViewRelief:
    """The relief of the landmarks that the view observes and that lie in front of it."""
    view = model.view(view_id)
    landmarks, pixels = model.observed_by(view_id)
    z = view.pose.to_camera(model.xyz[landmarks])[:, 2]
    front = z > 0
    z, pixels = z[front], pixels[front]
    median = np.median(z)
    rows = tuple(median_or_nan(z[m]) / median for m in third_masks(pixels[:, 1], height))
    columns = tuple(median_or_nan(z[m]) / median for m in third_masks(pixels[:, 0], width))
    spread = float(np.percentile(z, 95) / np.percentile(z, 5))
    return ViewRelief(view_id, len(z), spread, rows, columns)


def model_relief(
    name: str,
    model: Reconstruction,
    view_ids: Sequence[int],
    default_width: int,
    default_height: int,
) -> ModelRelief:
    """The relief of `view_ids` in a sparse model; the image size falls back to the defaults for
    a model whose views carry none. The ratio is the model's own or fx over the image width."""
    first = model.views[0]
    ratio = model.ratio if model.ratio is not None else first.camera.fx / (first.width or default_width)
    ids = sorted(v.id for v in model.views)
    turn = float(direction_angle_deg(model.view(ids[0]).pose.forward, model.view(ids[-1]).pose.forward))
    relief, unregistered = [], []
    for view_id in view_ids:
        if not model.has_view(view_id):
            unregistered.append(view_id)
            continue
        view = model.view(view_id)
        relief.append(view_relief(model, view_id, view.width or default_width, view.height or default_height))
    return ModelRelief(name, float(ratio), len(model.views), len(model.xyz), turn, tuple(relief), tuple(unregistered))
