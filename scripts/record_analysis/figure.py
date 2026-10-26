"""Two projections of the sparse clouds after the camera alignment: TypeCV's export, mapped into
COLMAP's frame by the similarity of `geometry.align_cameras`, against the COLMAP reference.

The vertical axis of the side view is the mean camera 'down' axis (row 2 of R) of the reference;
the top view looks down that axis. Points farther than 2.5 reference radii from the reference
cameras' mean are left out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .colmap import ColmapModel
from .reference import aligned_cameras
from .scene import Reconstruction

COLMAP_POINT = (70, 110, 220)
TYPECV_POINT = (230, 120, 30)
COLMAP_CAMERA = (0, 0, 160)
TYPECV_CAMERA = (180, 60, 0)


@dataclass(frozen=True)
class Projections:
    """The figure (two panels side by side) and what it shows."""

    image: Image.Image
    typecv_points: int
    colmap_points: int
    scale: float


@dataclass(frozen=True)
class _Scene:
    """The aligned clouds and cameras in COLMAP's world frame, and the view ids of the cameras."""

    typecv: np.ndarray
    colmap: np.ndarray
    typecv_cameras: np.ndarray
    colmap_cameras: np.ndarray
    ids: list[int]


def _panel(scene: _Scene, horizontal: np.ndarray, vertical: np.ndarray, title: str, size: int) -> Image.Image:
    """One panel: the clouds and cameras projected on two world axes, scaled to the 1st to 99th
    percentile of all of them."""
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    everything = np.vstack([scene.colmap, scene.typecv, scene.colmap_cameras, scene.typecv_cameras])
    a, b = everything @ horizontal, everything @ vertical
    low_a, high_a = np.percentile(a, [1, 99])
    low_b, high_b = np.percentile(b, [1, 99])
    span = max(high_a - low_a, high_b - low_b) * 1.1
    center_a, center_b = (low_a + high_a) / 2, (low_b + high_b) / 2

    def pixels(points: np.ndarray) -> np.ndarray:
        return np.stack([(points @ horizontal - center_a) / span * (size - 40) + size / 2,
                         (points @ vertical - center_b) / span * (size - 40) + size / 2], 1)

    for points, color, step in ((scene.colmap, COLMAP_POINT, 2), (scene.typecv, TYPECV_POINT, 1)):
        for x, y in pixels(points)[::step]:
            draw.point((x, y), fill=color)
    for points, color in ((scene.colmap_cameras, COLMAP_CAMERA), (scene.typecv_cameras, TYPECV_CAMERA)):
        for x, y in pixels(points):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    for k in (0, len(scene.ids) - 1):
        x, y = pixels(scene.colmap_cameras)[k]
        draw.text((x + 5, y - 5), f'cam {scene.ids[k]}', fill=COLMAP_CAMERA)
    draw.text((10, 10), title + '   blue: COLMAP reference, orange: TypeCV export (aligned on cameras)', fill='black')
    return image


def render_projections(reconstruction: Reconstruction, model: ColmapModel, size: int = 900) -> Projections:
    """The top and side views of both sparse clouds, `size` x `size` pixels each."""
    aligned = aligned_cameras(reconstruction, model)
    colmap, similarity = aligned.colmap, aligned.similarity
    down = np.mean([p.R[1] for p in colmap], 0)
    down /= np.linalg.norm(down)
    forward = np.mean([p.R[2] for p in colmap], 0)
    forward -= down * np.dot(forward, down)
    forward /= np.linalg.norm(forward)
    side = np.cross(down, forward)
    mean = similarity.reference_mean

    def near(points: np.ndarray) -> np.ndarray:
        return points[np.linalg.norm(points - mean, axis=1) < 2.5 * aligned.radius]

    scene = _Scene(near(similarity.apply(reconstruction.xyz)), near(model.points),
                   similarity.apply(aligned.typecv_centers), aligned.colmap_centers, aligned.ids)
    top = _panel(scene, side, forward, 'top view (looking down the mean camera-down axis)', size)
    side_view = _panel(scene, forward, down, 'side view (horizontal: mean forward, vertical: mean camera down)', size)
    image = Image.fromarray(np.hstack([np.array(top), np.array(side_view)]))
    return Projections(image, len(scene.typecv), len(scene.colmap), similarity.scale)
