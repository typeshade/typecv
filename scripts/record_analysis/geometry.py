"""Shared geometry of the record analysis.

Cameras and poses, pinhole projection, angles, the robust plane fit, the octahedral normal decode
of the MVS maps, back-projection and finite-difference normals of a depth map, the image block
grid and the similarity alignment of two camera sets.

Frames and units: a pose maps the world (model) frame to the camera frame, x_cam = R x_world + t,
with R row-major. The camera looks along +z, x points right and y down, so image rows grow
downwards. Pixel coordinates are those of the image the camera belongs to. Angles are in degrees.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

# The scale from a median absolute deviation to a normal standard deviation.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class Camera:
    """A pinhole camera in pixels, with the radial terms of the export (zero when undistorted)."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Camera:
        """The camera of a JSON object with fx, fy, cx, cy and optional k1, k2."""
        return cls(
            float(data['fx']),
            float(data['fy']),
            float(data['cx']),
            float(data['cy']),
            float(data.get('k1') or 0.0),
            float(data.get('k2') or 0.0),
        )

    @property
    def distorted(self) -> bool:
        """True when a radial term is not zero."""
        return self.k1 != 0.0 or self.k2 != 0.0


@dataclass(frozen=True)
class Pose:
    """A world-to-camera motion, x_cam = R x_world + t."""

    R: np.ndarray
    t: np.ndarray

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Pose:
        """The pose of a JSON object with a row-major `R` (9 values) and `t` (3 values)."""
        return cls(np.array(data['R'], float).reshape(3, 3), np.array(data['t'], float).reshape(3))

    @property
    def center(self) -> np.ndarray:
        """The camera centre in the world frame, -R^T t."""
        return camera_center(self.R, self.t)

    @property
    def forward(self) -> np.ndarray:
        """The optical axis (camera +z) in the world frame: row 3 of R."""
        return self.R[2]

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        """World points (N, 3) in the camera frame (N, 3)."""
        return points @ self.R.T + self.t


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The camera centre in the world frame of the pose (R, t): -R^T t."""
    return -R.T @ t


def project(points: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole pixels (u, v) of camera-frame points (N, 3), and the mask of points with z > 0.

    The radial terms are not applied. A point with z <= 0 is divided by 1 instead of its z, so
    its pixel is finite but meaningless; the mask excludes it.
    """
    z = points[:, 2]
    front = z > 0
    divisor = np.where(front, z, 1)
    u = camera.fx * points[:, 0] / divisor + camera.cx
    v = camera.fy * points[:, 1] / divisor + camera.cy
    return u, v, front


def inside_image(u: np.ndarray, v: np.ndarray, width: int, height: int, margin: int = 0) -> np.ndarray:
    """The mask of pixels with 0 <= u < width - margin and 0 <= v < height - margin."""
    return (u >= 0) & (u < width - margin) & (v >= 0) & (v < height - margin)


def unit(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """The vectors (..., 3) divided by their length, a length below `eps` counting as `eps`."""
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), eps)


def line_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The angle between unit vectors taken as lines, arccos |a . b|, 0 to 90 degrees."""
    return np.degrees(np.arccos(np.clip(np.abs(np.einsum('...i,...i->...', a, b)), 0, 1)))


def direction_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The angle between two directions of any length, 0 to 180 degrees."""
    cos = np.einsum('...i,...i->...', a, b) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """The angle of the rotation Ra Rb^T, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


@dataclass(frozen=True)
class PlaneFit:
    """A robust plane through `centroid` with the unit `normal`.

    In a camera frame the normal faces the camera (normal z <= 0). `inlier_share` is the share of
    the input points that the final fit kept. `rms_over_depth` is the RMS distance of the kept
    points to the plane over the centroid's z, so it is a relative depth error in a camera frame.
    """

    normal: np.ndarray
    centroid: np.ndarray
    inlier_share: float
    rms_over_depth: float


def _pca_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The centroid and the unit normal (the least singular direction) of the points."""
    centroid = points.mean(0)
    normal = np.linalg.svd(points - centroid, full_matrices=False)[2][2]
    return centroid, normal


def robust_plane(
    points: np.ndarray,
    iterations: int = 6,
    min_points: int = 12,
    inlier_sigmas: float = 2.5,
) -> PlaneFit | None:
    """A PCA plane of the points (N, 3) with iterative trimming, or None below `min_points`.

    Each iteration fits the current inliers, measures the signed distance of every point to that
    plane, and keeps the points within `inlier_sigmas` robust deviations (1.4826 times the median
    absolute deviation of the inliers' distances, plus 1e-12). A point that left can come back.
    The trimming stops early when fewer than `min_points` would remain. The final fit uses the
    last inlier set and orients the normal to z <= 0.
    """
    if len(points) < min_points:
        return None
    keep = np.arange(len(points))
    for _ in range(iterations):
        centroid, normal = _pca_plane(points[keep])
        residual = (points - centroid) @ normal
        mad = np.median(np.abs(residual[keep] - np.median(residual[keep]))) + 1e-12
        candidate = np.where(np.abs(residual) < inlier_sigmas * MAD_TO_SIGMA * mad)[0]
        if len(candidate) < min_points:
            break
        keep = candidate
    centroid, normal = _pca_plane(points[keep])
    if normal[2] > 0:
        normal = -normal
    rms = np.sqrt(np.mean(((points[keep] - centroid) @ normal) ** 2)) / centroid[2]
    return PlaneFit(normal, centroid, len(keep) / len(points), float(rms))


def plane_angle_deg(a: PlaneFit | None, b: PlaneFit | None) -> float | None:
    """The angle between two planes (their normals as lines), None when one is missing."""
    return None if a is None or b is None else float(line_angle_deg(a.normal, b.normal))


def plane_ray_angle_deg(plane: PlaneFit | None, ray: np.ndarray) -> float | None:
    """The angle between a plane's normal and a unit ray (as lines), None without a plane.

    0 means the plane faces along the ray (fronto-parallel); 90 means the ray grazes it.
    """
    return None if plane is None else float(line_angle_deg(plane.normal, ray))


def decode_octahedral(words: np.ndarray) -> np.ndarray:
    """Unit vectors (..., 3) of octahedral snorm16x2 words, as `decodeMvsNormal` in src/gpu/mvs.ts.

    x comes from the low 16 bits and y from the high 16 bits, each a signed 16-bit value v read
    as max(v / 32767, -1). The MVS maps store camera-frame normals this way.
    """
    w = np.asarray(words).astype(np.uint32)
    px = np.maximum((w & 0xFFFF).astype(np.uint16).view(np.int16) / 32767.0, -1.0)
    py = np.maximum((w >> 16).astype(np.uint16).view(np.int16) / 32767.0, -1.0)
    z = 1.0 - np.abs(px) - np.abs(py)
    t = np.maximum(-z, 0.0)
    x = px + np.where(px >= 0, -t, t)
    y = py + np.where(py >= 0, -t, t)
    return unit(np.stack([x, y, z], axis=-1))


def backproject(depth: np.ndarray, camera: Camera) -> np.ndarray:
    """Camera-frame points (H, W, 3) of a camera-axis depth map (H, W) on its pixel grid.

    The point of pixel (u, v) is ((u - cx) / fx * d, (v - cy) / fy * d, d), in the depth's units.
    """
    height, width = depth.shape
    v, u = np.mgrid[0:height, 0:width]
    x = (u - camera.cx) / camera.fx * depth
    y = (v - camera.cy) / camera.fy * depth
    return np.stack([x, y, depth], -1)


def finite_difference_normals(points: np.ndarray, valid: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals (H, W, 3) of a back-projected depth surface, and the mask where they hold.

    The normal at a pixel is cross(P(u + step) - P(u - step), P(v + step) - P(v - step)). The
    mask keeps a valid pixel at least `step` pixels from the border whose four neighbours at
    +-step are valid too. Outside the mask the normal is meaningless.
    """
    inner = valid.copy()
    inner[:, :step] = inner[:, -step:] = False
    inner[:step, :] = inner[-step:, :] = False
    mask = (
        inner
        & np.roll(valid, -step, 1)
        & np.roll(valid, step, 1)
        & np.roll(valid, -step, 0)
        & np.roll(valid, step, 0)
    )
    du = np.roll(points, -step, 1) - np.roll(points, step, 1)
    dv = np.roll(points, -step, 0) - np.roll(points, step, 0)
    return unit(np.cross(du, dv)), mask


@dataclass(frozen=True)
class Block:
    """One cell (bx, by) of an image grid: pixels x0 <= x < x1 and y0 <= y < y1."""

    bx: int
    by: int
    x0: int
    x1: int
    y0: int
    y1: int

    def contains(self, pixels: np.ndarray) -> np.ndarray:
        """The mask of pixels (N, 2) inside the block."""
        x, y = pixels[:, 0], pixels[:, 1]
        return (x >= self.x0) & (x < self.x1) & (y >= self.y0) & (y < self.y1)

    def ray(self, camera: Camera) -> np.ndarray:
        """The unit camera-frame ray through the block's centre, (x0 + x1) / 2 and (y0 + y1) / 2."""
        ray = np.array([
            ((self.x0 + self.x1) / 2 - camera.cx) / camera.fx,
            ((self.y0 + self.y1) / 2 - camera.cy) / camera.fy,
            1.0,
        ])
        return ray / np.linalg.norm(ray)


def block_grid(width: int, height: int, nx: int, ny: int) -> list[Block]:
    """The nx x ny blocks of a width x height image, row by row from the top left.

    Block (bx, by) spans bx * width // nx to (bx + 1) * width // nx, and the same for y, so the
    blocks tile the image without gaps or overlaps.
    """
    return [
        Block(bx, by, bx * width // nx, (bx + 1) * width // nx, by * height // ny, (by + 1) * height // ny)
        for by in range(ny)
        for bx in range(nx)
    ]


Cell = tuple[int, int]
BlockRecord = TypeVar('BlockRecord')


def block_pair_angles(
    blocks: Sequence[BlockRecord],
    cell: Callable[[BlockRecord], Block],
    planes: Callable[[BlockRecord], Sequence[PlaneFit | None]],
    pairs: Sequence[tuple[Cell, Cell]],
) -> list[tuple[Cell, Cell, tuple[float | None, ...]]]:
    """For each pair of cells (a, b), the angles between the planes of block a and of block b, one
    angle per plane that `planes` gives for a block record (None where a plane is missing)."""
    by_cell = {(cell(record).bx, cell(record).by): record for record in blocks}
    return [
        (a, b, tuple(plane_angle_deg(x, y) for x, y in zip(planes(by_cell[a]), planes(by_cell[b]), strict=True)))
        for a, b in pairs
    ]


@dataclass(frozen=True)
class Similarity:
    """x_ref = scale * (rotation x_src - source_mean) + reference_mean."""

    rotation: np.ndarray
    scale: float
    source_mean: np.ndarray
    reference_mean: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Source-frame points (N, 3) in the reference frame."""
        return self.scale * (points @ self.rotation.T - self.source_mean) + self.reference_mean


def align_cameras(
    source_rotations: Sequence[np.ndarray],
    source_centers: np.ndarray,
    reference_rotations: Sequence[np.ndarray],
    reference_centers: np.ndarray,
) -> Similarity:
    """The similarity that maps a camera set's world frame onto a reference camera set's.

    The rotation Q (reference world <- source world) solves the orthogonal Procrustes problem
    on sum(R_ref^T R_src), with det(Q) = +1, from the orientations alone. Scale and translation
    are the least-squares fit of the rotated source centres to the reference centres; the scale
    can come out negative when the two trajectories disagree. The sets pair up by position.
    """
    m = sum((r.T @ s for r, s in zip(reference_rotations, source_rotations, strict=True)), np.zeros((3, 3)))
    u, _, vt = np.linalg.svd(m)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    rotated = source_centers @ rotation.T
    source_mean, reference_mean = rotated.mean(0), reference_centers.mean(0)
    scale = float(
        np.sum((rotated - source_mean) * (reference_centers - reference_mean))
        / np.sum((rotated - source_mean) ** 2)
    )
    return Similarity(rotation, scale, source_mean, reference_mean)
