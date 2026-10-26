"""Dense-stage analyses of one run record.

- `seed_agreement`: the MVS depth at the pixels of the sparse landmarks against the landmarks'
  own depth. These pixels are the MVS seeds (src/mvs/gpu/mvs-views.ts).
- `dense_normal_angles`: the angle between each dense point's normal and the rays of the cameras.
- `stage_normals`: one view's depth-surface normals (finite differences) and stored PatchMatch
  normals against the viewing ray.
- `block_planes`: robust planes per image block of the sparse landmarks, the depth map and the
  dense cloud, in the view's camera frame.

Angles between a normal and a ray are sign-free, 0 to 90 degrees: 0 means the surface faces the
camera (fronto-parallel). Reading (inference): a fronto-parallel ('cardboard') bias would show as
a distribution peaked near 0 degrees.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .archive import DepthArchive
from .geometry import (
    Block,
    Camera,
    PlaneFit,
    backproject,
    block_grid,
    block_pair_angles,
    finite_difference_normals,
    inside_image,
    line_angle_deg,
    project,
    robust_plane,
    unit,
)
from .ply import DenseCloud
from .relief import median_or_nan
from .scene import Reconstruction, View

FILTERED_STAGE = 'geometric-filtered'
ANGLE_BINS = (0, 10, 20, 30, 45, 60, 75, 90)
DECADE_BINS = tuple(range(0, 100, 10))


def require_undistorted(camera: Camera, what: str) -> None:
    """Refuse a camera with radial terms. Its observation pixels lie on the original grid, not on
    the undistorted grid of the depth maps; src/mvs/gpu/mvs-views.ts undistorts a seed's pixel with
    k1, k2 first, and this package does not."""
    if camera.distorted:
        raise ValueError(
            f'{what}: the camera has radial terms (k1 {camera.k1}, k2 {camera.k2}); observation '
            'pixels on the original grid do not index the undistorted depth grid'
        )


@dataclass(frozen=True)
class SeedAgreement:
    """Dense depth against sparse depth at the landmarks of one view (camera-axis z, model units).

    `observed` counts the landmarks that the view observes, in front of it, whose rounded pixel
    is inside the map; `without_depth` counts those whose pixel has no valid dense depth. The
    statistics are of dense / sparse - 1 over the rest, NaN when no landmark is left. `within`
    pairs a threshold with the share of |dense / sparse - 1| below it.
    """

    view_id: int
    stage: str
    observed: int
    without_depth: int
    median_relative: float
    abs_relative_p50: float
    abs_relative_p90: float
    within: tuple[tuple[float, float], ...]


def seed_agreement(
    reconstruction: Reconstruction,
    archive: DepthArchive,
    view_id: int,
    stage: str = FILTERED_STAGE,
    thresholds: Sequence[float] = (0.02, 0.10),
) -> SeedAgreement:
    """The agreement of the view's dense depth with its sparse landmarks at their observed pixels.

    Pixels are rounded half to even (numpy's rounding); src/mvs/gpu/mvs-views.ts rounds a seed's pixel
    with Math.round (half up). The two differ only at an exact half pixel.
    """
    view = reconstruction.view(view_id)
    require_undistorted(view.camera, f'view {view_id}')
    record = archive.map_record(view_id, stage)
    depth = archive.model_depth(record)
    landmarks, pixels = reconstruction.observed_by(view_id)
    z = view.pose.to_camera(reconstruction.xyz[landmarks])[:, 2]
    u = np.round(pixels[:, 0]).astype(np.int64)
    v = np.round(pixels[:, 1]).astype(np.int64)
    counted = (z > 0) & inside_image(u, v, record.width, record.height)
    dense = depth[v[counted], u[counted]]
    has = dense > 0
    relative = dense[has] / z[counted][has] - 1
    magnitude = np.abs(relative)
    some = len(magnitude) > 0
    return SeedAgreement(
        view_id=view_id,
        stage=stage,
        observed=int(counted.sum()),
        without_depth=int((~has).sum()),
        median_relative=median_or_nan(relative),
        abs_relative_p50=median_or_nan(magnitude),
        abs_relative_p90=float(np.percentile(magnitude, 90)) if some else float('nan'),
        within=tuple((float(t), float(np.mean(magnitude < t)) if some else float('nan')) for t in thresholds),
    )


@dataclass(frozen=True)
class ViewNormalAngles:
    """The dense points inside one view's frustum and the angle of their normals to its rays."""

    view_id: int
    time: float | None
    points: int
    median_deg: float
    share_below_20: float
    share_above_60: float


@dataclass(frozen=True)
class DenseNormalAngles:
    """The dense cloud's normals against the cameras' rays.

    `views` is in time order. The frustum statistics pool every (point, view) pair with the
    point inside the view. The nearest-camera and mean-forward angles take one angle per point.
    Histograms count 10-degree bins from 0 to 90. Extents are the 1st and 99th percentiles per
    world axis.
    """

    points: int
    normal_length_min: float
    normal_length_max: float
    sparse_extent: tuple[tuple[float, ...], tuple[float, ...]]
    dense_extent: tuple[tuple[float, ...], tuple[float, ...]]
    views: tuple[ViewNormalAngles, ...]
    frustum_median_deg: float
    frustum_histogram: tuple[int, ...]
    nearest_camera_median_deg: float
    nearest_camera_histogram: tuple[int, ...]
    mean_forward_median_deg: float
    mean_forward_histogram: tuple[int, ...]


def frustum_angles(positions: np.ndarray, normals: np.ndarray, view: View) -> np.ndarray:
    """angle(normal, ray from the camera centre) of the points that project inside the view:
    z > 0, 0 <= u < width and 0 <= v < height. `normals` are unit vectors."""
    if view.width is None or view.height is None:
        raise ValueError(f'view {view.id} has no image size')
    u, v, front = project(view.pose.to_camera(positions), view.camera)
    seen = front & inside_image(u, v, view.width, view.height)
    return line_angle_deg(normals[seen], unit(positions[seen] - view.pose.center))


def nearest_camera_angles(
    positions: np.ndarray, normals: np.ndarray, centers: np.ndarray, chunk: int = 65536,
) -> np.ndarray:
    """angle(normal, ray from the nearest camera centre) of every point, in chunks of points."""
    angles = np.empty(len(positions))
    for start in range(0, len(positions), chunk):
        part = positions[start:start + chunk]
        nearest = np.argmin(((part[:, None, :] - centers[None, :, :]) ** 2).sum(-1), axis=1)
        angles[start:start + chunk] = line_angle_deg(normals[start:start + chunk], unit(part - centers[nearest]))
    return angles


def _share(mask: np.ndarray) -> float:
    """The share of True in the mask, NaN for an empty mask."""
    return float(mask.mean()) if len(mask) else float('nan')


def _extent(points: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The 1st and the 99th percentile of the points per axis."""
    low, high = np.percentile(points, [1, 99], axis=0)
    return tuple(float(x) for x in low), tuple(float(x) for x in high)


def _histogram(angles: np.ndarray) -> tuple[int, ...]:
    """The counts of angles in 10-degree bins from 0 to 90."""
    return tuple(int(c) for c in np.histogram(angles, bins=DECADE_BINS)[0])


def dense_normal_angles(reconstruction: Reconstruction, cloud: DenseCloud) -> DenseNormalAngles:
    """The angles of the dense cloud's normals to the rays of the reconstruction's cameras."""
    if cloud.normals is None:
        raise ValueError('the dense PLY has no normals')
    lengths = np.linalg.norm(cloud.normals, axis=1)
    normals = unit(cloud.normals, eps=1e-9)
    rows, pooled = [], []
    for view in reconstruction.time_order():
        angles = frustum_angles(cloud.positions, normals, view)
        pooled.append(angles)
        rows.append(ViewNormalAngles(
            view.id, view.time, len(angles), median_or_nan(angles), _share(angles < 20), _share(angles > 60),
        ))
    pooled_angles = np.concatenate(pooled)
    centers = np.array([v.pose.center for v in reconstruction.views])
    nearest = nearest_camera_angles(cloud.positions, normals, centers)
    forward = np.array([v.pose.forward for v in reconstruction.views]).mean(0)
    forward_angles = line_angle_deg(normals, forward / np.linalg.norm(forward))
    return DenseNormalAngles(
        points=len(cloud),
        normal_length_min=float(lengths.min()),
        normal_length_max=float(lengths.max()),
        sparse_extent=_extent(reconstruction.xyz),
        dense_extent=_extent(cloud.positions),
        views=tuple(rows),
        frustum_median_deg=float(np.median(pooled_angles)),
        frustum_histogram=_histogram(pooled_angles),
        nearest_camera_median_deg=float(np.median(nearest)),
        nearest_camera_histogram=_histogram(nearest),
        mean_forward_median_deg=float(np.median(forward_angles)),
        mean_forward_histogram=_histogram(forward_angles),
    )


@dataclass(frozen=True)
class AngleSummary:
    """The median of angles over a pixel mask, and the share of the mask in each ANGLE_BINS bin."""

    median_deg: float
    shares: tuple[float, ...]


def angle_summary(angles: np.ndarray, mask: np.ndarray) -> AngleSummary:
    """The summary of `angles` (degrees) over the pixels of `mask`."""
    selected = angles[mask]
    shares = np.histogram(selected, bins=ANGLE_BINS)[0] / mask.sum()
    return AngleSummary(float(np.median(selected)), tuple(float(s) for s in shares))


@dataclass(frozen=True)
class StageNormals:
    """One view's map of one stage: validity, depth, and normal angles to the viewing ray.

    `surface_raw` is the finite-difference normal of the raw depth at `raw_step` pixels;
    `surface_filtered` that of the median-filtered depth at `filtered_step` pixels
    (`filtered_surface_normals`). `stored` is the archived PatchMatch normal over the mask of
    `surface_raw`, and `stored_vs_surface_median_deg` its median angle to the raw surface normal.
    Depth percentiles are in the MVS normalized frame.
    """

    view_id: int
    stage: str
    valid_share: float
    depth_percentiles: tuple[float, float, float]
    depth_scale_to_model: float
    source_view_ids: tuple[int, ...]
    raw_step: int
    filtered_step: int
    median_size: int
    surface_raw: AngleSummary
    surface_filtered: AngleSummary
    stored: AngleSummary | None
    stored_vs_surface_median_deg: float | None


@dataclass(frozen=True)
class StageImages:
    """8-bit renderings of one map: inverse depth (2nd to 98th percentile, invalid black), and the
    angles of the stored and the filtered surface normals to the ray (0 to 90 degrees as 0 to
    255, outside the filtered mask black)."""

    inverse_depth: np.ndarray
    stored_angle: np.ndarray | None
    surface_angle: np.ndarray


def _angle_image(angles: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (np.where(mask, angles / 90, 0) * 255).astype(np.uint8)


def _inverse_depth_image(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    inverse = np.where(valid, 1 / np.maximum(depth, 1e-6), 0)
    low, high = np.percentile(inverse[valid], [2, 98])
    image = np.clip((inverse - low) / (high - low), 0, 1)
    image[~valid] = 0
    return (image * 255).astype(np.uint8)


def filtered_surface_normals(
    depth: np.ndarray, valid: np.ndarray, camera: Camera, step: int, size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The finite-difference normals (`finite_difference_normals`, at `step` pixels) of the depth
    after a `size` x `size` median filter, with invalid pixels as zero, and their mask.

    The mask keeps a valid pixel whose filtered depth is not zero, at the pixel and at its four
    neighbours: a median over a window that is mostly invalid is zero, and a zero depth would put
    the back-projected point at the camera centre.
    """
    filtered = ndimage.median_filter(np.where(valid, depth, 0), size=size)
    return finite_difference_normals(backproject(filtered, camera), valid & (filtered > 0), step)


def stage_normals(
    archive: DepthArchive,
    view_id: int,
    stage: str,
    raw_step: int = 2,
    filtered_step: int = 6,
    median_size: int = 5,
    render: bool = False,
) -> tuple[StageNormals, StageImages | None]:
    """The normal angles of one map, on its own camera; the renderings when `render` is set."""
    record = archive.map_record(view_id, stage)
    depth = archive.depth(record)
    valid = np.isfinite(depth) & (depth > 0)
    points = backproject(depth, record.camera)
    rays = unit(points)
    raw_normals, raw_mask = finite_difference_normals(points, valid, raw_step)
    filtered_normals, filtered_mask = filtered_surface_normals(depth, valid, record.camera, filtered_step, median_size)
    filtered_angle = line_angle_deg(filtered_normals, rays)
    stored = archive.normals(record)
    stored_angle = None if stored is None else line_angle_deg(stored, rays)
    summary = StageNormals(
        view_id=view_id,
        stage=stage,
        valid_share=float(valid.mean()),
        depth_percentiles=tuple(float(p) for p in np.percentile(depth[valid], [5, 50, 95])),
        depth_scale_to_model=record.depth_scale_to_model,
        source_view_ids=record.source_view_ids,
        raw_step=raw_step,
        filtered_step=filtered_step,
        median_size=median_size,
        surface_raw=angle_summary(line_angle_deg(raw_normals, rays), raw_mask),
        surface_filtered=angle_summary(filtered_angle, filtered_mask),
        stored=None if stored_angle is None else angle_summary(stored_angle, raw_mask),
        stored_vs_surface_median_deg=None if stored is None else float(
            np.median(line_angle_deg(raw_normals, stored)[raw_mask])),
    )
    if not render:
        return summary, None
    images = StageImages(
        inverse_depth=_inverse_depth_image(depth, valid),
        stored_angle=None if stored_angle is None else _angle_image(stored_angle, filtered_mask),
        surface_angle=_angle_image(filtered_angle, filtered_mask),
    )
    return summary, images


@dataclass(frozen=True)
class BlockPlanes:
    """The robust planes of one image block in the view's camera frame.

    `sparse` fits the landmarks observed in the block, `depth` every `depth_stride`-th valid
    pixel of the depth map, `ply` the dense points consistent with the depth map. The counts are
    those before subsampling. `ray` is the unit ray through the block's centre.
    """

    block: Block
    ray: np.ndarray
    sparse_points: int
    depth_points: int
    ply_points: int
    sparse: PlaneFit | None
    depth: PlaneFit | None
    ply: PlaneFit | None


@dataclass(frozen=True)
class PlanePair:
    """The angle between the planes of two blocks per source, None where a plane is missing."""

    first: tuple[int, int]
    second: tuple[int, int]
    sparse_deg: float | None
    depth_deg: float | None
    ply_deg: float | None


@dataclass(frozen=True)
class BlockComparison:
    """The per-block planes of one view. `ply_in_view` counts the dense points that project
    inside the image (one pixel from the right and bottom borders), `ply_consistent` those whose
    depth is within `consistency` (relative) of the depth map at the rounded pixel."""

    view_id: int
    stage: str
    grid: tuple[int, int]
    depth_stride: int
    consistency: float
    sparse_landmarks: int
    ply_in_view: int
    ply_consistent: int
    blocks: tuple[BlockPlanes, ...]
    pairs: tuple[PlanePair, ...]


def consistent_ply_points(
    cloud: DenseCloud, view: View, depth: np.ndarray, consistency: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """The dense points in the view's camera frame whose depth agrees with the depth map (model
    units) at their rounded pixel within `consistency`, their pixels, and the number of points
    that project inside the image."""
    points = view.pose.to_camera(cloud.positions)
    u, v, seen = project(points, view.camera)
    seen &= inside_image(u, v, depth.shape[1], depth.shape[0], margin=1)
    d = depth[np.round(v[seen]).astype(np.int64), np.round(u[seen]).astype(np.int64)]
    with np.errstate(divide='ignore', invalid='ignore'):
        keep = (d > 0) & (np.abs(points[seen][:, 2] / d - 1) < consistency)
    pixels = np.stack([u[seen][keep], v[seen][keep]], 1)
    return points[seen][keep], pixels, int(seen.sum())


def block_planes(
    reconstruction: Reconstruction,
    archive: DepthArchive,
    cloud: DenseCloud,
    view_id: int,
    stage: str = FILTERED_STAGE,
    grid: tuple[int, int] = (4, 3),
    consistency: float = 0.05,
    depth_stride: int = 7,
    pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]] = (((1, 1), (3, 1)),),
) -> BlockComparison:
    """The per-block planes of the view, on the export's camera and pose."""
    view = reconstruction.view(view_id)
    require_undistorted(view.camera, f'view {view_id}')
    record = archive.map_record(view_id, stage)
    if (record.width, record.height) != (view.width, view.height):
        raise ValueError(f'view {view_id}: map {record.width} x {record.height}, view {view.width} x {view.height}')
    depth = archive.model_depth(record)
    landmarks, pixels = reconstruction.observed_by(view_id)
    sparse = view.pose.to_camera(reconstruction.xyz[landmarks])
    front = sparse[:, 2] > 0
    sparse, sparse_pixels = sparse[front], pixels[front]
    dense, dense_pixels, ply_in_view = consistent_ply_points(cloud, view, depth, consistency)
    depth_points = backproject(depth, view.camera)
    blocks = []
    for block in block_grid(view.width, view.height, *grid):
        in_sparse = block.contains(sparse_pixels)
        in_dense = block.contains(dense_pixels)
        tile = depth[block.y0:block.y1, block.x0:block.x1] > 0
        samples = depth_points[block.y0:block.y1, block.x0:block.x1][tile][::depth_stride]
        blocks.append(BlockPlanes(
            block=block,
            ray=block.ray(view.camera),
            sparse_points=int(in_sparse.sum()),
            depth_points=int(tile.sum()),
            ply_points=int(in_dense.sum()),
            sparse=robust_plane(sparse[in_sparse]),
            depth=robust_plane(samples),
            ply=robust_plane(dense[in_dense]),
        ))
    plane_pairs = tuple(
        PlanePair(a, b, *angles)
        for a, b, angles in block_pair_angles(blocks, lambda r: r.block, lambda r: (r.sparse, r.depth, r.ply), pairs)
    )
    return BlockComparison(
        view_id=view_id,
        stage=stage,
        grid=tuple(grid),
        depth_stride=depth_stride,
        consistency=consistency,
        sparse_landmarks=len(sparse),
        ply_in_view=ply_in_view,
        ply_consistent=len(dense),
        blocks=tuple(blocks),
        pairs=plane_pairs,
    )
