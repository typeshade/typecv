"""TypeCV against the COLMAP reference.

- `pose_agreement`: the similarity alignment of the two camera sets and what is left after it.
- `depth_spans`: the depth spread of all landmarks in front of chosen views, in each system.
- `reference_blocks`: per-block robust planes of the sparse points each view observes, each in
  its own system's camera frame, so the angles between planes compare without ground truth.
- `common_pixel_relief`: COLMAP's sparse depth at its feature pixels against TypeCV's dense depth
  at the same pixels.

COLMAP is a reference, not ground truth. The numbers are measurements (facts); a reading of a
disagreement as an error of one side is an inference.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .archive import DepthArchive
from .colmap import ColmapModel
from .dense import FILTERED_STAGE
from .geometry import (
    Block,
    PlaneFit,
    align_cameras,
    block_grid,
    direction_angle_deg,
    inside_image,
    plane_angle_deg,
    plane_ray_angle_deg,
    robust_plane,
    rotation_angle_deg,
)
from .relief import (
    COLUMN_THIRDS,
    ROW_THIRDS,
    least_squares_slope,
    robust_scatter,
    theil_sen_slope,
    third_masks,
)
from .scene import Reconstruction


def common_view_ids(reconstruction: Reconstruction, model: ColmapModel) -> list[int]:
    """The ids of the views registered in both systems, ascending."""
    return sorted({v.id for v in reconstruction.views} & set(model.images))


@dataclass(frozen=True)
class PoseAgreement:
    """The camera poses after a similarity alignment of TypeCV onto COLMAP (common views).

    Rotation differences are the angle of R_colmap (R_typecv Q^T)^T per view. Centre differences
    are |aligned TypeCV centre - COLMAP centre| over the largest distance of a COLMAP centre from
    their mean (the reference radius). The turns are the angle between the optical axes of the
    lowest and the highest common view id.
    """

    common_views: int
    scale: float
    rotation_median_deg: float
    rotation_max_deg: float
    center_median: float
    center_max: float
    typecv_turn_deg: float
    colmap_turn_deg: float


def pose_agreement(reconstruction: Reconstruction, model: ColmapModel) -> PoseAgreement:
    """The agreement of the two systems' camera poses."""
    ids = common_view_ids(reconstruction, model)
    typecv = [reconstruction.view(i).pose for i in ids]
    colmap = [model.images[i].pose for i in ids]
    typecv_centers = np.array([p.center for p in typecv])
    colmap_centers = np.array([p.center for p in colmap])
    similarity = align_cameras([p.R for p in typecv], typecv_centers, [p.R for p in colmap], colmap_centers)
    radius = np.max(np.linalg.norm(colmap_centers - similarity.reference_mean, axis=1))
    rotation = [rotation_angle_deg(c.R, t.R @ similarity.rotation.T) for c, t in zip(colmap, typecv, strict=True)]
    center = np.linalg.norm(similarity.apply(typecv_centers) - colmap_centers, axis=1) / radius
    return PoseAgreement(
        common_views=len(ids),
        scale=similarity.scale,
        rotation_median_deg=float(np.median(rotation)),
        rotation_max_deg=float(np.max(rotation)),
        center_median=float(np.median(center)),
        center_max=float(np.max(center)),
        typecv_turn_deg=float(direction_angle_deg(typecv[0].forward, typecv[-1].forward)),
        colmap_turn_deg=float(direction_angle_deg(colmap[0].forward, colmap[-1].forward)),
    )


@dataclass(frozen=True)
class DepthSpan:
    """The 95th over the 5th percentile of the depths of all landmarks in front of a view."""

    view_id: int
    typecv: float
    colmap: float


def _span(depths: np.ndarray) -> float:
    front = depths[depths > 0]
    p = np.percentile(front, [5, 50, 95])
    return float(p[2] / p[0])


def depth_spans(reconstruction: Reconstruction, model: ColmapModel, view_ids: Sequence[int]) -> list[DepthSpan]:
    """The depth span of every landmark of each system from each of the views."""
    return [
        DepthSpan(
            view_id,
            _span(reconstruction.view(view_id).pose.to_camera(reconstruction.xyz)[:, 2]),
            _span(model.images[view_id].pose.to_camera(model.points)[:, 2]),
        )
        for view_id in view_ids
    ]


@dataclass(frozen=True)
class ReferenceBlock:
    """One block's robust planes of the points each system's view observes (own camera frame).
    The ray angles are those of the plane to the block's central ray in that system."""

    block: Block
    typecv_points: int
    colmap_points: int
    typecv: PlaneFit | None
    colmap: PlaneFit | None
    typecv_ray_deg: float | None
    colmap_ray_deg: float | None


@dataclass(frozen=True)
class ReferencePair:
    """The angle between two blocks' planes in each system, None where a plane is missing."""

    first: tuple[int, int]
    second: tuple[int, int]
    typecv_deg: float | None
    colmap_deg: float | None


@dataclass(frozen=True)
class ReferenceBlocks:
    """The per-block planes of one view in both systems."""

    view_id: int
    typecv_landmarks: int
    colmap_points: int
    blocks: tuple[ReferenceBlock, ...]
    pairs: tuple[ReferencePair, ...]


def reference_blocks(
    reconstruction: Reconstruction,
    model: ColmapModel,
    view_id: int,
    grid: tuple[int, int] = (4, 3),
    pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]] = (((1, 1), (3, 1)),),
) -> ReferenceBlocks:
    """The per-block planes of the points the view observes in TypeCV and in COLMAP.

    Every observed point counts, also one behind the camera. The grid is the TypeCV view's image
    size. Each system assigns its points to blocks by its own pixels and takes the central ray
    with its own camera, as the scratchpad did; COLMAP's pixel centres are at +0.5, so its blocks
    sit half a pixel off TypeCV's (a ray difference of about 0.03 degrees at f = 1255 px).
    """
    view = reconstruction.view(view_id)
    image = model.images[view_id]
    landmarks, typecv_pixels = reconstruction.observed_by(view_id)
    typecv_points = view.pose.to_camera(reconstruction.xyz[landmarks])
    colmap_points = image.pose.to_camera(image.xyz)
    colmap_camera = model.camera_of(view_id).pinhole
    blocks = []
    for block in block_grid(view.width, view.height, *grid):
        in_typecv = block.contains(typecv_pixels)
        in_colmap = block.contains(image.pixels)
        typecv = robust_plane(typecv_points[in_typecv])
        colmap = robust_plane(colmap_points[in_colmap])
        blocks.append(ReferenceBlock(
            block, int(in_typecv.sum()), int(in_colmap.sum()), typecv, colmap,
            plane_ray_angle_deg(typecv, block.ray(view.camera)), plane_ray_angle_deg(colmap, block.ray(colmap_camera)),
        ))
    by_cell = {(b.block.bx, b.block.by): b for b in blocks}
    plane_pairs = tuple(
        ReferencePair(
            a, b,
            plane_angle_deg(by_cell[a].typecv, by_cell[b].typecv),
            plane_angle_deg(by_cell[a].colmap, by_cell[b].colmap),
        )
        for a, b in pairs
    )
    return ReferenceBlocks(view_id, len(typecv_points), len(colmap_points), tuple(blocks), plane_pairs)


@dataclass(frozen=True)
class ThirdRelief:
    """The median depth in one image third over the view's median depth, in each system."""

    name: str
    pixels: int
    colmap: float
    typecv: float


@dataclass(frozen=True)
class CommonPixelRelief:
    """COLMAP's sparse depth (z_c, camera axis) against TypeCV's dense depth (z_t, model units) at
    the same pixels of one view.

    The pixels are COLMAP's feature pixels with a 3D point, rounded half to even after the
    -0.5 shift to TypeCV's pixel convention, where z_c > 0 and TypeCV has a valid depth. The
    spreads are the 95th over the 5th percentile. The slopes are of log z_t against log z_c
    (Theil-Sen and least squares), `residual_scatter` the robust deviation of log z_t from the
    Theil-Sen line. A third appears when it holds more than the minimum number of pixels.
    """

    view_id: int
    stage: str
    pixels: int
    colmap_spread: float
    typecv_spread: float
    theil_sen_slope: float
    least_squares_slope: float
    residual_scatter: float
    rows: tuple[ThirdRelief, ...]
    columns: tuple[ThirdRelief, ...]


def common_pixels(model: ColmapModel, view_id: int, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COLMAP's depths, TypeCV's depths (the dtype of `depth`) and the integer pixels (n, 2) where
    both systems have a positive depth, in COLMAP's points2D order."""
    image = model.images[view_id]
    colmap_depth = image.pose.to_camera(image.xyz)[:, 2]
    u = np.round(image.pixels[:, 0] - 0.5).astype(np.int64)
    v = np.round(image.pixels[:, 1] - 0.5).astype(np.int64)
    inside = inside_image(u, v, depth.shape[1], depth.shape[0])
    typecv_depth = np.zeros(len(u), depth.dtype)
    typecv_depth[inside] = depth[v[inside], u[inside]]
    keep = inside & (typecv_depth > 0) & (colmap_depth > 0)
    return colmap_depth[keep], typecv_depth[keep], np.stack([u[keep], v[keep]], 1)


def _thirds(names: Sequence[str], masks: Sequence[np.ndarray], zc: np.ndarray, zt: np.ndarray,
            mc: float, mt: float, min_pixels: int) -> tuple[ThirdRelief, ...]:
    return tuple(
        ThirdRelief(name, int(m.sum()), float(np.median(zc[m]) / mc), float(np.median(zt[m]) / mt))
        for name, m in zip(names, masks, strict=True)
        if m.sum() > min_pixels
    )


def common_pixel_relief(
    archive: DepthArchive,
    model: ColmapModel,
    view_id: int,
    stage: str = FILTERED_STAGE,
    sample: int = 1500,
    seed: int = 0,
    min_dx: float = 0.05,
    min_third_pixels: int = 20,
) -> CommonPixelRelief:
    """The relief of TypeCV's dense depth against COLMAP's sparse depth on their common pixels."""
    record = archive.map_record(view_id, stage)
    depth = archive.model_depth(record)
    zc, zt, pixels = common_pixels(model, view_id, depth)
    lc, lt = np.log(zc), np.log(zt)
    slope = theil_sen_slope(lc, lt, sample=sample, seed=seed, min_dx=min_dx)
    pc, pt = np.percentile(zc, [5, 50, 95]), np.percentile(zt, [5, 50, 95])
    residual = lt - (slope * lc + np.median(lt - slope * lc))
    height, width = depth.shape
    return CommonPixelRelief(
        view_id=view_id,
        stage=stage,
        pixels=len(zc),
        colmap_spread=float(pc[2] / pc[0]),
        typecv_spread=float(pt[2] / pt[0]),
        theil_sen_slope=slope,
        least_squares_slope=least_squares_slope(lc, lt),
        residual_scatter=robust_scatter(residual),
        rows=_thirds(ROW_THIRDS, third_masks(pixels[:, 1], height), zc, zt, pc[1], pt[1], min_third_pixels),
        columns=_thirds(COLUMN_THIRDS, third_masks(pixels[:, 0], width), zc, zt, pc[1], pt[1], min_third_pixels),
    )
