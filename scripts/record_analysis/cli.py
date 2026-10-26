"""The command line of the record analysis: one subcommand per analysis.

Every subcommand prints a human-readable table. With `--json OUT.json` it also writes the
numbers, the arguments and the tool versions. The library modules compute; this module parses,
prints and writes.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .archive import STAGES, DepthArchive, extract_parts, inside, load_manifest, verify_extraction

DEFAULT_PAIR = ((1, 1), (3, 1))


def _ids(text: str) -> list[int]:
    """View ids from '0,62,120'."""
    return [int(part) for part in text.split(',') if part.strip()]


def _grid(text: str) -> tuple[int, int]:
    """A block grid from '4x3' (columns x rows)."""
    nx, ny = (int(part) for part in text.lower().split('x'))
    return nx, ny


def _size(text: str) -> tuple[int, int]:
    """An image size from '1280x720' (width x height)."""
    return _grid(text)


def _cell(text: str) -> tuple[int, int]:
    """A block from '1,1' (bx, by)."""
    bx, by = (int(part) for part in text.split(','))
    return bx, by


def jsonable(value: Any) -> Any:
    """The value with dataclasses, numpy values, tuples and paths as JSON types; NaN as null."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)
                if f.repr and not _is_image(getattr(value, f.name))}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _is_image(value: Any) -> bool:
    """True for a PIL image, which the JSON output leaves out."""
    return type(value).__module__.startswith('PIL.')


def _versions() -> dict[str, str]:
    """The versions of Python and of the libraries the numbers depend on."""
    import PIL
    import scipy

    versions = {'python': platform.python_version(), 'numpy': np.__version__,
                'scipy': scipy.__version__, 'pillow': PIL.__version__}
    pycolmap = getattr(sys.modules.get('pycolmap'), '__version__', None)
    if pycolmap:
        versions['pycolmap'] = pycolmap
    return versions


def _f(value: float | None, digits: int = 1, suffix: str = '') -> str:
    """A number with `digits` decimals, '-' for None or NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '-'
    return f'{value:.{digits}f}{suffix}'


def _n(value: int) -> str:
    """An integer with thousands separators."""
    return f'{value:,}'


def table(headers: Sequence[str], rows: Iterable[Sequence[str]], indent: str = '') -> str:
    """The rows under the headers in right-aligned columns separated by ' | '."""
    rows = [list(map(str, row)) for row in rows]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    lines = [' | '.join(h.rjust(w) for h, w in zip(headers, widths, strict=True))]
    lines.append('-+-'.join('-' * w for w in widths))
    lines += [' | '.join(c.rjust(w) for c, w in zip(row, widths, strict=True)) for row in rows]
    return '\n'.join(indent + line for line in lines)


def _counts(histogram: Mapping[int, int]) -> str:
    """A histogram as 'key: count, ...'."""
    return ', '.join(f'{k}: {v}' for k, v in histogram.items())


# extract ----------------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> tuple[Any, int]:
    """extract: the parts into one directory; with a manifest, its parts and a verification."""
    parts = list(args.parts)
    manifest = None
    if args.manifest:
        manifest = load_manifest(args.manifest)
        folder = args.parts_dir or args.manifest.parent
        named = [inside(folder, name) for name in manifest.archives]
        absent = [p for p in named if not p.is_file()]
        if absent:
            raise SystemExit(f'missing parts: {", ".join(str(p) for p in absent)}')
        parts = named + parts
    if not parts:
        raise SystemExit('nothing to extract: give --manifest or parts')
    summary = extract_parts(parts, args.out)
    print(f'extracted {summary.parts} parts into {args.out}: {_n(summary.files)} files, {_n(summary.bytes)} bytes')
    result: dict[str, Any] = {'extraction': summary}
    status = 0
    if manifest:
        problems = verify_extraction(manifest, args.out)
        print(f'manifest: outcome {manifest.outcome}, complete {manifest.complete}, error {manifest.error}, '
              f'{len(manifest.expected_views)} views x {len(manifest.expected_stages)} stages, '
              f'{len(manifest.missing)} missing maps, {len(manifest.maps)} maps listed')
        print(f'verification: {len(problems)} problems' if problems else
              'verification: every listed map and array segment is present with its size')
        for problem in problems:
            print('  ' + problem)
        result.update(outcome=manifest.outcome, complete=manifest.complete, error=manifest.error,
                      missing=manifest.missing, maps=len(manifest.maps), problems=problems)
        status = 1 if problems else 0
    return result, status


# frames -----------------------------------------------------------------------------------


def cmd_frames(args: argparse.Namespace) -> tuple[Any, int]:
    """frames: the archived analysis frames as JPEG and PNG files."""
    from .frames import write_frames

    summary = write_frames(DepthArchive(args.archive), args.out, args.views, args.png_views or (), args.quality)
    print(f'frames written: {len(summary.jpeg)} JPEG (quality {args.quality}), {len(summary.png)} PNG, into {args.out}')
    if summary.first:
        first = summary.first
        camera = 'none' if first.camera is None else ', '.join(f'{k} {v:g}' for k, v in vars(first.camera).items())
        print(f'input of view {first.view_id}: {first.width} x {first.height}, grid {first.grid}, camera {camera}')
    return summary, 0


# sparse -----------------------------------------------------------------------------------


def cmd_sparse(args: argparse.Namespace) -> tuple[Any, int]:
    """sparse: the statistics of the reconstruction export."""
    from .scene import load_reconstruction
    from .sparse import sparse_summary

    s = sparse_summary(load_reconstruction(args.export), args.min_shared, tuple(args.time_window))
    camera = ', '.join(f'{k} {v:g}' for k, v in s.first_camera.items())
    print(f'format {s.format} | source {s.source} | assumed focal ratio {s.assumed_focal_ratio}')
    print(f'views {s.views} | landmarks {_n(s.landmarks)} | observations {_n(s.observations)} | '
          f'image {s.width} x {s.height}')
    print(f'camera of the first view: {camera}')
    print(f'fx min {s.fx_min:.3f} max {s.fx_max:.3f} | ratio to the image width {_f(s.focal_ratio, 4)}')
    if s.focal_search:
        scalars = {k: v for k, v in s.focal_search.items() if not isinstance(v, (list, dict))}
        print('focal search: ' + ', '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}'
                                           for k, v in scalars.items()))
        candidates = s.focal_search.get('candidates') or []
        if candidates:
            rows = [[_f(c.get('ratio'), 4), c.get('views'), _f(c.get('rmsePx'), 4), c.get('observations'),
                     c.get('score')] for c in candidates]
            print(table(['ratio', 'views', 'rmse px', 'observations', 'score'], rows, '  '))
        for refinement in s.focal_search.get('refinements') or []:
            print(f'  refinement: {json.dumps(refinement)}')
    for key, value in s.diagnostics.items():
        print(f'diagnostics {key}: {json.dumps(value)[:500]}')
    print('view ids (time s): ' + ', '.join(f'{i} ({_f(t, 2)})' for i, t in s.view_times))
    print(f'view direction change first to last: {s.turn_deg:.1f} deg '
          f'(largest from the first view: {s.max_turn_from_first_deg:.1f} deg)')
    print(f'camera path: extent (max radius) {s.path_extent:.3f}, mean step {s.mean_step:.3f}')
    p1, p50, p99 = s.first_view_depth
    print(f'landmark depth from the first view: p1 {p1:.3f} p50 {p50:.3f} p99 {p99:.3f}')
    t = s.track_length
    print(f'observations per landmark: mean {t.mean:.2f} median {t.median:g} p90 {t.p90:g} max {t.max:g}')
    print(f'  landmarks by track length: {_counts(s.track_length_histogram)}')
    span = s.track_span
    print(f'track span in keyframes: median {span.median:g} p90 {span.p90:g} max {span.max:g}')
    print(f'view pairs sharing >= {s.min_shared} landmarks: {s.connected_pairs}')
    print(f'  pairs by keyframe-index gap: {_counts(s.pair_gap_histogram)}')
    rows = [[c.index, c.view_id, _f(c.time, 2), c.connected, c.max_index_gap, ' '.join(map(str, c.neighbours))]
            for c in s.connectivity]
    print(table(['index', 'view', 'time s', 'connected views', 'max index gap',
                 'shared with the 3 views before and after'], rows))
    low, high = s.time_window
    print(f'views between {low:g} and {high:g} s: ' + ', '.join(f'{i} ({t:.2f} s)' for i, t in s.views_in_window))
    print(f'consecutive camera step / median depth from the first view: median {s.baseline_over_depth:.4f}')
    return s, 0


# seeds ------------------------------------------------------------------------------------


def cmd_seeds(args: argparse.Namespace) -> tuple[Any, int]:
    """seeds: dense depth against sparse depth at the landmark pixels, per view."""
    from .dense import seed_agreement
    from .scene import load_reconstruction

    reconstruction = load_reconstruction(args.export)
    archive = DepthArchive(args.archive)
    results = [seed_agreement(reconstruction, archive, v, args.stage, args.within) for v in args.views]
    print(f'dense ({args.stage}) / sparse landmark depth - 1 at the landmark pixels')
    headers = ['view', 'observed', 'no dense depth', 'median', '|rel| p50', '|rel| p90']
    headers += [f'within {t * 100:g}%' for t in args.within]
    rows = []
    for r in results:
        missing = f'{_n(r.without_depth)} ({r.without_depth / r.observed * 100:.0f}%)'
        rows.append([r.view_id, _n(r.observed), missing, _f(r.median_relative * 100, 2, '%'),
                     _f(r.abs_relative_p50 * 100, 2, '%'), _f(r.abs_relative_p90 * 100, 1, '%')]
                    + [_f(share, 2) for _, share in r.within])
    print(table(headers, rows))
    return results, 0


# normals ----------------------------------------------------------------------------------


def _extent(extent: tuple[Sequence[float], Sequence[float]]) -> str:
    """Per-axis low and high values as '[x, y, z] .. [x, y, z]'."""
    low, high = extent
    return '[' + ', '.join(f'{x:.2f}' for x in low) + '] .. [' + ', '.join(f'{x:.2f}' for x in high) + ']'


def cmd_normals(args: argparse.Namespace) -> tuple[Any, int]:
    """normals: the dense PLY normals against the camera rays."""
    from .dense import dense_normal_angles
    from .ply import read_dense_ply
    from .scene import load_reconstruction

    r = dense_normal_angles(load_reconstruction(args.export), read_dense_ply(args.ply))
    print(f'dense points {_n(r.points)} | normal length min {r.normal_length_min:.8f} max {r.normal_length_max:.8f}')
    print(f'extent p1 .. p99 per axis: sparse {_extent(r.sparse_extent)}, dense {_extent(r.dense_extent)}')
    print(f'angle(normal, ray) of the points in each view\'s frustum, every {args.row_step} views in time order:')
    rows = [[v.view_id, _f(v.time, 1), _n(v.points), _f(v.median_deg), _f(v.share_below_20, 2),
             _f(v.share_above_60, 2)] for v in r.views[::args.row_step]]
    print(table(['view', 'time s', 'points', 'median deg', 'share < 20 deg', 'share > 60 deg'], rows))
    print(f'every frustum point/view pair: median {r.frustum_median_deg:.1f} deg; '
          f'10-degree bins from 0: {list(r.frustum_histogram)}')
    print(f'ray to the nearest camera: median {r.nearest_camera_median_deg:.1f} deg; '
          f'bins: {list(r.nearest_camera_histogram)}')
    print(f'mean camera forward axis: median {r.mean_forward_median_deg:.1f} deg; '
          f'bins: {list(r.mean_forward_histogram)}')
    return r, 0


# depth-view -------------------------------------------------------------------------------


def _angle_row(label: str, summary: Any) -> list[str]:
    """A table row: the label, the median and the share per angle bin."""
    return [label, _f(summary.median_deg)] + [_f(x, 3) for x in summary.shares]


def _save_stage_images(out: Path, view_id: int, images: Any, record: Any) -> list[Path]:
    """Write the renderings of one map as PNG and its camera, pose and scale as JSON."""
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, array in (('inverse-depth', images.inverse_depth), ('stored-normal-angle', images.stored_angle),
                        ('surface-normal-angle', images.surface_angle)):
        if array is not None:
            path = out / f'view-{view_id}-{name}.png'
            Image.fromarray(array).save(path)
            written.append(path)
    meta = {k: record.metadata[k] for k in ('camera', 'pose', 'depthScaleToModel', 'sourceViewIds', 'width', 'height')}
    path = out / f'view-{view_id}-meta.json'
    path.write_text(json.dumps(meta))
    written.append(path)
    return written


def cmd_depth_view(args: argparse.Namespace) -> tuple[Any, int]:
    """depth-view: one view's normal angles per stage, and the renderings of one stage."""
    from .dense import ANGLE_BINS, stage_normals

    archive = DepthArchive(args.archive)
    bins = [f'{a}-{b}' for a, b in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:], strict=True)]
    results, missing, written = [], [], []
    for stage in args.stages:
        if not archive.has_map(args.view, stage):
            missing.append(stage)
            print(f'--- view {args.view} {stage}: not in the archive')
            continue
        render = args.out is not None and stage == args.render_stage
        s, images = stage_normals(archive, args.view, stage, args.raw_step, args.filtered_step, args.median_size,
                                  render)
        results.append(s)
        p5, p50, p95 = s.depth_percentiles
        print(f'--- view {s.view_id} {stage}: valid {s.valid_share * 100:.1f}% | depth p5/p50/p95 (normalized) '
              f'{p5:.3f} / {p50:.3f} / {p95:.3f} | scale to model {s.depth_scale_to_model:.3f} | '
              f'sources {", ".join(map(str, s.source_view_ids))}')
        rows = [_angle_row(f'depth surface, step {s.raw_step} px', s.surface_raw),
                _angle_row(f'depth surface, step {s.filtered_step} px, median {s.median_size}', s.surface_filtered)]
        if s.stored:
            rows.append(_angle_row(f'stored PatchMatch normal (step {s.raw_step} mask)', s.stored))
        print(table(['normal vs viewing ray', 'median deg'] + bins, rows, '  '))
        if s.stored_vs_surface_median_deg is not None:
            print(f'  stored normal vs depth-surface normal (step {s.raw_step} px): '
                  f'median {s.stored_vs_surface_median_deg:.1f} deg')
        if images is not None:
            written += _save_stage_images(args.out, args.view, images, archive.map_record(args.view, stage))
    for path in written:
        print(f'wrote {path}')
    return {'stages': results, 'missing': missing, 'written': written}, 0


# blocks -----------------------------------------------------------------------------------


def _fit(plane: Any) -> str:
    """A plane's inlier share and rms over depth, '-' without a plane."""
    return '-' if plane is None else f'{plane.inlier_share:.2f} {plane.rms_over_depth * 100:.1f}%'


def cmd_blocks(args: argparse.Namespace) -> tuple[Any, int]:
    """blocks: per-block planes of the sparse landmarks, the depth map and the dense PLY."""
    from .dense import block_planes
    from .geometry import plane_angle_deg, plane_ray_angle_deg
    from .ply import read_dense_ply
    from .scene import load_reconstruction

    pairs = args.pair or [DEFAULT_PAIR]
    r = block_planes(load_reconstruction(args.export), DepthArchive(args.archive), read_dense_ply(args.ply),
                     args.view, args.stage, args.grid, args.consistency, args.depth_stride, pairs)
    print(f'view {r.view_id} ({r.stage}): sparse {_n(r.sparse_landmarks)} landmarks | dense PLY points in view '
          f'{_n(r.ply_in_view)}, consistent with the depth map within {r.consistency * 100:g}% {_n(r.ply_consistent)}')
    rows = [[f'({b.block.bx},{b.block.by})', b.sparse_points, _n(b.depth_points), _n(b.ply_points),
             _f(plane_ray_angle_deg(b.sparse, b.ray)), _f(plane_ray_angle_deg(b.depth, b.ray)),
             _f(plane_ray_angle_deg(b.ply, b.ray)), _f(plane_angle_deg(b.sparse, b.depth)),
             _f(plane_angle_deg(b.sparse, b.ply)), _fit(b.sparse), _fit(b.depth)]
            for b in r.blocks]
    print('angles in deg; "ray" is the angle of the plane to the block\'s central ray; '
          'fit is inlier share and rms/depth')
    print(table(['block', 'n sparse', 'n depth map', 'n PLY', 'ray sparse', 'ray depth map', 'ray PLY',
                 'sparse vs depth map', 'sparse vs PLY', 'fit sparse', 'fit depth map'], rows))
    for p in r.pairs:
        print(f'angle between the planes of blocks {p.first} and {p.second}: sparse {_f(p.sparse_deg)} deg, '
              f'depth map {_f(p.depth_deg)} deg, dense PLY {_f(p.ply_deg)} deg')
    return r, 0


# colmap-run -------------------------------------------------------------------------------


def cmd_colmap_run(args: argparse.Namespace) -> tuple[Any, int]:
    """colmap-run: the pycolmap reference reconstruction of the frames."""
    from .colmap import run_reference

    summary = run_reference(args.images, args.out, args.camera_model, args.focal_prior, args.seed, args.threads,
                            progress=lambda phase, seconds: print(f'{phase} {seconds:.1f} s', flush=True))
    print(f'pycolmap {summary["pycolmap"]}, {summary["model"]}, focal prior {summary["prior"]}, '
          f'{summary["elapsedS"]:.1f} s, {len(summary["models"])} models; summary in {args.out / "summary.json"}')
    for m in summary['models']:
        cameras = '; '.join(f'{c["model"]} {[round(p, 3) for p in c["params"]]}' for c in m['cameras'].values())
        print(f'model {m["modelId"]}: {m["registered"]} images, {_n(m["points"])} points, mean reprojection '
              f'{m["meanReprojPx"]:.3f} px, track length mean {m["meanTrackLength"]:.2f} '
              f'median {m["medianTrackLength"]}, turn {m["first"]} to {m["last"]} '
              f'{_f(m["viewDirChangeFirstLastDeg"])} deg, camera {cameras}')
    return summary, 0


# colmap-compare ---------------------------------------------------------------------------


def cmd_colmap_compare(args: argparse.Namespace) -> tuple[Any, int]:
    """colmap-compare: pose agreement, depth spans and per-block planes against COLMAP."""
    from .colmap import load_model
    from .reference import common_view_ids, depth_spans, pose_agreement, reference_blocks
    from .scene import load_reconstruction

    reconstruction = load_reconstruction(args.export)
    model = load_model(args.colmap)
    first = common_view_ids(reconstruction, model)[0]
    camera = model.camera_of(first)
    print(f'COLMAP camera: {camera.model} {[round(p, 3) for p in camera.params]} | '
          f'TypeCV fx {reconstruction.view(first).camera.fx:.2f} (view {first})')
    poses = pose_agreement(reconstruction, model)
    print(f'pose alignment over {poses.common_views} common views (scale {poses.scale:.4f}): rotation difference '
          f'median {poses.rotation_median_deg:.2f} deg, max {poses.rotation_max_deg:.2f} deg; centre difference / '
          f'reference radius median {poses.center_median:.3f}, max {poses.center_max:.3f}')
    if poses.scale <= 0:
        print('  note: the alignment scale is not positive, so the centre differences do not measure a fit')
    print(f'view direction change first to last: TypeCV {poses.typecv_turn_deg:.1f} deg, '
          f'COLMAP {poses.colmap_turn_deg:.1f} deg')
    spans = depth_spans(reconstruction, model, args.depth_views)
    print('depth p95/p5 of all landmarks in front of the view:')
    print(table(['view', 'TypeCV', 'COLMAP'], [[s.view_id, _f(s.typecv, 2), _f(s.colmap, 2)] for s in spans], '  '))
    pairs = args.pair or [DEFAULT_PAIR]
    blocks = [reference_blocks(reconstruction, model, v, args.grid, pairs) for v in args.views]
    for b in blocks:
        print(f'--- view {b.view_id}: TypeCV {_n(b.typecv_landmarks)} observed landmarks, '
              f'COLMAP {_n(b.colmap_points)} points; angle of the block plane to its central ray (deg)')
        rows = [[f'({c.block.bx},{c.block.by})', _f(c.typecv_ray_deg), _f(c.colmap_ray_deg), c.typecv_points,
                 c.colmap_points] for c in b.blocks]
        print(table(['block', 'TypeCV', 'COLMAP', 'n TypeCV', 'n COLMAP'], rows, '  '))
        for p in b.pairs:
            print(f'  angle between the planes of blocks {p.first} and {p.second}: TypeCV {_f(p.typecv_deg)} deg, '
                  f'COLMAP {_f(p.colmap_deg)} deg')
    return {'poses': poses, 'depthSpans': spans, 'blocks': blocks}, 0


# relief -----------------------------------------------------------------------------------


def cmd_relief(args: argparse.Namespace) -> tuple[Any, int]:
    """relief: COLMAP's sparse depth against TypeCV's dense depth on the same pixels."""
    from .colmap import load_model
    from .reference import common_pixel_relief

    archive = DepthArchive(args.archive)
    model = load_model(args.colmap)
    results = [common_pixel_relief(archive, model, v, args.stage, args.sample, args.seed, args.min_dx, args.min_third)
               for v in args.views]
    print(f'COLMAP sparse depth at its feature pixels against TypeCV dense depth ({args.stage}) at the same pixels')
    print(table(['view', 'common pixels', 'p95/p5 COLMAP', 'p95/p5 TypeCV', 'slope Theil-Sen', 'slope least squares',
                 'residual scatter'],
                [[r.view_id, _n(r.pixels), _f(r.colmap_spread, 3), _f(r.typecv_spread, 3), _f(r.theil_sen_slope, 3),
                  _f(r.least_squares_slope, 3), _f(r.residual_scatter, 3)] for r in results]))
    print(f'median depth of an image third over the view\'s median depth '
          f'(thirds with more than {args.min_third} pixels):')
    rows = [[r.view_id, t.name, _n(t.pixels), _f(t.colmap, 3), _f(t.typecv, 3)]
            for r in results for t in r.rows + r.columns]
    print(table(['view', 'third', 'pixels', 'COLMAP', 'TypeCV'], rows))
    return results, 0


# sweep-relief -----------------------------------------------------------------------------


def cmd_sweep_relief(args: argparse.Namespace) -> tuple[Any, int]:
    """sweep-relief: the depth relief per image third of each sparse model."""
    from .relief import model_relief
    from .scene import load_reconstruction

    width, height = args.image_size
    results = [model_relief(path.name, load_reconstruction(path), args.views, width, height) for path in args.models]
    for m in results:
        print(f'ratio {m.ratio:.4f} ({m.name}): views {m.views}, landmarks {_n(m.landmarks)}, '
              f'turn {m.turn_deg:.1f} deg')
        rows = [[v.view_id, _n(v.landmarks), _f(v.spread, 2), '/'.join(_f(x, 3) for x in v.rows),
                 '/'.join(_f(x, 3) for x in v.columns)] for v in m.relief]
        rows += [[view_id, 'not registered', '', '', ''] for view_id in m.unregistered]
        print(table(['view', 'landmarks', 'p95/p5', 'rows top/mid/bottom', 'columns left/centre/right'], rows, '  '))
    return results, 0


# figure -----------------------------------------------------------------------------------


def cmd_figure(args: argparse.Namespace) -> tuple[Any, int]:
    """figure: two projections of the aligned TypeCV and COLMAP clouds as one PNG."""
    from .colmap import load_model
    from .figure import render_projections
    from .scene import load_reconstruction

    figure = render_projections(load_reconstruction(args.export), load_model(args.colmap), args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.image.save(args.out)
    print(f'points drawn: TypeCV {_n(figure.typecv_points)}, COLMAP {_n(figure.colmap_points)} | '
          f'alignment scale {figure.scale:.4f}')
    print(f'wrote {args.out}')
    return figure, 0


# parser -----------------------------------------------------------------------------------


def _command(subparsers: Any, name: str, handler: Callable, help_text: str) -> argparse.ArgumentParser:
    """A subcommand parser with the shared --json option."""
    parser = subparsers.add_parser(name, help=help_text, description=help_text)
    parser.add_argument('--json', type=Path, metavar='OUT.json', help='also write the numbers as JSON')
    parser.set_defaults(handler=handler)
    return parser


def _add_pair(parser: argparse.ArgumentParser) -> None:
    """The repeatable --pair option of two blocks."""
    parser.add_argument('--pair', nargs=2, type=_cell, action='append', metavar=('BX,BY', 'BX,BY'),
                        help='two blocks whose plane angle to report; repeatable (default 1,1 3,1)')


def build_parser() -> argparse.ArgumentParser:
    """The argument parser with every subcommand."""
    parser = argparse.ArgumentParser(prog='analyze-record.py',
                                     description='Offline analysis of a TypeCV lab run record.')
    sub = parser.add_subparsers(dest='command', required=True)

    p = _command(sub, 'extract', cmd_extract, 'extract the depth archive parts into one directory, safely')
    p.add_argument('--out', type=Path, required=True, help='target directory')
    p.add_argument('--manifest', type=Path, help='typecv-depths-cNNN.json: extract its parts and verify the result')
    p.add_argument('--parts-dir', type=Path, help="directory of the manifest's parts (default: the manifest's)")
    p.add_argument('parts', type=Path, nargs='*', help='further tar.gz parts')

    p = _command(sub, 'frames', cmd_frames, 'write the archived analysis frames as JPEG (and PNG)')
    p.add_argument('--archive', type=Path, required=True, help='extracted depth archive directory')
    p.add_argument('--out', type=Path, required=True, help='output directory (jpg/ and png/)')
    p.add_argument('--views', type=_ids, help='view ids to write (default: every view with an input)')
    p.add_argument('--png-views', type=_ids, help='view ids to write as PNG too')
    p.add_argument('--quality', type=int, default=95, help='JPEG quality (default 95)')

    p = _command(sub, 'sparse', cmd_sparse, 'statistics of the sparse reconstruction export')
    p.add_argument('--export', type=Path, required=True, help='typecv-reconstruction.json')
    p.add_argument('--min-shared', type=int, default=16, help='landmarks two views must share to count as connected')
    p.add_argument('--time-window', type=float, nargs=2, default=(28.0, 32.0), metavar=('FROM', 'TO'),
                   help='list the views in this time range, s (default 28 32)')

    p = _command(sub, 'seeds', cmd_seeds, 'dense depth at the sparse landmark pixels against the landmark depth')
    p.add_argument('--archive', type=Path, required=True, help='extracted depth archive directory')
    p.add_argument('--export', type=Path, required=True, help='typecv-reconstruction.json')
    p.add_argument('--views', type=_ids, default=[0, 62, 104, 120, 137], help='view ids (default 0,62,104,120,137)')
    p.add_argument('--stage', choices=STAGES, default='geometric-filtered', help='depth map stage')
    p.add_argument('--within', type=float, nargs='+', default=[0.02, 0.10],
                   help='relative thresholds (default 0.02 0.10)')

    p = _command(sub, 'normals', cmd_normals, 'dense PLY normals against the camera rays')
    p.add_argument('--ply', type=Path, required=True, help='typecv-dense.ply')
    p.add_argument('--export', type=Path, required=True, help='typecv-reconstruction.json')
    p.add_argument('--row-step', type=int, default=3, help='print every n-th view (default 3)')

    p = _command(sub, 'depth-view', cmd_depth_view,
                 "one view's depth-surface and stored normals against the viewing ray")
    p.add_argument('--archive', type=Path, required=True, help='extracted depth archive directory')
    p.add_argument('--view', type=int, required=True, help='view id')
    p.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES), help='stages (default all)')
    p.add_argument('--raw-step', type=int, default=2, help='finite-difference step on the raw depth, px (default 2)')
    p.add_argument('--filtered-step', type=int, default=6, help='step on the median-filtered depth, px (default 6)')
    p.add_argument('--median-size', type=int, default=5, help='median filter size, px (default 5)')
    p.add_argument('--out', type=Path, help='write PNG renderings and the map metadata here')
    p.add_argument('--render-stage', choices=STAGES, default='geometric-filtered', help='stage to render')

    p = _command(sub, 'blocks', cmd_blocks, 'per-block planes: sparse landmarks, depth map and dense PLY')
    p.add_argument('--archive', type=Path, required=True, help='extracted depth archive directory')
    p.add_argument('--export', type=Path, required=True, help='typecv-reconstruction.json')
    p.add_argument('--ply', type=Path, required=True, help='typecv-dense.ply')
    p.add_argument('--view', type=int, default=120, help='view id (default 120)')
    p.add_argument('--stage', choices=STAGES, default='geometric-filtered', help='depth map stage')
    p.add_argument('--grid', type=_grid, default=(4, 3), help='blocks across x down (default 4x3)')
    p.add_argument('--consistency', type=float, default=0.05, help='PLY-to-depth-map relative tolerance (default 0.05)')
    p.add_argument('--depth-stride', type=int, default=7, help='use every n-th valid depth pixel (default 7)')
    _add_pair(p)

    p = _command(sub, 'colmap-run', cmd_colmap_run, 'run the pycolmap reference on the frames')
    p.add_argument('--images', type=Path, required=True, help='frame directory (frames --out .../jpg)')
    p.add_argument('--out', type=Path, required=True, help='output directory')
    p.add_argument('--camera-model', default='SIMPLE_RADIAL', help='COLMAP camera model (default SIMPLE_RADIAL)')
    p.add_argument('--focal-prior', type=float, default=1.2, help='focal prior x max(width, height) (default 1.2)')
    p.add_argument('--seed', type=int, default=42, help='mapper random seed (default 42)')
    p.add_argument('--threads', type=int, default=8, help='mapper threads (default 8)')

    p = _command(sub, 'colmap-compare', cmd_colmap_compare, 'poses and per-block planes against the COLMAP reference')
    p.add_argument('--export', type=Path, required=True, help='typecv-reconstruction.json')
    p.add_argument('--colmap', type=Path, required=True, help='COLMAP model directory (sparse/0)')
    p.add_argument('--depth-views', type=_ids, default=[0, 62, 120], help='views for the depth span (default 0,62,120)')
    p.add_argument('--views', type=_ids, default=[120, 137], help='views for the block planes (default 120,137)')
    p.add_argument('--grid', type=_grid, default=(4, 3), help='blocks across x down (default 4x3)')
    _add_pair(p)

    p = _command(sub, 'relief', cmd_relief, 'COLMAP depth against TypeCV dense depth on the same pixels')
    p.add_argument('--archive', type=Path, required=True, help='extracted depth archive directory')
    p.add_argument('--colmap', type=Path, required=True, help='COLMAP model directory (sparse/0)')
    p.add_argument('--views', type=_ids, default=[0, 62, 104, 120, 137], help='view ids (default 0,62,104,120,137)')
    p.add_argument('--stage', choices=STAGES, default='geometric-filtered', help='depth map stage')
    p.add_argument('--sample', type=int, default=1500, help='Theil-Sen subsample size (default 1500)')
    p.add_argument('--seed', type=int, default=0, help='Theil-Sen subsample seed (default 0)')
    p.add_argument('--min-dx', type=float, default=0.05, help='smallest |d log z| of a Theil-Sen pair (default 0.05)')
    p.add_argument('--min-third', type=int, default=20, help='report a third with more pixels than this (default 20)')

    p = _command(sub, 'sweep-relief', cmd_sweep_relief, 'depth relief per image third of sparse models')
    p.add_argument('models', type=Path, nargs='+', help='exports or fixed-focal sweep models')
    p.add_argument('--views', type=_ids, default=[120, 137], help='view ids (default 120,137)')
    p.add_argument('--image-size', type=_size, default=(1280, 720),
                   help='image size of models whose views carry none (default 1280x720)')

    p = _command(sub, 'figure', cmd_figure, 'two projections of the aligned TypeCV and COLMAP clouds')
    p.add_argument('--export', type=Path, required=True, help='typecv-reconstruction.json')
    p.add_argument('--colmap', type=Path, required=True, help='COLMAP model directory (sparse/0)')
    p.add_argument('--out', type=Path, required=True, help='PNG file')
    p.add_argument('--size', type=int, default=900, help='panel size, px (default 900)')
    return parser


def write_json(path: Path, args: argparse.Namespace, result: Any) -> None:
    """Write the result with the command, its arguments and the tool versions."""
    arguments = {k: v for k, v in vars(args).items() if k not in ('handler', 'json', 'command')}
    document = {'command': args.command, 'arguments': arguments, 'versions': _versions(), 'result': result}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(document), indent=1, allow_nan=False) + '\n')


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; the exit status is 0 on success."""
    args = build_parser().parse_args(argv)
    result, status = args.handler(args)
    if args.json:
        write_json(args.json, args, result)
        print(f'wrote {args.json}')
    return status
