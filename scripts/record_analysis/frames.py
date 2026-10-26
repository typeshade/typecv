"""The archived analysis frames (`view-<id>/input.rgba8.*`) as image files: JPEG for COLMAP
(`jpg/view-<id:03d>.jpg`) and PNG for viewing (`png/view-<id:03d>.png`). The alpha channel is
dropped; the pixels keep the input's original camera and distortion.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .archive import DepthArchive, InputRecord


@dataclass(frozen=True)
class FrameSummary:
    """The frames written and the input record of the first one."""

    jpeg: tuple[Path, ...]
    png: tuple[Path, ...]
    first: InputRecord | None


def write_frames(
    archive: DepthArchive,
    out: Path,
    views: Iterable[int] | None = None,
    png_views: Iterable[int] = (),
    quality: int = 95,
) -> FrameSummary:
    """Write the frames of `views` (every view with an input when None) as JPEG of `quality`, and
    those of `png_views` also as PNG."""
    out = Path(out)
    (out / 'jpg').mkdir(parents=True, exist_ok=True)
    (out / 'png').mkdir(parents=True, exist_ok=True)
    wanted = set(png_views)
    ids = archive.view_ids() if views is None else sorted(views)
    jpeg, png, first = [], [], None
    for view_id in ids:
        if not (archive.root / f'view-{view_id}' / 'input.json').is_file():
            if views is None:
                continue
            raise FileNotFoundError(f'view {view_id} has no archived input')
        record = archive.input_record(view_id)
        first = first or record
        image = Image.fromarray(archive.input_rgba(record)[:, :, :3])
        path = out / 'jpg' / f'view-{view_id:03d}.jpg'
        image.save(path, quality=quality)
        jpeg.append(path)
        if view_id in wanted:
            path = out / 'png' / f'view-{view_id:03d}.png'
            image.save(path)
            png.append(path)
    return FrameSummary(tuple(jpeg), tuple(png), first)
