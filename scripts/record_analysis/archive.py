"""The depth archive `typecv-depth-archive-v1` (website/src/lib/depth-archive.ts).

A run writes a manifest `typecv-depths-cNNN.json` and numbered `typecv-depths-cNNN-NNNN.tar.gz`
parts. Extracted into one directory, each view has `view-<id>/<stage>/map.json` with the map's
metadata and array segment lists, the segments themselves, and `view-<id>/input.json` with the
segments of the original analysis RGBA image. A segment list is concatenated in offset order.
Arrays are little-endian and row-major: float32 depth, confidence and cost, uint32 normal words.

Depth is camera-axis z in the MVS normalized frame on the undistorted pinhole grid of the map's
camera; zero is invalid; depth * depthScaleToModel is in model units. The input image keeps its
original camera and distortion.
"""
from __future__ import annotations

import json
import shutil
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .geometry import Camera, Pose, decode_octahedral

ARCHIVE_FORMAT = 'typecv-depth-archive-v1'
STAGES = ('photometric', 'geometric-unfiltered', 'geometric-filtered', 'planar')
ARRAY_NAMES = ('depth', 'confidence', 'normal', 'cost')
_DTYPES = {'float32-le': '<f4', 'uint32-le': '<u4', 'rgba8': 'u1'}
# The device names of Windows (Python's ntpath.isreserved), which a file name must not use, also
# with an extension: a member `view-1/CON` would open the console there.
_WINDOWS_DEVICES = frozenset(
    {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'}
    | {f'{prefix}{digit}' for prefix in ('COM', 'LPT') for digit in '123456789\u00b9\u00b2\u00b3'}
)
# The characters that a Windows file name must not hold: the control characters and <>:"/\|?*.
_WINDOWS_CHARACTERS = frozenset('<>:"/\\|?*') | frozenset(chr(code) for code in range(32))


class UnsafeMemberError(ValueError):
    """An archive member that extraction refuses: a link, a device, or a path outside the target."""


@dataclass(frozen=True)
class Segment:
    """One stored piece of an array: the file `path` (relative to the root) holds bytes
    [offset, offset + bytes) of the array."""

    path: str
    offset: int
    bytes: int

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Segment:
        return cls(str(data['path']), int(data['offset']), int(data['bytes']))


def _segments(items: Iterable[Mapping[str, Any]]) -> tuple[Segment, ...]:
    return tuple(Segment.from_json(item) for item in items)


@dataclass(frozen=True)
class MapRecord:
    """One view's map of one stage: its grid, camera, pose, scale and array segments."""

    view_id: int
    stage: str
    width: int
    height: int
    camera: Camera
    pose: Pose
    depth_scale_to_model: float
    source_view_ids: tuple[int, ...]
    arrays: Mapping[str, tuple[Segment, ...]]
    dtypes: Mapping[str, str]
    metadata: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> MapRecord:
        arrays = {name: _segments(data[name]) for name in ARRAY_NAMES if name in data}
        dtype = _DTYPES[data.get('dtype', 'float32-le')]
        dtypes = {name: dtype for name in arrays}
        if 'normal' in arrays:
            dtypes['normal'] = _DTYPES[data.get('normalDtype', 'uint32-le')]
        return cls(
            view_id=int(data['viewId']),
            stage=str(data['stage']),
            width=int(data['width']),
            height=int(data['height']),
            camera=Camera.from_json(data['camera']),
            pose=Pose.from_json(data['pose']),
            depth_scale_to_model=float(data['depthScaleToModel']),
            source_view_ids=tuple(int(v) for v in data.get('sourceViewIds') or ()),
            arrays=arrays,
            dtypes=dtypes,
            metadata=data,
        )


@dataclass(frozen=True)
class InputRecord:
    """One view's archived analysis image: size, original camera (may be absent) and segments."""

    view_id: int
    width: int
    height: int
    camera: Camera | None
    dtype: str
    grid: str
    parts: tuple[Segment, ...]


@dataclass(frozen=True)
class Manifest:
    """The manifest `typecv-depths-cNNN.json`: capture outcome, expected maps, parts and maps."""

    outcome: str
    complete: bool
    error: str | None
    expected_views: tuple[int, ...]
    expected_stages: tuple[str, ...]
    missing: tuple[Mapping[str, Any], ...]
    archives: tuple[str, ...]
    maps: tuple[MapRecord, ...]


def load_manifest(path: Path) -> Manifest:
    """The manifest of a depth archive, or ValueError when the format is not the expected one."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('format') != ARCHIVE_FORMAT:
        raise ValueError(f'{path}: format {data.get("format")!r} is not {ARCHIVE_FORMAT}')
    return Manifest(
        outcome=str(data.get('outcome')),
        complete=bool(data.get('complete')),
        error=data.get('error'),
        expected_views=tuple(int(v) for v in data.get('expectedViews') or ()),
        expected_stages=tuple(data.get('expectedStages') or ()),
        missing=tuple(data.get('missing') or ()),
        archives=tuple(data.get('archives') or ()),
        maps=tuple(MapRecord.from_json(m) for m in data.get('maps') or ()),
    )


def windows_reserved(name: str) -> bool:
    """True for a path component that Windows refuses or opens as a device: a device name (CON,
    PRN, AUX, NUL, COM1 to COM9, LPT1 to LPT9, also with an extension such as `nul.txt`), a name
    that ends in a dot or a space, and a name with a control character or one of <>:"/\\|?*."""
    if name[-1:] in ('.', ' ') and name not in ('.', '..'):
        return True
    if _WINDOWS_CHARACTERS.intersection(name):
        return True
    return name.partition('.')[0].rstrip(' ').upper() in _WINDOWS_DEVICES


def inside(root: Path, relative: str) -> Path:
    """The path of `relative` under `root`, or UnsafeMemberError when it could leave `root`.

    Refused: an empty name, an absolute name, a `..` component, a component that Windows
    reserves (`windows_reserved`: a backslash, a colon, a NUL, a device name, a trailing dot or
    space), and a path that resolves outside `root` through a link.
    """
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or not pure.parts
        or any(part == '..' or windows_reserved(part) for part in pure.parts)
    ):
        raise UnsafeMemberError(f'unsafe path {relative!r}')
    base = Path(root).resolve()
    target = base.joinpath(*pure.parts)
    if not target.resolve().is_relative_to(base):
        raise UnsafeMemberError(f'path {relative!r} resolves outside {base}')
    return target


def read_segments(root: Path, segments: Sequence[Segment]) -> bytes:
    """The bytes of an array: its segments in offset order, which must be contiguous from 0."""
    chunks = []
    at = 0
    for segment in sorted(segments, key=lambda s: s.offset):
        if segment.offset != at:
            raise ValueError(f'segment {segment.path} starts at byte {segment.offset}, expected {at}')
        data = inside(root, segment.path).read_bytes()
        if len(data) != segment.bytes:
            raise ValueError(f'segment {segment.path} holds {len(data)} bytes, expected {segment.bytes}')
        chunks.append(data)
        at += segment.bytes
    return b''.join(chunks)


def _grid(data: bytes, dtype: str, shape: tuple[int, ...], name: str) -> np.ndarray:
    """The bytes as a row-major array of `shape`, or ValueError when the size disagrees."""
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(data) != expected:
        raise ValueError(f'{name}: {len(data)} bytes, expected {expected} for shape {shape}')
    return np.frombuffer(data, dtype).reshape(shape)


class DepthArchive:
    """An extracted depth archive directory (the root that holds the `view-<id>` folders)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f'{self.root} is not a directory')

    def view_ids(self) -> list[int]:
        """The ids of the `view-<id>` folders, ascending."""
        ids = []
        for child in self.root.glob('view-*'):
            suffix = child.name[len('view-'):]
            if child.is_dir() and suffix.isdigit():
                ids.append(int(suffix))
        return sorted(ids)

    def has_map(self, view_id: int, stage: str) -> bool:
        """True when the archive holds this view's map of this stage."""
        return (self.root / f'view-{view_id}' / stage / 'map.json').is_file()

    def map_record(self, view_id: int, stage: str) -> MapRecord:
        """The record of `view-<id>/<stage>/map.json`."""
        path = self.root / f'view-{view_id}' / stage / 'map.json'
        return MapRecord.from_json(json.loads(path.read_text(encoding='utf-8')))

    def array(self, record: MapRecord, name: str) -> np.ndarray:
        """The map's array `name` (depth, confidence, normal or cost), shape (height, width)."""
        if name not in record.arrays:
            raise KeyError(f'view {record.view_id} {record.stage} has no {name} array')
        data = read_segments(self.root, record.arrays[name])
        label = f'view {record.view_id} {record.stage} {name}'
        return _grid(data, record.dtypes[name], (record.height, record.width), label)

    def depth(self, record: MapRecord) -> np.ndarray:
        """The float32 depth (H, W) in the MVS normalized frame; zero is invalid."""
        return self.array(record, 'depth')

    def model_depth(self, record: MapRecord) -> np.ndarray:
        """The float32 depth (H, W) in model units: depth * float32(depthScaleToModel)."""
        return self.depth(record) * np.float32(record.depth_scale_to_model)

    def normals(self, record: MapRecord) -> np.ndarray | None:
        """The stored camera-frame unit normals (H, W, 3), or None when the map has none."""
        if 'normal' not in record.arrays:
            return None
        return decode_octahedral(self.array(record, 'normal'))

    def input_record(self, view_id: int) -> InputRecord:
        """The record of `view-<id>/input.json`."""
        data = json.loads((self.root / f'view-{view_id}' / 'input.json').read_text(encoding='utf-8'))
        camera = data.get('camera')
        return InputRecord(
            view_id=view_id,
            width=int(data['width']),
            height=int(data['height']),
            camera=None if camera is None else Camera.from_json(camera),
            dtype=str(data.get('dtype', 'rgba8')),
            grid=str(data.get('grid', '')),
            parts=_segments(data['parts']),
        )

    def input_rgba(self, record: InputRecord) -> np.ndarray:
        """The archived analysis image (H, W, 4), uint8 RGBA."""
        if record.dtype != 'rgba8':
            raise ValueError(f'view {record.view_id}: input dtype {record.dtype!r} is not rgba8')
        data = read_segments(self.root, record.parts)
        return _grid(data, 'u1', (record.height, record.width, 4), f'view {record.view_id} input')


@dataclass(frozen=True)
class ExtractionSummary:
    """What an extraction wrote: the parts read, the files written and their bytes."""

    parts: int
    files: int
    bytes: int


def extract_parts(parts: Iterable[Path], target: Path) -> ExtractionSummary:
    """Extract tar.gz parts into `target`, streaming one member at a time.

    Only regular files and directories are extracted, each at a path that `inside` accepts;
    any other member raises UnsafeMemberError before anything is written for it. Files are
    written with default permissions; an existing file at the same path is replaced.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    count = files = written = 0
    for part in parts:
        count += 1
        with tarfile.open(part, mode='r|gz') as tar:
            for member in tar:
                destination = inside(target, member.name)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise UnsafeMemberError(f'{part}: member {member.name!r} is not a regular file')
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                with open(destination, 'wb') as out:
                    shutil.copyfileobj(source, out)
                files += 1
                written += member.size
    return ExtractionSummary(count, files, written)


def map_differences(listed: MapRecord, extracted: MapRecord) -> list[str]:
    """The fields in which an extracted map.json differs from its manifest entry: viewId, stage,
    size, camera, pose, depthScaleToModel, dtype and segments (the segment lists of the arrays)."""
    same = {
        'viewId': listed.view_id == extracted.view_id,
        'stage': listed.stage == extracted.stage,
        'size': (listed.width, listed.height) == (extracted.width, extracted.height),
        'camera': listed.camera == extracted.camera,
        'pose': bool(np.array_equal(listed.pose.R, extracted.pose.R)
                     and np.array_equal(listed.pose.t, extracted.pose.t)),
        'depthScaleToModel': listed.depth_scale_to_model == extracted.depth_scale_to_model,
        'dtype': dict(listed.dtypes) == dict(extracted.dtypes),
        'segments': dict(listed.arrays) == dict(extracted.arrays),
    }
    return [name for name, equal in same.items() if not equal]


def verify_extraction(manifest: Manifest, root: Path) -> list[str]:
    """The problems of an extracted archive against its manifest; empty when every listed map has
    its map.json, the map.json agrees with the manifest entry (`map_differences`), and every array
    segment is present with its declared size. A map.json of another run, with other poses or
    another scale, is a problem even when its segments have the same sizes."""
    problems = []
    for record in manifest.maps:
        folder = f'view-{record.view_id}/{record.stage}'
        map_path = Path(root) / folder / 'map.json'
        if not map_path.is_file():
            problems.append(f'{folder}/map.json is missing')
        else:
            try:
                extracted = MapRecord.from_json(json.loads(map_path.read_text(encoding='utf-8')))
            except (ValueError, KeyError, TypeError) as error:
                problems.append(f'{folder}/map.json is not a map record: {error!r}')
            else:
                differences = map_differences(record, extracted)
                if differences:
                    problems.append(f'{folder}/map.json differs from the manifest in {", ".join(differences)}')
        for name, segments in record.arrays.items():
            for segment in segments:
                path = inside(root, segment.path)
                if not path.is_file():
                    problems.append(f'{segment.path} ({name}) is missing')
                elif path.stat().st_size != segment.bytes:
                    size = path.stat().st_size
                    problems.append(f'{segment.path} ({name}) holds {size} bytes, expected {segment.bytes}')
    return problems
