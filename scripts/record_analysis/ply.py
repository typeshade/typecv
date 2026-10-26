"""The dense cloud PLY of the lab (`densePly` in website/src/lib/dense-cloud.ts).

`format binary_little_endian 1.0` with one `vertex` element: x, y, z (float), nx, ny, nz (float,
present when every chunk had normals) and red, green, blue (uchar); 27 bytes per point with
normals, 15 without. Positions and normals are in the model frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

_TYPES = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': '<i2', 'int16': '<i2', 'ushort': '<u2', 'uint16': '<u2',
    'int': '<i4', 'int32': '<i4', 'uint': '<u4', 'uint32': '<u4',
    'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
}


@dataclass(frozen=True)
class DenseCloud:
    """Points in the model frame: positions (N, 3) float64, sRGB colors (N, 3) uint8, and unit
    normals (N, 3) float64 or None."""

    positions: np.ndarray
    colors: np.ndarray
    normals: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.positions)


def _vertex_layout(stream: BinaryIO) -> tuple[int, np.dtype]:
    """The vertex count and record dtype of a binary little-endian PLY header; the stream is left
    at the first vertex. The vertex element must be the first element."""
    if stream.readline().strip() != b'ply':
        raise ValueError('not a PLY file')
    count, fields, element = None, [], None
    for raw in iter(stream.readline, b''):
        words = raw.decode('ascii').split()
        if not words or words[0] in ('comment', 'obj_info'):
            continue
        if words[0] == 'end_header':
            if count is None:
                raise ValueError('the PLY has no vertex element')
            return count, np.dtype(fields)
        if words[0] == 'format' and words[1:] != ['binary_little_endian', '1.0']:
            raise ValueError(f'unsupported PLY format {" ".join(words[1:])}')
        if words[0] == 'element':
            element = words[1]
            if element == 'vertex':
                if count is not None or fields:
                    raise ValueError('the vertex element must come first')
                count = int(words[2])
            elif count is None:
                raise ValueError('the vertex element must come first')
        if words[0] == 'property' and element == 'vertex':
            if words[1] == 'list':
                raise ValueError('list properties of vertices are not supported')
            if words[1] not in _TYPES:
                raise ValueError(f'unsupported PLY property type {words[1]}')
            fields.append((words[2], _TYPES[words[1]]))
    raise ValueError('the PLY header has no end_header')


def read_dense_ply(path: Path) -> DenseCloud:
    """The dense cloud of a binary little-endian PLY with x, y, z, optional nx, ny, nz and red,
    green, blue vertex properties."""
    with open(path, 'rb') as stream:
        count, dtype = _vertex_layout(stream)
        data = stream.read(count * dtype.itemsize)
    if len(data) != count * dtype.itemsize:
        raise ValueError(f'{path}: {len(data)} bytes of vertices, the header says {count} vertices')
    points = np.frombuffer(data, dtype=dtype)
    names = set(dtype.names)
    positions = np.stack([points['x'], points['y'], points['z']], 1).astype(np.float64)
    colors = np.stack([points['red'], points['green'], points['blue']], 1).astype(np.uint8)
    normals = None
    if {'nx', 'ny', 'nz'} <= names:
        normals = np.stack([points['nx'], points['ny'], points['nz']], 1).astype(np.float64)
    return DenseCloud(positions, colors, normals)


def write_dense_ply(path: Path, cloud: DenseCloud) -> None:
    """Write the cloud in the layout of `densePly`: float positions, float normals when the cloud
    has them, uchar colors."""
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if cloud.normals is not None:
        fields += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
    fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    records = np.empty(len(cloud), dtype=np.dtype(fields))
    for i, axis in enumerate('xyz'):
        records[axis] = cloud.positions[:, i]
        if cloud.normals is not None:
            records['n' + axis] = cloud.normals[:, i]
    for i, channel in enumerate(('red', 'green', 'blue')):
        records[channel] = cloud.colors[:, i]
    header = ['ply', 'format binary_little_endian 1.0', 'comment typecv dense MVS',
              f'element vertex {len(cloud)}']
    header += [f'property {"float" if t == "<f4" else "uchar"} {name}' for name, t in fields]
    header.append('end_header')
    with open(path, 'wb') as stream:
        stream.write(('\n'.join(header) + '\n').encode('ascii'))
        stream.write(records.tobytes())
