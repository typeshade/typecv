"""Small synthetic run records for the tests: a slanted plane seen by two pinhole views.

The plane is z = 5 + 0.3 x + 0.2 y in the world frame. View 0 sits at the origin with the
identity rotation; view 1 is turned 4 degrees about y and moved 0.4 along x. Every file mirrors a
lab export on a 64 x 48 image: the reconstruction JSON, the depth archive (maps with split depth
segments and octahedral normals, inputs, tar.gz parts and a manifest) and the dense PLY.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np

from record_analysis.geometry import Camera, Pose
from record_analysis.ply import DenseCloud, write_dense_ply

WIDTH, HEIGHT = 64, 48
CAMERA = Camera(50.0, 50.0, 31.5, 23.5)
SCALE = 2.0
INVALID_COLUMNS = 8
PLANE = np.array([-0.3, -0.2, 1.0])  # PLANE . X = 5
STAGES = ('photometric', 'geometric-filtered')


def plane_normal() -> np.ndarray:
    """The plane's unit normal in the world frame, facing the cameras (z < 0)."""
    return -PLANE / np.linalg.norm(PLANE)


def rotation_y(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def poses() -> dict[int, Pose]:
    """The two views' world-to-camera poses."""
    turned = rotation_y(4.0)
    center = np.array([0.4, 0.0, 0.0])
    return {0: Pose(np.eye(3), np.zeros(3)), 1: Pose(turned, -turned @ center)}


def pack2x16snorm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """WGSL pack2x16snorm: floor(0.5 + 32767 clamp(v, -1, 1)) per component, x low, y high."""
    def snorm(v: np.ndarray) -> np.ndarray:
        return np.floor(0.5 + 32767 * np.clip(v, -1, 1)).astype(np.int64) & 0xFFFF
    return (snorm(x) | (snorm(y) << 16)).astype(np.uint32)


def encode_octahedral(normals: np.ndarray) -> np.ndarray:
    """The reference encoder, `encodeNormal` of src/mvs/gpu/mvs.shade.ts, for unit vectors (..., 3)."""
    n = np.asarray(normals, float)
    s = np.abs(n).sum(-1)
    px, py = n[..., 0] / s, n[..., 1] / s
    ox = (1 - np.abs(py)) * np.where(px >= 0, 1.0, -1.0)
    oy = (1 - np.abs(px)) * np.where(py >= 0, 1.0, -1.0)
    below = n[..., 2] < 0
    return pack2x16snorm(np.where(below, ox, px), np.where(below, oy, py))


def plane_points(pose: Pose, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The world points of the plane seen at pixels (u, v) of the view, and their camera-axis
    depths (model units)."""
    rays = np.stack([(u - CAMERA.cx) / CAMERA.fx, (v - CAMERA.cy) / CAMERA.fy, np.ones(np.shape(u))], -1)
    directions = rays @ pose.R
    depth = (5 - PLANE @ pose.center) / (directions @ PLANE)
    return pose.center + directions * depth[..., None], depth


def plane_depth(pose: Pose) -> np.ndarray:
    """The camera-axis depth (H, W) of the plane on the view's pixel grid, in model units."""
    v, u = np.mgrid[0:HEIGHT, 0:WIDTH]
    return plane_points(pose, u, v)[1]


def landmarks() -> np.ndarray:
    """World points of the plane under a 13 x 10 grid of view 0's pixels (every 5th from 2)."""
    v, u = np.mgrid[2:HEIGHT:5, 2:WIDTH:5]
    return plane_points(poses()[0], u, v)[0].reshape(-1, 3)


def pixel_of(pose: Pose, point: np.ndarray) -> tuple[float, float]:
    x, y, z = pose.to_camera(point[None])[0]
    return float(CAMERA.fx * x / z + CAMERA.cx), float(CAMERA.fy * y / z + CAMERA.cy)


def export_json() -> dict:
    """The reconstruction export of the scene (views listed out of time order)."""
    views = []
    for view_id, time in ((1, 0.5), (0, 0.0)):
        pose = poses()[view_id]
        views.append({
            'id': view_id, 'time': time, 'width': WIDTH, 'height': HEIGHT,
            'camera': {'fx': CAMERA.fx, 'fy': CAMERA.fy, 'cx': CAMERA.cx, 'cy': CAMERA.cy, 'k1': 0, 'k2': 0},
            'pose': {'R': pose.R.reshape(-1).tolist(), 't': pose.t.tolist()},
        })
    points = []
    for index, xyz in enumerate(landmarks()):
        observations = []
        for view_id, pose in poses().items():
            u, v = pixel_of(pose, xyz)
            if 0 <= u < WIDTH and 0 <= v < HEIGHT:
                observations.append({'viewId': view_id, 'featureId': index, 'pixel': [u, v]})
        points.append({'trackId': index, 'xyz': xyz.tolist(), 'observations': observations})
    return {'format': 'typecv-reconstruction-v1', 'source': 'synthetic plane', 'assumedFocalRatio': 0.8,
            'diagnostics': {}, 'views': views, 'landmarks': points}


def _write(root: Path, path: str, data: bytes) -> dict:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {'path': path, 'bytes': len(data)}


def _segments(root: Path, name: str, data: bytes, pieces: int) -> list[dict]:
    """Write `data` as `pieces` segments `name.0000`, ... and list them in reverse order."""
    size = -(-len(data) // pieces)
    listed = []
    for i in range(pieces):
        entry = _write(root, f'{name}.{i:04d}', data[i * size:(i + 1) * size])
        listed.append({'path': entry['path'], 'offset': i * size, 'bytes': entry['bytes']})
    return listed[::-1]


def write_archive(root: Path) -> list[dict]:
    """Write the extracted depth archive of both views; returns the map records."""
    maps = []
    rng = np.random.default_rng(1)
    for view_id, pose in poses().items():
        depth = (plane_depth(pose) / SCALE).astype(np.float32)
        depth[:, :INVALID_COLUMNS] = 0
        words = np.full(depth.shape, encode_octahedral(pose.R @ plane_normal()), np.uint32)
        for stage in STAGES:
            folder = f'view-{view_id}/{stage}'
            record = {
                'viewId': view_id, 'stage': stage, 'width': WIDTH, 'height': HEIGHT,
                'depthScaleToModel': SCALE,
                'camera': {'fx': CAMERA.fx, 'fy': CAMERA.fy, 'cx': CAMERA.cx, 'cy': CAMERA.cy},
                'sourceViewIds': [1 - view_id], 'pose': {'R': pose.R.reshape(-1).tolist(), 't': pose.t.tolist()},
                'validPixels': int((depth > 0).sum()), 'invalidDepth': 0, 'dtype': 'float32-le',
                'grid': 'undistorted-pinhole', 'depthMeaning': 'camera-axis-z',
                'depth': _segments(root, f'{folder}/depth.f32le', depth.astype('<f4').tobytes(), 2),
                'normal': _segments(root, f'{folder}/normal.u32le', words.astype('<u4').tobytes(), 1),
                'normalDtype': 'uint32-le', 'normalEncoding': 'octahedral-snorm16x2-camera',
            }
            _write(root, f'{folder}/map.json', json.dumps(record).encode())
            maps.append(record)
        rgba = rng.integers(0, 256, (HEIGHT, WIDTH, 4), dtype=np.uint8)
        source = {'width': WIDTH, 'height': HEIGHT,
                  'camera': {'fx': CAMERA.fx, 'fy': CAMERA.fy, 'cx': CAMERA.cx, 'cy': CAMERA.cy, 'k1': 0, 'k2': 0},
                  'dtype': 'rgba8', 'grid': 'original-distorted',
                  'parts': _segments(root, f'view-{view_id}/input.rgba8', rgba.tobytes(), 1)}
        _write(root, f'view-{view_id}/input.json', json.dumps(source).encode())
    return maps


def tar_gz(path: Path, files: dict[str, bytes]) -> None:
    """Write a tar.gz with one regular member per entry."""
    with tarfile.open(path, 'w:gz') as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def write_parts(root: Path, out: Path) -> Path:
    """Pack an extracted archive `root` into two tar.gz parts and a manifest in `out`; returns the
    manifest path."""
    maps = write_archive(root)
    files = sorted(p for p in root.rglob('*') if p.is_file())
    halves = (files[: len(files) // 2], files[len(files) // 2:])
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for i, half in enumerate(halves):
        name = f'typecv-depths-c000-{i:04d}.tar.gz'
        tar_gz(out / name, {p.relative_to(root).as_posix(): p.read_bytes() for p in half})
        names.append(name)
    manifest = {'format': 'typecv-depth-archive-v1', 'outcome': 'completed', 'complete': True, 'error': None,
                'expectedViews': [0, 1], 'expectedStages': list(STAGES), 'missing': [], 'archives': names,
                'maps': maps}
    path = out / 'typecv-depths-c000.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path


def write_ply(path: Path) -> DenseCloud:
    """Write the dense cloud: the plane under view 0's valid pixels, a quarter pixel inside each
    pixel so that no point sits on the image border, with the plane normal."""
    v, u = np.mgrid[0:HEIGHT, INVALID_COLUMNS:WIDTH]
    positions = plane_points(poses()[0], u + 0.25, v + 0.25)[0].reshape(-1, 3)
    normals = np.tile(plane_normal(), (len(positions), 1))
    colors = np.full((len(positions), 3), 128, np.uint8)
    cloud = DenseCloud(positions, colors, normals)
    write_dense_ply(path, cloud)
    return cloud


def write_record(folder: Path) -> dict[str, Path]:
    """Write every file of the synthetic record into `folder`; returns their paths by role."""
    paths = {'export': folder / 'typecv-reconstruction.json', 'ply': folder / 'typecv-dense.ply',
             'archive': folder / 'extracted'}
    paths['export'].write_text(json.dumps(export_json()), encoding='utf-8')
    write_ply(paths['ply'])
    paths['manifest'] = write_parts(paths['archive'], folder / 'parts')
    return paths
