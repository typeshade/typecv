"""The COLMAP (pycolmap) reference: the incremental reconstruction of the archived frames and the
loading of a COLMAP model into plain arrays.

The reference is an independent reconstruction, not ground truth. pycolmap is imported inside the
functions that need it, so the rest of the package works without it. COLMAP's pixel convention
puts the centre of the top-left pixel at (0.5, 0.5); TypeCV's puts it at (0, 0).
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import Camera, Pose, direction_angle_deg

_IMAGE_NAME = re.compile(r'^view-(\d+)\.[A-Za-z0-9]+$')


def _pycolmap() -> Any:
    """The pycolmap module, or ImportError that names the requirements file."""
    try:
        import pycolmap
    except ImportError as error:
        raise ImportError('this command needs pycolmap (scripts/requirements-record-analysis.txt)') from error
    return pycolmap


def view_id_from_name(name: str) -> int:
    """The view id of a frame named `view-<id>.<ext>`, as `frames` writes them."""
    match = _IMAGE_NAME.match(Path(name).name)
    if not match:
        raise ValueError(f'image name {name!r} is not view-<id>.<ext>')
    return int(match.group(1))


@dataclass(frozen=True)
class ColmapCamera:
    """A COLMAP camera: its model name and parameters, image size, and pinhole part in COLMAP's
    pixel convention."""

    camera_id: int
    model: str
    params: tuple[float, ...]
    width: int
    height: int
    pinhole: Camera


@dataclass(frozen=True)
class ColmapImage:
    """A registered image: world-to-camera pose and the points2D that have a 3D point.

    `pixels` (n, 2) are in COLMAP's pixel convention, `xyz` (n, 3) the world positions of their
    3D points, both in the image's points2D order.
    """

    view_id: int
    name: str
    camera_id: int
    pose: Pose
    pixels: np.ndarray
    xyz: np.ndarray


@dataclass(frozen=True)
class ColmapModel:
    """A COLMAP model: cameras by id, registered images by view id, and every 3D point (P, 3)."""

    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    points: np.ndarray

    def camera_of(self, view_id: int) -> ColmapCamera:
        """The camera of the view's image."""
        return self.cameras[self.images[view_id].camera_id]


def _camera(camera: Any) -> ColmapCamera:
    pinhole = Camera(
        float(camera.focal_length_x),
        float(camera.focal_length_y),
        float(camera.principal_point_x),
        float(camera.principal_point_y),
    )
    return ColmapCamera(int(camera.camera_id), camera.model.name, tuple(float(p) for p in camera.params),
                        int(camera.width), int(camera.height), pinhole)


def _image(image: Any, points3d: Any) -> ColmapImage:
    pose = image.cam_from_world()
    observed = [p for p in image.points2D if p.has_point3D()]
    pixels = np.array([p.xy for p in observed], float).reshape(-1, 2)
    xyz = np.array([points3d[p.point3D_id].xyz for p in observed], float).reshape(-1, 3)
    return ColmapImage(
        view_id=view_id_from_name(image.name),
        name=image.name,
        camera_id=int(image.camera_id),
        pose=Pose(np.array(pose.rotation.matrix(), float), np.array(pose.translation, float)),
        pixels=pixels,
        xyz=xyz,
    )


def load_model(path: Path) -> ColmapModel:
    """The COLMAP model in a `sparse/<n>` directory, read with pycolmap."""
    pycolmap = _pycolmap()
    reconstruction = pycolmap.Reconstruction(str(path))
    images = [_image(im, reconstruction.points3D) for im in reconstruction.images.values()]
    return ColmapModel(
        cameras={int(cid): _camera(c) for cid, c in reconstruction.cameras.items()},
        images={im.view_id: im for im in images},
        points=np.array([p.xyz for p in reconstruction.points3D.values()], float).reshape(-1, 3),
    )


def _model_summary(model_id: int, reconstruction: Any) -> dict[str, Any]:
    """The summary of one reconstructed model, with the keys of the scratchpad's summary.json."""
    views = []
    for image in reconstruction.images.values():
        pose = image.cam_from_world()
        views.append({'name': image.name, 'R': pose.rotation.matrix().reshape(-1).tolist(),
                      't': pose.translation.tolist()})
    views.sort(key=lambda v: v['name'])
    tracks = [len(p.track.elements) for p in reconstruction.points3D.values()]
    axes = [np.array(v['R']).reshape(3, 3)[2] for v in views]
    turn = float(direction_angle_deg(axes[0], axes[-1])) if len(axes) > 1 else None
    return {
        'modelId': model_id,
        'registered': reconstruction.num_reg_images(),
        'points': reconstruction.num_points3D(),
        'meanReprojPx': float(reconstruction.compute_mean_reprojection_error()),
        'meanTrackLength': float(reconstruction.compute_mean_track_length()),
        'medianTrackLength': float(np.median(tracks)) if tracks else None,
        'cameras': {int(cid): {'model': c.model.name, 'params': [float(x) for x in c.params], 'width': int(c.width),
                               'height': int(c.height)} for cid, c in reconstruction.cameras.items()},
        'viewDirChangeFirstLastDeg': turn,
        'first': views[0]['name'] if views else None,
        'last': views[-1]['name'] if views else None,
        'views': views,
    }


def run_reference(
    images: Path,
    out: Path,
    camera_model: str = 'SIMPLE_RADIAL',
    focal_prior: float = 1.2,
    seed: int = 42,
    threads: int = 8,
    progress: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Run the COLMAP reference on the frames in `images` and write `out`.

    SIFT on the CPU, exhaustive matching, incremental mapping with one shared camera of
    `camera_model`; the focal starts at `focal_prior` x max(width, height) (COLMAP's default
    prior is 1.2) and is refined with the extra parameters, the principal point is held. Writes
    `out/database.db` (replaced), `out/sparse/<n>` and `out/summary.json`, and returns the
    summary. `progress(phase, seconds)` is called after each phase.
    """
    pycolmap = _pycolmap()
    images, out = Path(images), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    database = out / 'database.db'
    if database.exists():
        database.unlink()
    report = progress or (lambda phase, seconds: None)
    start = time.perf_counter()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    reader.default_focal_length_factor = focal_prior
    pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, device=pycolmap.Device.cpu)
    features = time.perf_counter()
    report('features', features - start)
    pycolmap.match_exhaustive(database, device=pycolmap.Device.cpu)
    matching = time.perf_counter()
    report('matching', matching - features)
    options = pycolmap.IncrementalPipelineOptions()
    options.random_seed = seed
    options.num_threads = threads
    options.ba_refine_focal_length = True
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = True
    models = pycolmap.incremental_mapping(database, images, out / 'sparse', options)
    mapping = time.perf_counter()
    report('mapping', mapping - matching)
    summary = {
        'pycolmap': pycolmap.__version__,
        'model': camera_model,
        'prior': focal_prior,
        'elapsedS': mapping - start,
        'models': [_model_summary(model_id, rec) for model_id, rec in models.items()],
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    return summary
