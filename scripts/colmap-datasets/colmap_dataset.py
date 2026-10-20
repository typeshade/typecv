"""COLMAP demo datasets as TypeCV benchmarks.

Each dataset ships images, a COLMAP database (SIFT keypoints, descriptors, raw and
inlier matches) and a bundled sparse model. This tool exports the database to the
TypeCV sequence JSON, summarizes the bundled model as a reference, optionally runs
pycolmap's mapper on the same database, and compares a TypeCV model to a reference.

Subcommands:
  fetch      Download one dataset archive, check its SHA-256 and extract it.
  export     Write TypeCV sequence JSON from the dataset's COLMAP database.
  reference  Summarize the bundled sparse model; optionally run pycolmap's mapper.
  compare    Align a TypeCV model to a reference model and report pose agreement.

The bundled model is a reference, not ground truth. Both systems can share
systematic errors from the same features and matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

RELEASE = 'https://github.com/colmap/colmap/releases/download/3.11.1/'
DATASETS = {
    # person-hall and graham-hall are split archives (.z01 ... and .zip): download every part next
    # to each other, and `fetch` reads them as one archive (SplitArchive).
    'south-building': {
        'url': RELEASE + 'south-building.zip',
        'sha256': 'd210016bd2de20936a5f02b87fd38a76bf0440c42d045231218372cf9db9a7a1',
    },
    'gerrard-hall': {'url': RELEASE + 'gerrard-hall.zip', 'sha256': None},
    # Split archives: fetch downloads no part; place person-hall.z01 and person-hall.zip (graham-hall:
    # .z01 to .z05 and .zip) from the release next to each other first.
    'person-hall': {'url': RELEASE + 'person-hall.zip', 'sha256': None, 'parts': 2},
    'graham-hall': {'url': RELEASE + 'graham-hall.zip', 'sha256': None, 'parts': 6},
}
# COLMAP stores pixel centers at +0.5. TypeCV uses integer pixel centers.
HALF = 0.5


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_fetch(a: argparse.Namespace) -> None:
    spec = DATASETS.get(a.name, {'url': RELEASE + f'{a.name}.zip', 'sha256': None})
    a.directory.mkdir(parents=True, exist_ok=True)
    archive = a.directory / f'{a.name}.zip'
    parts = sorted(a.directory.glob(f'{a.name}.z[0-9][0-9]'))
    if spec.get('parts') and len(parts) + 1 != spec['parts']:
        sys.exit(f'{a.name} is a split archive of {spec["parts"]} parts: download them from {RELEASE} into {a.directory}')
    if not archive.exists() and not parts:
        print(f'downloading {spec["url"]}', file=sys.stderr)
        urllib.request.urlretrieve(spec['url'], archive)
    digest = sha256_of(archive)
    if spec['sha256'] and digest != spec['sha256']:
        sys.exit(f'SHA-256 mismatch for {archive}: {digest}')
    print(json.dumps({'archive': str(archive), 'parts': [str(p) for p in parts], 'sha256': digest}))
    target = a.directory / a.name
    if target.exists():
        print(f'{target} exists; skipping extraction', file=sys.stderr)
        return
    # person-hall and graham-hall are split archives, read in place by split_zip.py. With
    # --select N, only the database, the sparse model and the first N images by name are extracted.
    def selected(names: list[str]) -> list[str]:
        if not a.select:
            return names
        images = sorted(n for n in names if '/images/' in n and not n.endswith('/'))[: a.select]
        keep = set(images) | {n for n in names if '/sparse/' in n or n.endswith('database.db')}
        return [n for n in names if n in keep]

    if parts:
        from split_zip import SplitZip  # noqa: PLC0415

        with SplitZip(parts + [archive]) as bundle:
            wanted = set(selected(bundle.names()))
            for entry in bundle.entries:
                if entry.name in wanted:
                    bundle.extract(entry, a.directory)
    else:
        with zipfile.ZipFile(archive) as bundle:
            for name in selected(bundle.namelist()):
                bundle.extract(name, a.directory)


def load_pycolmap():
    try:
        import pycolmap  # noqa: PLC0415
    except ImportError:
        sys.exit('pycolmap is required: pip install -r scripts/requirements-sequence.txt')
    return pycolmap


def typecv_camera(camera, scale: float) -> dict:
    """Convert a COLMAP camera to TypeCV's pinhole + radial k1/k2 model at the analysis scale."""
    model = str(camera.model).split('.')[-1]
    p = [float(v) for v in camera.params]
    if model == 'SIMPLE_PINHOLE':
        f, cx, cy, k1, k2 = p[0], p[1], p[2], 0.0, 0.0
        fx = fy = f
    elif model == 'PINHOLE':
        fx, fy, cx, cy, k1, k2 = p[0], p[1], p[2], p[3], 0.0, 0.0
    elif model == 'SIMPLE_RADIAL':
        fx = fy = p[0]
        cx, cy, k1, k2 = p[1], p[2], p[3], 0.0
    elif model == 'RADIAL':
        fx = fy = p[0]
        cx, cy, k1, k2 = p[1], p[2], p[3], p[4]
    elif model == 'OPENCV':
        # fx, fy, cx, cy, k1, k2, p1, p2. TypeCV has no tangential terms: they are dropped when
        # their largest displacement at the analysis size is under half a pixel (gerrard-hall:
        # p1 1.2e-4, p2 3.0e-4, 0.27 px at 640 wide), and the drop is recorded.
        fx, fy, cx, cy, k1, k2 = p[0], p[1], p[2], p[3], p[4], p[5]
        tangential_px = max_tangential_px(p[6], p[7], fx, fy, cx, cy, camera.width, camera.height) * scale
        if tangential_px > 0.5:
            sys.exit(f'camera model OPENCV with tangential terms that move a point {tangential_px:.2f} px at the analysis size; TypeCV has none')
    else:
        sys.exit(f'camera model {model} is not supported by TypeCV (pinhole + radial k1/k2 only)')
    # Normalized radial distortion is invariant to a uniform image scale.
    return {
        'fx': fx * scale,
        'fy': fy * scale,
        'cx': cx * scale - HALF,
        'cy': cy * scale - HALF,
        'k1': k1,
        'k2': k2,
        'model': model,
        'width': round(camera.width * scale),
        'height': round(camera.height * scale),
        **({'droppedTangential': [p[6], p[7], tangential_px]} if model == 'OPENCV' else {}),
    }


def max_tangential_px(p1: float, p2: float, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> float:
    """The largest displacement, in source pixels, of OpenCV's tangential terms over the image corners."""
    worst = 0.0
    for u, v in ((0, 0), (width, 0), (0, height), (width, height)):
        x, y = (u - cx) / fx, (v - cy) / fy
        r2 = x * x + y * y
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        worst = max(worst, float(np.hypot(dx * fx, dy * fy)))
    return worst


def select_images(images, count: int, step: int):
    ordered = sorted(images, key=lambda image: image.name)
    ordered = ordered[::step]
    if count:
        ordered = ordered[:count]
    return ordered


def cmd_export(a: argparse.Namespace) -> None:
    pycolmap = load_pycolmap()
    database_path = a.dataset / 'database.db'
    db = pycolmap.Database.open(str(database_path))
    cameras = {camera.camera_id: camera for camera in db.read_all_cameras()}
    if a.camera == 'reference':
        reference = pycolmap.Reconstruction(str(a.dataset / 'sparse'))
        cameras = dict(reference.cameras)
    selected = select_images(db.read_all_images(), a.images, a.step)
    if len(selected) < 2:
        sys.exit('at least two images are needed')
    width = max(cameras[image.camera_id].width for image in selected)
    height = max(cameras[image.camera_id].height for image in selected)
    scale = a.max_edge / max(width, height) if a.max_edge else 1.0
    views, by_id = [], {}
    for index, image in enumerate(selected):
        keypoints = np.asarray(db.read_keypoints(image.image_id), dtype=np.float64)[:, :2]
        keypoints = keypoints * scale - HALF
        camera = typecv_camera(cameras[image.camera_id], scale)
        views.append({
            'id': index,
            'name': image.name,
            'imageId': int(image.image_id),
            'camera': {k: camera[k] for k in ('fx', 'fy', 'cx', 'cy', 'k1', 'k2')},
            'keypoints': keypoints.reshape(-1).tolist(),
        })
        by_id[image.image_id] = index
    pairs, skipped = [], 0
    for i, first in enumerate(selected):
        for second in selected[i + 1:]:
            if a.window and by_id[second.image_id] - by_id[first.image_id] > a.window:
                continue
            if a.matches == 'inlier':
                if not db.exists_two_view_geometry(first.image_id, second.image_id):
                    continue
                matches = db.read_two_view_geometry(first.image_id, second.image_id).inlier_matches
            else:
                if not db.exists_matches(first.image_id, second.image_id):
                    continue
                matches = db.read_matches(first.image_id, second.image_id)
            matches = np.asarray(matches, dtype=np.uint32)
            if len(matches) < a.min_matches:
                skipped += 1
                continue
            pairs.append({
                'first': by_id[first.image_id],
                'second': by_id[second.image_id],
                'matches': matches.reshape(-1).tolist(),
            })
    camera_source = cameras[selected[0].camera_id]
    metadata = {
        'dataset': a.dataset.name,
        'databaseSha256': sha256_of(database_path),
        'images': len(views),
        'step': a.step,
        'window': a.window,
        'matches': a.matches,
        'minMatches': a.min_matches,
        'pairs': len(pairs),
        'pairsSkipped': skipped,
        'correspondences': sum(len(pair['matches']) // 2 for pair in pairs),
        'keypoints': sum(len(view['keypoints']) // 2 for view in views),
        'sourceWidth': int(camera_source.width),
        'sourceHeight': int(camera_source.height),
        'scale': scale,
        'cameraSource': a.camera,
        'cameraModel': str(camera_source.model).split('.')[-1],
        'cameraParams': [float(v) for v in camera_source.params],
        'pixelConvention': 'COLMAP +0.5 pixel centers shifted to TypeCV integer centers',
        'features': 'COLMAP SIFT keypoints and matches from the dataset database; TypeCV performs verification, tracks, registration, triangulation and BA',
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'metadata': metadata, 'views': views, 'pairs': pairs}))
    print(json.dumps(metadata, indent=2))


def pose_record(image) -> dict:
    pose = image.cam_from_world()
    return {
        'R': pose.rotation.matrix().reshape(-1).tolist(),
        't': pose.translation.tolist(),
    }


def summarize_model(model, names: set[str] | None) -> dict:
    views = []
    for image in model.images.values():
        if not image.has_pose:
            continue
        if names is not None and image.name not in names:
            continue
        views.append({'name': image.name, 'pose': pose_record(image)})
    views.sort(key=lambda view: view['name'])
    return {
        'registeredViews': len(views),
        'landmarks': model.num_points3D(),
        'observations': model.compute_num_observations(),
        'meanTrackLength': model.compute_mean_track_length(),
        'nativeMeanReprojectionErrorPx': model.compute_mean_reprojection_error(),
        'cameras': [
            {'id': camera.camera_id, 'model': str(camera.model).split('.')[-1],
             'params': [float(v) for v in camera.params]}
            for camera in model.cameras.values()
        ],
        'views': views,
    }


def cmd_reference(a: argparse.Namespace) -> None:
    pycolmap = load_pycolmap()
    names = None
    if a.sequence:
        sequence = json.loads(a.sequence.read_text())
        names = {view['name'] for view in sequence['views']}
    bundled = pycolmap.Reconstruction(str(a.dataset / 'sparse'))
    report = {
        'dataset': a.dataset.name,
        'pycolmapVersion': pycolmap.__version__,
        'bundled': {
            'scope': 'Sparse model shipped with the dataset: COLMAP full pipeline with refined intrinsics. Views are limited to the exported subset when --sequence is given.',
            **summarize_model(bundled, names),
        },
    }
    if a.run_mapper:
        if a.run_mapper.exists() and any(a.run_mapper.iterdir()):
            sys.exit(f'{a.run_mapper} must be empty')
        a.run_mapper.mkdir(parents=True, exist_ok=True)
        database = a.run_mapper / 'database.db'
        shutil.copyfile(a.dataset / 'database.db', database)
        db = pycolmap.Database.open(str(database))
        if a.fixed_camera:
            for camera in bundled.cameras.values():
                db.update_camera(camera)
        db.close()
        options = pycolmap.IncrementalPipelineOptions()
        if names is not None:
            options.image_names = sorted(names)
        if a.fixed_camera:
            options.ba_refine_focal_length = False
            options.ba_refine_principal_point = False
            options.ba_refine_extra_params = False
        options.num_threads = a.threads
        options.random_seed = 42
        started = time.perf_counter()
        models = pycolmap.incremental_mapping(str(database), str(a.dataset / 'images'), str(a.run_mapper), options)
        elapsed = time.perf_counter() - started
        best = max(models.values(), key=lambda model: model.num_reg_images()) if models else None
        report['mapper'] = {
            'scope': 'pycolmap incremental_mapping on the dataset database (same SIFT features and matches). CPU-only wheel unless pycolmap reports CUDA.',
            'fixedCamera': a.fixed_camera,
            'threads': a.threads,
            'hasCuda': bool(getattr(pycolmap, 'has_cuda', False)),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'elapsedSeconds': elapsed,
            'models': len(models),
            **(summarize_model(best, names) if best else {'registeredViews': 0}),
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ('views', 'cameras', 'scope')}
               for k, v in report.items() if isinstance(v, dict)}
    print(json.dumps(summary, indent=2))


def align(source: dict[str, dict], target: dict[str, dict]) -> dict:
    """Similarity-align source camera poses to target poses on their common images."""
    ids = sorted(source.keys() & target.keys())
    if len(ids) < 3:
        return {'commonViews': len(ids), 'status': 'fewer than three common cameras'}
    Rs = np.array([source[i]['R'] for i in ids]).reshape(-1, 3, 3)
    Rr = np.array([target[i]['R'] for i in ids]).reshape(-1, 3, 3)
    xs = np.array([-R.T @ np.array(source[i]['t']) for i, R in zip(ids, Rs)])
    ys = np.array([-R.T @ np.array(target[i]['t']) for i, R in zip(ids, Rr)])
    u, _, vt = np.linalg.svd(sum(b.T @ c for b, c in zip(Rr, Rs)))
    Q = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    x = (Q @ xs.T).T
    cx, cy = x.mean(0), ys.mean(0)
    variance = float(np.sum((x - cx) ** 2))
    if variance <= 1e-20:
        return {'commonViews': len(ids), 'status': 'source trajectory has no extent'}
    scale = float(np.sum((x - cx) * (ys - cy)) / variance)
    if not np.isfinite(scale) or scale <= 0:
        return {'commonViews': len(ids), 'status': 'similarity scale is not positive'}
    aligned = scale * (x - cx) + cy
    extent = float(np.max(np.linalg.norm(ys - cy, axis=1)))
    if extent <= 1e-12:
        return {'commonViews': len(ids), 'status': 'reference trajectory has no extent'}
    angles = []
    for a_rot, b_rot in zip(Rr, Rs):
        D = a_rot @ (b_rot @ Q.T).T
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1)))))
    angles = np.array(angles)
    centers = np.linalg.norm(aligned - ys, axis=1) / extent
    return {
        'status': 'aligned',
        'commonViews': len(ids),
        'rotationDeg': {
            'median': float(np.median(angles)), 'mean': float(angles.mean()), 'max': float(angles.max()),
            'fractionWithin': {str(t): float(np.mean(angles <= t)) for t in (0.1, 0.25, 0.5, 1, 2, 5)},
        },
        'centerNormalizedByReferenceRadius': {
            'median': float(np.median(centers)), 'mean': float(centers.mean()), 'max': float(centers.max()),
            'fractionWithin': {str(t): float(np.mean(centers <= t)) for t in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)},
        },
        'perView': [{'name': i, 'rotationDeg': float(r), 'centerNormalized': float(c)}
                    for i, r, c in zip(ids, angles, centers)],
        'alignment': {'Q': Q.reshape(-1).tolist(), 'scale': scale, 'translation': (cy - scale * cx).tolist()},
    }


def cmd_compare(a: argparse.Namespace) -> None:
    sequence = json.loads(a.sequence.read_text())
    names = {view['id']: view['name'] for view in sequence['views']}
    summary = json.loads((a.typecv / 'summary.json').read_text())
    model = json.loads((a.typecv / 'model.json').read_text())
    reference = json.loads(a.reference.read_text())
    source = {names[view['id']]: view['pose'] for view in model['views']}
    report = {
        'interpretation': 'Same COLMAP SIFT features and matches; fixed intrinsics in TypeCV. The references are not ground truth. Counts use different accepted observation sets.',
        'sequence': sequence['metadata'],
        'typecv': {
            'registeredViews': len(model['views']),
            'landmarks': len(model['landmarks']),
            'observations': summary['observations'],
            'meanTrackLength': summary['observations'] / max(1, len(model['landmarks'])),
            'meanEuclideanErrorPx': summary['diagnostics']['meanReprojectionErrorPx'],
            'elapsedSeconds': summary['elapsedMs'] / 1000,
            'backend': summary.get('backend', 'cpu'),
            'options': summary.get('options'),
            'unregistered': summary.get('unregistered'),
        },
        'references': {},
    }
    for key in ('bundled', 'mapper'):
        if key not in reference:
            continue
        entry = reference[key]
        target = {view['name']: view['pose'] for view in entry['views']}
        report['references'][key] = {
            **{k: v for k, v in entry.items() if k not in ('views', 'cameras')},
            'agreement': align(source, target),
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    brief = {
        'typecv': {k: report['typecv'][k] for k in ('registeredViews', 'landmarks', 'observations', 'meanEuclideanErrorPx', 'elapsedSeconds')},
        'references': {
            key: {
                'registeredViews': value['registeredViews'],
                'landmarks': value.get('landmarks'),
                'observations': value.get('observations'),
                'elapsedSeconds': value.get('elapsedSeconds'),
                'medianRotationDeg': value['agreement'].get('rotationDeg', {}).get('median'),
                'medianCenterNormalized': value['agreement'].get('centerNormalizedByReferenceRadius', {}).get('median'),
            }
            for key, value in report['references'].items()
        },
    }
    print(json.dumps(brief, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    fetch = sub.add_parser('fetch', help='download, verify and extract a dataset')
    fetch.add_argument('--select', type=int, default=0, help='extract only the database, the sparse model and the first N images by name')
    fetch.add_argument('name', choices=sorted(DATASETS))
    fetch.add_argument('directory', type=Path)
    fetch.set_defaults(run=cmd_fetch)

    export = sub.add_parser('export', help='write TypeCV sequence JSON from the COLMAP database')
    export.add_argument('dataset', type=Path, help='extracted dataset directory (images/, sparse/, database.db)')
    export.add_argument('output', type=Path)
    export.add_argument('--images', type=int, default=0, help='keep the first N images by name (0 = all)')
    export.add_argument('--step', type=int, default=1, help='keep every k-th image by name')
    export.add_argument('--window', type=int, default=0, help='only pairs within this index distance (0 = all pairs)')
    export.add_argument('--matches', choices=('raw', 'inlier'), default='raw',
                        help='raw putative matches (TypeCV verifies) or COLMAP two-view inlier matches')
    export.add_argument('--min-matches', type=int, default=15)
    export.add_argument('--max-edge', type=int, default=0, help='scale coordinates to this longest edge (0 = source resolution)')
    export.add_argument('--camera', choices=('reference', 'database'), default='reference',
                        help="intrinsics from the bundled refined model or the database's prior")
    export.set_defaults(run=cmd_export)

    reference = sub.add_parser('reference', help='summarize the bundled model and optionally run the pycolmap mapper')
    reference.add_argument('dataset', type=Path)
    reference.add_argument('output', type=Path)
    reference.add_argument('--sequence', type=Path, help='exported sequence JSON; limits views to its subset')
    reference.add_argument('--run-mapper', type=Path, help='empty directory for a pycolmap incremental_mapping run')
    reference.add_argument('--fixed-camera', action='store_true',
                           help='use the bundled refined intrinsics and keep them fixed in the mapper run')
    reference.add_argument('--threads', type=int, default=-1)
    reference.set_defaults(run=cmd_reference)

    compare = sub.add_parser('compare', help='align a TypeCV model to the references')
    compare.add_argument('typecv', type=Path, help='TypeCV output directory (summary.json, model.json)')
    compare.add_argument('sequence', type=Path, help='exported sequence JSON (view id to image name)')
    compare.add_argument('reference', type=Path, help='reference JSON from the reference subcommand')
    compare.add_argument('output', type=Path)
    compare.set_defaults(run=cmd_compare)

    a = parser.parse_args()
    a.run(a)


if __name__ == '__main__':
    main()
