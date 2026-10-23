"""Derive a small SfM lab sample from a COLMAP demo dataset.

Writes resized JPEG frames, the dataset's refined intrinsics scaled to the analysis
size (TypeCV integer pixel centers), and the bundled COLMAP poses of the chosen images,
so the lab can run the sample with known calibration and compare its cameras to
COLMAP's. With --sequence (the output of colmap_dataset.py export on the same images
and --max-edge), it also writes colmap-sift.bin: the dataset's SIFT keypoints that take
part in a raw match, and those matches, so the lab can run the mapper alone on
COLMAP's features. Only the derived previews, the JSON and the binary are meant for
the website.

  python scripts/colmap-datasets/colmap_dataset.py export data/colmap/south-building \\
      /tmp/sb12/sequence.json --images 12 --max-edge 640
  python scripts/prepare-lab-colmap-sample.py data/colmap/south-building \\
      tests/fixtures/lab-images/south-building --images 12 --max-edge 640 \\
      --sequence /tmp/sb12/sequence.json

colmap-sift.bin holds, in little-endian order: for each frame its keypoints as
float32 (x, y) pairs in analysis pixels, then for each pair its matches as uint16
(first keypoint, second keypoint) pairs. sample.json's "features" index gives the
element offset and count of each block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', type=Path, help='extracted dataset (images/, sparse/)')
    parser.add_argument('output', type=Path)
    parser.add_argument('--images', type=int, default=12, help='N images by name, after --offset')
    parser.add_argument('--offset', type=int, default=0, help='skip the first K images by name')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--max-edge', type=int, default=640)
    parser.add_argument('--quality', type=int, default=82)
    parser.add_argument('--name', default='COLMAP south-building')
    parser.add_argument('--sequence', type=Path, help='colmap_dataset.py export of the same images and --max-edge')
    parser.add_argument('--credit', default='Dataset: south-building, COLMAP demo datasets (https://demuc.de/colmap/datasets/), image credit Christopher Zach.')
    a = parser.parse_args()
    import pycolmap  # noqa: PLC0415

    model = pycolmap.Reconstruction(str(a.dataset / 'sparse'))
    images = sorted(model.images.values(), key=lambda image: image.name)[a.offset :][:: a.step][: a.images]
    if len(images) < 2:
        raise SystemExit('need at least two images')
    a.output.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, image in enumerate(images):
        camera = model.cameras[image.camera_id]
        kind = str(camera.model).split('.')[-1]
        dropped = None
        p = [float(v) for v in camera.params]
        if kind == 'SIMPLE_RADIAL':
            fx = fy = p[0]
            cx, cy, k1, k2 = p[1], p[2], p[3], 0.0
        elif kind == 'RADIAL':
            fx = fy = p[0]
            cx, cy, k1, k2 = p[1], p[2], p[3], p[4]
        elif kind == 'SIMPLE_PINHOLE':
            fx = fy = p[0]
            cx, cy, k1, k2 = p[1], p[2], 0.0, 0.0
        elif kind == 'PINHOLE':
            fx, fy, cx, cy, k1, k2 = p[0], p[1], p[2], p[3], 0.0, 0.0
        elif kind == 'OPENCV':
            # The tangential terms p1, p2 are dropped (TypeCV has none) when they move no point by
            # one pixel at the analysis size; the manifest records the drop under `camera.droppedTangential`.
            fx, fy, cx, cy, k1, k2 = p[0], p[1], p[2], p[3], p[4], p[5]
            dropped = [p[6], p[7]]
        else:
            raise SystemExit(f'unsupported camera model {kind}')
        source = cv2.imread(str(a.dataset / 'images' / image.name))
        if source is None:
            raise SystemExit(f'cannot read {image.name}')
        height, width = source.shape[:2]
        scale = min(1.0, a.max_edge / max(width, height))
        analysis = (round(width * scale), round(height * scale))
        resized = cv2.resize(source, analysis, interpolation=cv2.INTER_AREA)
        name = f'frame-{index:02d}.jpg'
        cv2.imwrite(str(a.output / name), resized, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        sx, sy = analysis[0] / width, analysis[1] / height
        if dropped:
            worst = 0.0
            for u, v in ((0, 0), (width, 0), (0, height), (width, height)):
                x, y = (u - cx) / fx, (v - cy) / fy
                r2 = x * x + y * y
                dx = 2 * dropped[0] * x * y + dropped[1] * (r2 + 2 * x * x)
                dy = dropped[0] * (r2 + 2 * y * y) + 2 * dropped[1] * x * y
                worst = max(worst, float(np.hypot(dx * fx * sx, dy * fy * sy)))
            if worst > 1.0:
                raise SystemExit(f'tangential terms move a point {worst:.2f} px at the analysis size')
        pose = image.cam_from_world()
        frames.append({
            'id': index,
            'file': name,
            'sourceName': image.name,
            'time': float(index),
            'width': analysis[0],
            'height': analysis[1],
            'sourceWidth': width,
            'sourceHeight': height,
            # COLMAP keeps pixel centers at +0.5; TypeCV uses integer centers. Radial terms are scale-invariant.
            'camera': {
                'fx': fx * sx,
                'fy': fy * sy,
                'cx': (cx * sx) - 0.5,
                'cy': (cy * sy) - 0.5,
                'k1': k1,
                'k2': k2,
                **({'droppedTangential': dropped} if dropped else {}),
            },
            'reference': {
                'R': pose.rotation.matrix().reshape(-1).tolist(),
                't': pose.translation.tolist(),
            },
        })
    features = write_features(a, frames) if a.sequence else None
    sample = {
        'format': 'typecv-lab-sample-v1',
        'name': a.name,
        'credit': a.credit,
        'reference': 'COLMAP sparse model shipped with the dataset (full pipeline, refined SIMPLE_RADIAL intrinsics)',
        'source': {'width': frames[0]['sourceWidth'], 'height': frames[0]['sourceHeight'], 'images': len(frames)},
        'frames': frames,
        **({'features': features} if features else {}),
    }
    (a.output / 'sample.json').write_text(json.dumps(sample, indent=1) + '\n')
    (a.output / 'README.txt').write_text(
        a.credit + '\n' + 'Derived ' + f'{analysis[0]} x {analysis[1]}' + ' JPEG previews of '
        + f'{len(frames)} images by name' + (f' after the first {a.offset}' if a.offset else '') + ', with the bundled COLMAP intrinsics and poses in sample.json, '
        + 'for an in-browser SfM demonstration. The original images and database are not included.\n'
        + ('colmap-sift.bin holds the SIFT keypoints of these images that take part in a raw match in the '
           + 'dataset database, scaled to the preview size, and those matches, so the lab can run its mapper '
           + 'on COLMAP\'s own features.\n' if features else '')
    )
    digest = hashlib.sha256(b''.join((a.output / f['file']).read_bytes() for f in frames)).hexdigest()
    print(json.dumps({'frames': len(frames), 'analysis': analysis, 'framesSha256': digest}))


def write_features(a: argparse.Namespace, frames: list[dict]) -> dict:
    """Compact the exported sequence to the matched keypoints and write colmap-sift.bin."""
    sequence = json.loads(a.sequence.read_text())
    views = {view['id']: view for view in sequence['views']}
    if [views[i]['name'] for i in sorted(views)] != [frame['sourceName'] for frame in frames]:
        raise SystemExit('--sequence covers different images than the frames')
    for view, frame in zip((views[i] for i in sorted(views)), frames):
        camera = view['camera']
        if abs(camera['fx'] - frame['camera']['fx']) > 1e-6 or abs(camera['cx'] - frame['camera']['cx']) > 1e-6:
            raise SystemExit('--sequence was exported at another --max-edge')
    used = {view_id: set() for view_id in views}
    for pair in sequence['pairs']:
        matches = pair['matches']
        used[pair['first']].update(matches[0::2])
        used[pair['second']].update(matches[1::2])
    remap, keypoints, index = {}, [], []
    offset = 0
    for view_id in sorted(views):
        kept = sorted(used[view_id])
        if len(kept) > 65535:
            raise SystemExit('more than 65535 matched keypoints in one image')
        remap[view_id] = {old: new for new, old in enumerate(kept)}
        source = views[view_id]['keypoints']
        block = np.array([[source[2 * k], source[2 * k + 1]] for k in kept], dtype=np.float32).reshape(-1)
        keypoints.append(block)
        index.append({'id': view_id, 'offset': offset, 'count': len(kept)})
        offset += block.size
    pairs, matches = [], []
    moffset = 0
    for pair in sorted(sequence['pairs'], key=lambda p: (p['second'], p['first'])):
        first, second = remap[pair['first']], remap[pair['second']]
        raw = pair['matches']
        block = np.array([[first[raw[i]], second[raw[i + 1]]] for i in range(0, len(raw), 2)], dtype=np.uint16).reshape(-1)
        matches.append(block)
        pairs.append({'first': pair['first'], 'second': pair['second'], 'offset': moffset, 'count': len(raw) // 2})
        moffset += block.size
    keypoint_bytes = np.concatenate(keypoints).astype('<f4').tobytes()
    match_bytes = np.concatenate(matches).astype('<u2').tobytes()
    (a.output / 'colmap-sift.bin').write_bytes(keypoint_bytes + match_bytes)
    return {
        'file': 'colmap-sift.bin',
        'source': sequence['metadata']['features'],
        'databaseSha256': sequence['metadata']['databaseSha256'],
        'keypointBytes': len(keypoint_bytes),
        'keypoints': index,
        'pairs': pairs,
    }


if __name__ == '__main__':
    main()
