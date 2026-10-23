"""Derive a small SfM lab sample from a TUM RGB-D sequence (Sturm et al., IROS 2012).

Writes JPEG frames at the analysis size, the sequence's RGB intrinsics, and the
motion-capture pose of each frame as the reference, in the lab's sample format
(`typecv-lab-sample-v1`, the format of scripts/prepare-lab-colmap-sample.py without
features). The reference is ground truth from the mocap system, not another SfM.

  python scripts/prepare-lab-tum-sample.py \\
      data/tum/rgbd_dataset_freiburg3_structure_texture_far \\
      tests/fixtures/lab-images/tum-fr3-structure-texture-far --frames 16 --spacing 2

Conventions. groundtruth.txt holds `timestamp tx ty tz qx qy qz qw`: the camera's pose in
the world (camera-to-world). The lab's reference is world-to-camera (COLMAP's
`cam_from_world`): R = R_wc^T, t = -R_wc^T t_wc. Each frame takes the mocap pose nearest
in time (the mocap runs at 100 Hz, so the error is at most 5 ms of motion). The
intrinsics are the TUM calibration of the sequence's camera in OpenCV's convention, whose
pixel centers are integers, as TypeCV's are: no half-pixel shift. The freiburg3 images are
already undistorted (k1 = k2 = 0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

# TUM RGB-D intrinsics of the RGB camera (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).
INTRINSICS = {
    'freiburg1': {'fx': 517.3, 'fy': 516.5, 'cx': 318.6, 'cy': 255.3, 'k1': 0.2624, 'k2': -0.9531},
    'freiburg2': {'fx': 520.9, 'fy': 521.0, 'cx': 325.1, 'cy': 249.7, 'k1': 0.2312, 'k2': -0.7849},
    'freiburg3': {'fx': 535.4, 'fy': 539.2, 'cx': 320.1, 'cy': 247.6, 'k1': 0.0, 'k2': 0.0},
}


def read_list(path: Path) -> list[tuple[float, list[str]]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        rows.append((float(parts[0]), parts[1:]))
    return rows


def quaternion_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sequence', type=Path, help='extracted sequence (rgb/, rgb.txt, groundtruth.txt)')
    parser.add_argument('output', type=Path)
    parser.add_argument('--frames', type=int, default=16, help='number of frames')
    parser.add_argument('--spacing', type=float, default=2.0, help='seconds between frames')
    parser.add_argument('--start', type=float, default=0.0, help='seconds after the first frame with a mocap pose')
    parser.add_argument('--max-edge', type=int, default=640)
    parser.add_argument(
        '--depth',
        action='store_true',
        help='also write depth-NN.png: the sequence depth image nearest in time to each frame, 16-bit, '
        'at the analysis size, depth = value / 5000 m (docs/dense-mvs.md 7.3)',
    )
    parser.add_argument('--quality', type=int, default=82)
    parser.add_argument('--name', default='TUM fr3 structure_texture_far (16 frames)')
    parser.add_argument(
        '--credit',
        default='Dataset: rgbd_dataset_freiburg3_structure_texture_far, TUM RGB-D SLAM Dataset and Benchmark '
        '(J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, IROS 2012), CC BY 4.0.',
    )
    a = parser.parse_args()
    camera_kind = next((key for key in INTRINSICS if key in a.sequence.name), None)
    if camera_kind is None:
        raise SystemExit('the sequence name does not say which Freiburg camera it is')
    intrinsics = INTRINSICS[camera_kind]
    rgb = read_list(a.sequence / 'rgb.txt')
    truth = read_list(a.sequence / 'groundtruth.txt')
    truth_times = np.array([t for t, _ in truth])
    if len(rgb) < 2 or len(truth) < 2:
        raise SystemExit('rgb.txt or groundtruth.txt is empty')
    # The mocap log may start after the first RGB frame: the first frame is the first with a pose.
    depth_rows = read_list(a.sequence / 'depth.txt') if a.depth else []
    first = max(rgb[0][0], float(truth_times[0])) + a.start
    chosen = []
    for index in range(a.frames):
        target = first + index * a.spacing
        nearest = min(rgb, key=lambda row: abs(row[0] - target))
        if abs(nearest[0] - target) > 0.1:
            raise SystemExit(f'no RGB frame within 0.1 s of {target:.3f}')
        if chosen and nearest[0] <= chosen[-1][0]:
            raise SystemExit('frames must advance in time; lower --frames or --spacing')
        chosen.append(nearest)
    a.output.mkdir(parents=True, exist_ok=True)
    frames = []
    analysis = None
    for index, (time, (file_name,)) in enumerate(chosen):
        source = cv2.imread(str(a.sequence / file_name))
        if source is None:
            raise SystemExit(f'cannot read {file_name}')
        height, width = source.shape[:2]
        scale = min(1.0, a.max_edge / max(width, height))
        analysis = (round(width * scale), round(height * scale))
        resized = cv2.resize(source, analysis, interpolation=cv2.INTER_AREA) if scale < 1 else source
        name = f'frame-{index:02d}.jpg'
        cv2.imwrite(str(a.output / name), resized, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        if a.depth:
            nearest = min(depth_rows, key=lambda row: abs(row[0] - time))
            if abs(nearest[0] - time) > 0.05:
                raise SystemExit(f'no depth image within 50 ms of frame {file_name}')
            depth = cv2.imread(str(a.sequence / nearest[1][0]), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.dtype != np.uint16 or depth.shape[:2] != (height, width):
                raise SystemExit(f'cannot use depth image {nearest[1][0]}')
            if analysis != (width, height):
                # Nearest neighbour: averaging depths across an edge invents surfaces.
                depth = cv2.resize(depth, analysis, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(a.output / f'depth-{index:02d}.png'), depth, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        at = int(np.argmin(np.abs(truth_times - time)))
        if abs(truth_times[at] - time) > 0.02:
            raise SystemExit(f'no mocap pose within 20 ms of frame {file_name}')
        tx, ty, tz, qx, qy, qz, qw = (float(v) for v in truth[at][1][:7])
        r_wc = quaternion_matrix(qx, qy, qz, qw)
        t_wc = np.array([tx, ty, tz])
        r_cw = r_wc.T
        t_cw = -r_cw @ t_wc
        sx, sy = analysis[0] / width, analysis[1] / height
        frames.append({
            'id': index,
            'file': name,
            **({'depthFile': f'depth-{index:02d}.png'} if a.depth else {}),
            'sourceName': file_name,
            'time': round(time - rgb[0][0], 6),
            'width': analysis[0],
            'height': analysis[1],
            'sourceWidth': width,
            'sourceHeight': height,
            # OpenCV's integer pixel centers are TypeCV's; radial terms are scale-invariant.
            'camera': {
                'fx': intrinsics['fx'] * sx,
                'fy': intrinsics['fy'] * sy,
                'cx': intrinsics['cx'] * sx,
                'cy': intrinsics['cy'] * sy,
                'k1': intrinsics['k1'],
                'k2': intrinsics['k2'],
            },
            'reference': {'R': r_cw.reshape(-1).tolist(), 't': t_cw.tolist()},
        })
    sample = {
        'format': 'typecv-lab-sample-v1',
        'name': a.name,
        'credit': a.credit,
        'reference': 'motion-capture ground truth of the TUM RGB-D benchmark (the pose nearest each frame in time)',
        'source': {'width': frames[0]['sourceWidth'], 'height': frames[0]['sourceHeight'], 'images': len(frames)},
        'frames': frames,
    }
    if a.depth:
        sample['depthUnitsPerMeter'] = 5000
    (a.output / 'sample.json').write_text(json.dumps(sample, indent=1) + '\n')
    (a.output / 'README.txt').write_text(
        a.credit + '\n'
        + f'Derived {analysis[0]} x {analysis[1]} JPEG frames, {len(frames)} frames {a.spacing} s apart from {a.start} s, '
        + 'with the TUM RGB intrinsics and the motion-capture pose nearest each frame in sample.json, '
        + 'for an in-browser SfM demonstration. The original images are not included. '
        + ('The depth-NN.png files are the sequence depth images nearest in time, 16-bit, depth = value / 5000 m.\n'
           if a.depth else 'The depth maps are not included.\n')
    )
    digest = hashlib.sha256(b''.join((a.output / f['file']).read_bytes() for f in frames)).hexdigest()
    print(json.dumps({'frames': len(frames), 'analysis': analysis, 'framesSha256': digest}))


if __name__ == '__main__':
    main()
