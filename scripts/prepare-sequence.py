"""Decode video frames and export feature identities/matches for incremental TypeCV.

OpenCV supplies only SIFT and descriptor matching. Geometry verification, tracks,
poses, triangulation and bundle adjustment run in TypeCV. No media is downloaded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import cv2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('video', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--width', type=int, default=960)
parser.add_argument('--times', type=float, nargs='+', default=[0, 1, 2, 4, 8, 12, 16, 20])
parser.add_argument('--focal-width', type=float, default=1.2,
                    help='Assumed focal/width ratio; not camera calibration')
parser.add_argument('--camera', type=Path, help='Optional calibrated camera JSON (parameters for resized frames)')
a = parser.parse_args()
if a.width < 64 or not math.isfinite(a.focal_width) or a.focal_width <= 0 or any(not math.isfinite(t) for t in a.times) or len(a.times) < 2 or len(set(a.times)) != len(a.times) or min(a.times) < 0:
    parser.error('invalid width, focal assumption or timestamps')
cv2.setNumThreads(1)
cv2.setRNGSeed(42)
cap = cv2.VideoCapture(str(a.video))
if not cap.isOpened():
    parser.error('cannot open video')
fps = cap.get(cv2.CAP_PROP_FPS)
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
if max(a.times) >= duration:
    parser.error(f'timestamp outside duration {duration}')
if a.output.exists() and any(a.output.iterdir()):
    parser.error('output directory must be empty to avoid stale frame/feature files')
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'images').mkdir(exist_ok=True)
sift = cv2.SIFT_create(nfeatures=4000)
matcher = cv2.BFMatcher(cv2.NORM_L2)
views, descriptors, records = [], [], []
for i, t in enumerate(sorted(a.times)):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'cannot decode {t}s')
    frame = cv2.resize(frame, (a.width, round(frame.shape[0] * a.width / frame.shape[1])))
    camera = json.loads(a.camera.read_text()) if a.camera else {
        'fx': a.width * a.focal_width, 'fy': a.width * a.focal_width,
        'cx': frame.shape[1] / 2, 'cy': frame.shape[0] / 2,
    }
    keypoints, descriptor = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    if descriptor is None:
        raise RuntimeError(f'no features at {t}s')
    filename = f'frame-{i:04d}.png'
    cv2.imwrite(str(a.output / 'images' / filename), frame)
    views.append({'id': i, 'camera': camera, 'keypoints': [c for p in keypoints for c in p.pt]})
    descriptors.append(descriptor)
    records.append({'id': i, 'requestedSeconds': t, 'name': filename, 'features': len(keypoints)})
pairs = []
for i in range(len(views)):
    for j in range(i + 1, len(views)):
        def ratio(x, y):
            return [m for neighbors in matcher.knnMatch(x, y, k=2) if len(neighbors) == 2
                    for m, n in [neighbors] if m.distance < .75 * n.distance]
        backward = {(m.trainIdx, m.queryIdx) for m in ratio(descriptors[j], descriptors[i])}
        candidates = [m for m in ratio(descriptors[i], descriptors[j]) if (m.queryIdx, m.trainIdx) in backward]
        selected, counts = [], {}
        for m in sorted(candidates, key=lambda m: (m.distance, m.queryIdx, m.trainIdx)):
            x, y = views[i]['keypoints'][m.queryIdx * 2:m.queryIdx * 2 + 2]
            cell = (min(7, int(x / a.width * 8)), min(5, int(y / frame.shape[0] * 6)))
            if counts.get(cell, 0) >= 12:
                continue
            counts[cell] = counts.get(cell, 0) + 1
            selected.append(m)
            if len(selected) == 400:
                break
        pairs.append({'first': i, 'second': j, 'matches': [v for m in selected for v in (m.queryIdx, m.trainIdx)],
                      'distances': [m.distance for m in selected]})
        print(f'{i}-{j}: {len(selected)} correspondences')
cap.release()
with a.video.open('rb') as source:
    checksum = hashlib.file_digest(source, 'sha256').hexdigest()
result = {'metadata': {'sourceSha256': checksum, 'width': frame.shape[1], 'height': frame.shape[0],
                       'opencvVersion': cv2.__version__, 'frames': records,
                       'calibration': 'provided for resized frames' if a.camera else 'ASSUMED focal, zero distortion; no ground truth'},
          'views': views, 'pairs': pairs}
(a.output / 'sequence.json').write_text(json.dumps(result) + '\n')
