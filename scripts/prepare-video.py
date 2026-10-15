"""Extract SIFT correspondences for the TypeCV two-view video evaluator.

Requires opencv-python-headless==5.0.0.93; no OpenCV pose/essential solver is used.
The homography fit is a diagnostic only, not an input filter for TypeCV.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('video', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--width', type=int, default=960)
p.add_argument('--times', type=float, nargs='+', default=[1, 2, 4, 8, 12, 16, 20])
a = p.parse_args()
if a.width < 64 or any(t <= 0 for t in a.times):
    p.error('width must be >=64 and timestamps must be positive')
cv2.setNumThreads(1)
cv2.setRNGSeed(42)
cap = cv2.VideoCapture(str(a.video))
if not cap.isOpened():
    p.error('cannot open video')
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
a.output.mkdir(parents=True, exist_ok=True)
sift = cv2.SIFT_create(nfeatures=4000)
bf = cv2.BFMatcher(cv2.NORM_L2)

def frame(t):
    if t >= duration:
        raise ValueError(f'{t}s is outside video duration {duration}s')
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, image = cap.read()
    if not ok:
        raise ValueError(f'cannot decode frame at {t}s')
    image = cv2.resize(image, (a.width, round(image.shape[0] * a.width / image.shape[1])))
    cv2.imwrite(str(a.output / f'frame-{t:g}.jpg'), image)
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    if descriptors is None:
        raise ValueError(f'no features at {t}s')
    return image, keypoints, descriptors

base, k1, d1 = frame(0)
pairs = []
for t in a.times:
    image, k2, d2 = frame(t)
    def ratio(x, y):
        return [m for neighbors in bf.knnMatch(x, y, k=2) if len(neighbors) == 2
                for m, n in [neighbors] if m.distance < 0.75 * n.distance]
    reverse = {(m.trainIdx, m.queryIdx) for m in ratio(d2, d1)}
    mutual = [m for m in ratio(d1, d2) if (m.queryIdx, m.trainIdx) in reverse]
    # Spatial caps avoid filling the input with one highly textured patch.
    counts, selected = {}, []
    for m in sorted(mutual, key=lambda m: (m.distance, m.queryIdx, m.trainIdx)):
        x, y = k1[m.queryIdx].pt
        cell = (min(7, int(8*x/base.shape[1])), min(5, int(6*y/base.shape[0])))
        if counts.get(cell, 0) >= 12:
            continue
        counts[cell] = counts.get(cell, 0) + 1
        selected.append(m)
        if len(selected) >= 400:
            break
    matches = [[*k1[m.queryIdx].pt, *k2[m.trainIdx].pt] for m in selected]
    homography = None
    if len(matches) >= 4:
        import numpy as np
        pts = np.array(matches, dtype=np.float64)
        _, mask = cv2.findHomography(pts[:, :2], pts[:, 2:], cv2.RANSAC, 2.0,
                                     maxIters=2000, confidence=0.999)
        if mask is not None:
            homography = float(mask.mean())
    pairs.append({'id': f'0-{t:g}s', 'seconds': [0, t], 'matches': matches,
                  'mutualMatches': len(mutual), 'homographyInlierFraction': homography})
    print(f'0 -> {t:g}s: {len(mutual)} mutual, {len(matches)} selected, H fraction {homography}')
cap.release()
metadata = {'videoSha256': hashlib.file_digest(a.video.open('rb'), 'sha256').hexdigest(),
            'durationSeconds': duration, 'width': base.shape[1], 'height': base.shape[0],
            'opencvVersion': cv2.__version__, 'referenceSeconds': 0, 'pairs': pairs}
(a.output / 'matches.json').write_text(json.dumps(metadata, indent=2) + '\n')
