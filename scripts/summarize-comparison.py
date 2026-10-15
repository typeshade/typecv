"""Compare common-camera poses after similarity alignment; COLMAP is a reference, not truth."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('typecv', type=Path, help='TypeCV output directory')
p.add_argument('colmap', type=Path, help='COLMAP baseline output directory')
p.add_argument('output', type=Path)
a = p.parse_args()
type_summary = json.loads((a.typecv / 'summary.json').read_text())
model = json.loads((a.typecv / 'model.json').read_text())
col_summary = json.loads((a.colmap / 'summary.json').read_text())
if not col_summary['models']:
    p.error('COLMAP did not reconstruct a reference model')
reference = max(col_summary['models'], key=lambda m: m['registeredViews'])
source = {v['id']: v for v in model['views']}
target = {v['id']: v for v in reference['views']}
ids = sorted(source.keys() & target.keys())
if len(ids) < 3:
    p.error('at least three common cameras needed')
Rs = np.array([source[i]['pose']['R'] for i in ids]).reshape(-1, 3, 3)
Rr = np.array([target[i]['pose']['R'] for i in ids]).reshape(-1, 3, 3)
xs = np.array([-R.T @ np.array(source[i]['pose']['t']) for i, R in zip(ids, Rs)])
ys = np.array([-R.T @ np.array(target[i]['pose']['t']) for i, R in zip(ids, Rr)])
# Align world orientation from all camera rotations, avoiding roll ambiguity on a nearly linear flight path.
u, _, vt = np.linalg.svd(sum(a.T @ b for a, b in zip(Rr, Rs)))
Q = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
x = (Q @ xs.T).T
cx, cy = x.mean(0), ys.mean(0)
scale = float(np.sum((x-cx)*(ys-cy)) / np.sum((x-cx)**2))
if not np.isfinite(scale) or scale <= 0:
    p.error('reference/source similarity scale is not positive')
aligned = scale*(x-cx)+cy
extent = float(np.max(np.linalg.norm(ys-ys.mean(0), axis=1)))
if extent <= 1e-12:
    p.error('reference trajectory has no measurable extent')
angles = []
for a_rot, b_rot in zip(Rr, Rs):
    D = a_rot @ (b_rot @ Q.T).T
    angles.append(float(np.degrees(np.arccos(np.clip((np.trace(D)-1)/2, -1, 1)))))
center_errors = np.linalg.norm(aligned-ys, axis=1)/extent
report = {
    'interpretation': 'Controlled-input fixed-intrinsics feasibility comparison. Neither reconstruction is ground truth. Counts and errors use different accepted observations.',
    'typecv': {'registeredViews': len(source), 'landmarks': len(model['landmarks']), 'observations': type_summary['observations'],
               'meanEuclideanErrorPx': type_summary['diagnostics']['meanReprojectionErrorPx'], 'elapsedSeconds': type_summary['elapsedMs']/1000},
    'colmap': {k: v for k, v in reference.items() if k not in ('views',)},
    'referenceAgreement': {'commonViewIds': ids, 'rotationDifferenceDeg': angles,
                           'medianRotationDifferenceDeg': float(np.median(angles)),
                           'centerDifferenceNormalizedByReferenceRadius': center_errors.tolist(),
                           'medianCenterDifferenceNormalizedByReferenceRadius': float(np.median(center_errors)),
                           'alignment': {'Q': Q.reshape(-1).tolist(), 'scale': scale, 'translation': (cy-scale*cx).tolist()}},
}
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'typecv': report['typecv'], 'colmap': {k:v for k,v in report['colmap'].items() if k != 'views'}}))
