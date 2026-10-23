import { multiply, svd3, transpose } from '../dist/sfm/linear.js';

/** Evaluation only: one gauge from the truth seed pose and baseline, including rotation prefixes. */
export function compareSeedGauge(truth, candidate, initialPair) {
  const center = (p) =>
    [0, 1, 2].map((i) => -p.R[i] * p.t[0] - p.R[3 + i] * p.t[1] - p.R[6 + i] * p.t[2]);
  const anchor = truth.views.find((v) => v.id === initialPair[0]),
    other = truth.views.find((v) => v.id === initialPair[1]);
  const C = center(anchor.pose),
    B = center(other.pose),
    scale = Math.hypot(...B.map((x, i) => x - C[i]));
  if (!(scale > 1e-12))
    return { status: 'unavailable', reason: 'Seed has no ground-truth baseline' };
  const world = (p) =>
    [0, 1, 2].map(
      (i) =>
        C[i] +
        scale *
          (anchor.pose.R[i] * p[0] + anchor.pose.R[3 + i] * p[1] + anchor.pose.R[6 + i] * p[2]),
    );
  const cameras = candidate.views.map((v) => {
    const expected = truth.views.find((t) => t.id === v.id),
      R = multiply(expected.pose.R, transpose(anchor.pose.R, 3), 3),
      trace = v.pose.R.reduce((n, x, i) => n + x * R[i], 0),
      target = center(expected.pose);
    return {
      id: v.id,
      rotationErrorDeg: (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI,
      centerError: Math.hypot(...world(center(v.pose)).map((x, i) => x - target[i])),
    };
  });
  const points = candidate.landmarks.flatMap((p) => {
    const expected = truth.points?.[p.observations[0]?.featureId];
    return expected ? [Math.hypot(...world(p.xyz).map((x, i) => x - expected[i]))] : [];
  });
  return {
    status: 'aligned',
    method: 'one fixed truth seed pose and baseline gauge, no per-frame alignment',
    initialPair,
    scale,
    maxRotationErrorDeg: Math.max(...cameras.map((v) => v.rotationErrorDeg)),
    cameraCenterRmse: Math.sqrt(
      cameras.reduce((n, v) => n + v.centerError ** 2, 0) / cameras.length,
    ),
    evaluatedPoints: points.length,
    pointRmse: points.length
      ? Math.sqrt(points.reduce((n, x) => n + x * x, 0) / points.length)
      : null,
  };
}

// Independent CPU evaluation only. Models may use different seed frames/scales.
export function compareCameraModels(reference, candidate) {
  const expected = new Map(reference.views.map((v) => [v.id, v])),
    shared = candidate.views.filter((v) => expected.has(v.id));
  const center = (p) =>
    [0, 1, 2].map((i) => -p.R[i] * p.t[0] - p.R[3 + i] * p.t[1] - p.R[6 + i] * p.t[2]);
  const x = shared.map((v) => center(v.pose)),
    y = shared.map((v) => center(expected.get(v.id).pose)),
    mean = (points) => [0, 1, 2].map((j) => points.reduce((s, p) => s + p[j], 0) / points.length),
    mx = mean(x),
    my = mean(y),
    covariance = new Float64Array(9);
  let variance = 0;
  x.forEach((point, i) => {
    const a = point.map((v, j) => v - mx[j]),
      b = y[i].map((v, j) => v - my[j]);
    variance += a.reduce((s, v) => s + v * v, 0);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) covariance[r * 3 + c] += b[r] * a[c];
  });
  const reject = (reason) => ({ alignment: { status: 'unavailable', reason }, cameras: [] });
  if (shared.length < 3 || !(variance > 1e-20))
    return reject('Fewer than three shared noncoincident camera centers');
  let Q;
  try {
    const { U, Vt } = svd3(covariance);
    // The reference SVD completes U/V to right-handed bases: UV^T is the
    // proper-rotation Procrustes solution, including the reflection correction.
    Q = multiply(U, Vt, 3);
  } catch {
    return reject('Degenerate camera-center covariance; similarity rotation is not observable');
  }
  const rotate = (p) =>
      [0, 1, 2].map((r) => Q[r * 3] * p[0] + Q[r * 3 + 1] * p[1] + Q[r * 3 + 2] * p[2]),
    energy = Q.reduce((s, v, i) => s + v * covariance[i], 0),
    scale = energy / variance,
    translation = my.map((v, j) => v - scale * rotate(mx)[j]);
  if (!(scale > 0 && Number.isFinite(scale))) return reject('Invalid positive similarity scale');
  const cameras = shared.map((v, i) => {
    const transformed = rotate(x[i]).map((a, j) => scale * a + translation[j]),
      R = multiply(v.pose.R, transpose(Q, 3), 3),
      target = expected.get(v.id).pose,
      trace = R.reduce((s, a, j) => s + a * target.R[j], 0);
    return {
      id: v.id,
      rotationDifferenceDeg:
        (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI,
      centerDifference: Math.hypot(...transformed.map((a, j) => a - y[i][j])),
    };
  });
  return {
    alignment: {
      status: 'aligned',
      method: 'Umeyama camera-center similarity with proper rotation',
      convention: 'X_reference = scale * rotation * X_candidate + translation',
      scale,
      rotation: [...Q],
      translation,
      centerRmse: Math.sqrt(
        cameras.reduce((s, v) => s + v.centerDifference ** 2, 0) / cameras.length,
      ),
      maxCenterDifference: Math.max(...cameras.map((v) => v.centerDifference)),
      meanRotationDifferenceDeg:
        cameras.reduce((s, v) => s + v.rotationDifferenceDeg, 0) / cameras.length,
      maxRotationDifferenceDeg: Math.max(...cameras.map((v) => v.rotationDifferenceDeg)),
    },
    cameras,
  };
}
