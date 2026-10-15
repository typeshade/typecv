import test from 'node:test';
import assert from 'node:assert/strict';
import { estimatePosePnP } from '../src/geometry/pnp.ts';
import { estimateHomography } from '../src/geometry/homography.ts';
import { projectPoint, rotationExp, transformPoint } from '../src/geometry/pose.ts';
import { multiply, transpose } from '../src/sfm/linear.ts';
import { scene } from './scene.ts';
const K = { fx: 900, fy: 920, cx: 480, cy: 270, k1: -0.05, k2: 0.005 };
function errors(a: { R: Float64Array; t: Float64Array }, b: { R: Float64Array; t: Float64Array }) {
  const R = multiply(a.R, transpose(b.R, 3), 3);
  return {
    rotation: Math.acos(Math.max(-1, Math.min(1, (R[0]! + R[4]! + R[8]! - 1) / 2))),
    translation: Math.hypot(...a.t.map((v, i) => v - b.t[i]!)),
  };
}
test('P3P RANSAC and refit recover a known world-to-camera pose', () => {
  const { points, registered, inputs } = scene({ count: 60, camera: K });
  const v = registered[4]!,
    input = inputs[4]!;
  const result = estimatePosePnP(
    Float64Array.from(points.flatMap((p) => [...p])),
    input.keypoints,
    K,
    { seed: 42 },
  );
  assert.equal(result.inlierCount, 60);
  const e = errors(result.pose, v.pose);
  assert.ok(e.rotation < 1e-6);
  assert.ok(e.translation < 1e-7);
  assert.ok(result.meanErrorPx < 1e-7);
});
test('PnP withstands noise and 25% mismatched pixels, deterministically', () => {
  const { points, registered, inputs } = scene({ count: 100, camera: K, noise: 0.4 });
  const pixels = inputs[5]!.keypoints.slice();
  for (let i = 0; i < 25; i++) {
    pixels[2 * i] = 50 + i * 17;
    pixels[2 * i + 1] = 100 + i * 7;
  }
  const xyz = Float64Array.from(points.flatMap((p) => [...p])),
    opts = { seed: 91, iterations: 800 };
  const result = estimatePosePnP(xyz, pixels, K, opts),
    e = errors(result.pose, registered[5]!.pose);
  assert.ok(result.inlierCount >= 73);
  assert.ok(e.rotation < 0.005, JSON.stringify(e));
  assert.ok(e.translation < 0.03, JSON.stringify(e));
  assert.deepEqual(result, estimatePosePnP(xyz, pixels, K, opts));
});
test('PnP handles planar world points with positive depth', () => {
  const pose = { R: rotationExp([0.1, -0.14, 0.04]), t: Float64Array.of(0.2, -0.1, 1) };
  const X = Array.from({ length: 30 }, (_, i) =>
    Float64Array.of((i % 6) - 2.5, Math.floor(i / 6) - 2, 5),
  );
  const pixels = Float64Array.from(X.flatMap((p) => [...projectPoint(K, pose, p)!]));
  const result = estimatePosePnP(Float64Array.from(X.flatMap((p) => [...p])), pixels, K);
  const e = errors(result.pose, pose);
  assert.ok(e.rotation < 1e-6, JSON.stringify(e));
  assert.ok(e.translation < 1e-6, JSON.stringify(e));
});
test('PnP rejects collinear world points and invalid controls', () => {
  const X = Float64Array.from(Array.from({ length: 12 }, (_, i) => [i * 0.1, 0, 4]).flat()),
    pixels = Float64Array.from(Array.from({ length: 12 }, (_, i) => [480 + i * 20, 270]).flat());
  assert.throws(() => estimatePosePnP(X, pixels, K, { iterations: 20 }), /could not register/);
  assert.throws(() => estimatePosePnP(X, pixels, K, { thresholdPx: NaN }), RangeError);
  assert.throws(() => estimatePosePnP(X, pixels, K, { confidence: 1 }), RangeError);
});
test('robust homography identifies planar image correspondences with outliers', () => {
  const { points } = scene({ count: 80 });
  const pose = { R: rotationExp([0.03, -0.08, 0.01]), t: Float64Array.of(0.3, 0, 0.1) };
  const data = points.flatMap((p, i) => {
    const X = Float64Array.of(p[0]!, p[1]!, 5),
      a = projectPoint(K, { R: rotationExp([0, 0, 0]), t: new Float64Array(3) }, X)!,
      b = projectPoint(K, pose, X)!;
    return [...a, ...(i < 12 ? [100 + i * 19, 120 + i * 3] : b)];
  });
  // Distorted image coordinates are not homographic: test the undistorted pinhole model.
  const pin = { ...K, k1: 0, k2: 0 };
  const matches = Float64Array.from(
    points.flatMap((p, i) => {
      const X = Float64Array.of(p[0]!, p[1]!, 5),
        q = transformPoint(pose, X);
      const a = [(pin.fx * X[0]!) / 5 + pin.cx, (pin.fy * X[1]!) / 5 + pin.cy],
        b = [(pin.fx * q[0]!) / q[2]! + pin.cx, (pin.fy * q[1]!) / q[2]! + pin.cy];
      return [...a, ...(i < 12 ? [100 + i * 19, 120 + i * 3] : b)];
    }),
  );
  const H = estimateHomography(matches, { seed: 42 })!;
  assert.ok(H);
  assert.ok(H.inlierCount >= 68);
  assert.ok(H.inlierCount < 72);
  assert.equal(estimateHomography(new Float64Array(16)), undefined);
  assert.equal(data.length, 320);
});

test('four-point PnP disambiguates P3P branches without a six-point refit', () => {
  const { points, registered, inputs } = scene({ count: 4, views: 3, camera: K });
  const result = estimatePosePnP(
    Float64Array.from(points.flatMap((p) => [...p])),
    inputs[2]!.keypoints,
    K,
    { seed: 42 },
  );
  const e = errors(result.pose, registered[2]!.pose);
  assert.equal(result.inlierCount, 4);
  assert.ok(e.rotation < 1e-6, JSON.stringify(e));
  assert.ok(e.translation < 1e-6, JSON.stringify(e));
});
