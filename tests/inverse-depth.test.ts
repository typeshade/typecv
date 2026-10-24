import test from 'node:test';
import assert from 'node:assert/strict';
import {
  fitInverseDepth,
  initializeInverseDepth,
  inverseDepthJacobian,
  inverseDepthPoint,
  reanchorInverseDepth,
  rescaleInverseDepth,
} from '../src/geometry/inverse-depth.ts';
import { identityPose, projectPoint, rotationExp, updatePose } from '../src/geometry/pose.ts';
import { seededRandom } from '../src/geometry/numerics.ts';

const camera = { fx: 800, fy: 720, cx: 320, cy: 240, k1: -0.02 },
  X = Float64Array.of(1.2, -0.8, 50);
const observations = (centers: number[]) =>
  centers.map((x) => {
    const pose = { ...identityPose(), t: Float64Array.of(-x, 0, 0) };
    return { camera, pose, pixel: projectPoint(camera, pose, X)! };
  });

test('anchored point/camera/focal Jacobians match finite differences without a large-depth surrogate', () => {
  const obs = observations([0.2, 1]),
    p = initializeInverseDepth(obs[0]!, X),
    target = obs[1]!,
    j = inverseDepthJacobian(camera, target.pose, p)!,
    h = 1e-6;
  for (const rho of [0, p.parameters[2]!]) {
    const point = { ...p, parameters: p.parameters.slice() };
    point.parameters[2] = rho;
    const exact = inverseDepthJacobian(camera, target.pose, point)!;
    for (let k = 0; k < 3; k++) {
      const plus = { ...point, parameters: point.parameters.slice() },
        minus = { ...point, parameters: point.parameters.slice() };
      plus.parameters[k] += h;
      minus.parameters[k] -= h;
      const a = inverseDepthJacobian(camera, target.pose, plus)!,
        b = inverseDepthJacobian(camera, target.pose, minus)!;
      for (let row = 0; row < 2; row++)
        assert.ok(
          Math.abs((a.pixel[row]! - b.pixel[row]!) / (2 * h) - exact.point[3 * row + k]!) < 1e-5,
        );
    }
    for (let k = 0; k < 6; k++) {
      const d = new Float64Array(6);
      d[k] = h;
      const a = inverseDepthJacobian(camera, updatePose(target.pose, d), point)!;
      d[k] = -h;
      const b = inverseDepthJacobian(camera, updatePose(target.pose, d), point)!;
      for (let row = 0; row < 2; row++)
        assert.ok(
          Math.abs((a.pixel[row]! - b.pixel[row]!) / (2 * h) - exact.pose[6 * row + k]!) < 1e-5,
        );
    }
    if (rho === 0) for (const at of [3, 4, 5, 9, 10, 11]) assert.ok(exact.pose[at] === 0);
  }
  assert.ok(Math.hypot(...inverseDepthPoint(p)!.map((v, i) => v - X[i]!)) < 1e-10);
  assert.ok(Math.hypot(j.pixel[0] - target.pixel[0], j.pixel[1] - target.pixel[1]) < 1e-10);
});

test('pure rotation keeps depth unobservable; later translation resolves the same observations', () => {
  const rotating = [0, 0.05, 0.1, 0.15].map((yaw) => {
    const pose = { R: rotationExp([0, yaw, 0]), t: new Float64Array(3) };
    return { camera, pose, pixel: projectPoint(camera, pose, X)! };
  });
  const unknown = fitInverseDepth(rotating);
  assert.equal(unknown.covariance, undefined);
  assert.equal(unknown.point.parameters[2], 0);
  assert.equal(inverseDepthPoint(unknown.point), undefined);
  const resolved = fitInverseDepth([...rotating, ...observations([0.5, 1])]);
  assert.ok(resolved.covariance);
  assert.ok(resolved.rmsePx < 1e-6);
  assert.ok(Math.hypot(...inverseDepthPoint(resolved.point)!.map((v, i) => v - X[i]!)) < 1e-4);
  const weak = fitInverseDepth(observations([0, 0.02])),
    strong = fitInverseDepth(observations([0, 0.02, 1]));
  assert.ok(strong.inverseRange.stddev! < weak.inverseRange.stddev! / 20);
});

test('reference and scale changes transform covariance with the coordinate chart', () => {
  const result = fitInverseDepth(observations([0, 0.5, 1])),
    pose = observations([0.4])[0]!.pose,
    changed = reanchorInverseDepth(result.point, pose, result.covariance),
    back = reanchorInverseDepth(changed.point, result.point.reference, changed.covariance);
  assert.ok(back.covariance!.every((x, i) => Math.abs(x - result.covariance![i]!) < 1e-12));
  assert.ok(inverseDepthPoint(changed.point)!.every((x, i) => Math.abs(x - X[i]!) < 1e-4));
  const scaled = rescaleInverseDepth(result.point, 100, result.covariance);
  assert.ok(Math.abs((scaled.covariance![8]! * 10000) / result.covariance![8]! - 1) < 1e-12);
  assert.ok(inverseDepthPoint(scaled.point)!.every((x, i) => Math.abs(x / 100 - X[i]!) < 1e-4));
});

test('conditional covariance predicts independent pixel noise; it is not pose/calibration uncertainty', () => {
  const obs = observations([0, 0.4, 1]),
    sigma = 0.2,
    reference = fitInverseDepth(obs, { pixelNoiseStdDevPx: sigma }),
    rng = seededRandom(8351),
    values: number[] = [];
  const gaussian = () =>
    Math.sqrt(-2 * Math.log(Math.max(1e-12, rng()))) * Math.cos(2 * Math.PI * rng());
  for (let i = 0; i < 300; i++) {
    const noisy = obs.map((o) => ({
      ...o,
      pixel: [o.pixel[0] + sigma * gaussian(), o.pixel[1] + sigma * gaussian()] as const,
    }));
    values.push(fitInverseDepth(noisy, { pixelNoiseStdDevPx: sigma }).point.parameters[2]!);
  }
  const mean = values.reduce((s, v) => s + v, 0) / values.length,
    variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1);
  assert.equal(reference.conditionalOn, 'fixed-poses-and-intrinsics');
  assert.ok(Math.abs(variance / reference.covariance![8]! - 1) < 0.25);
});
