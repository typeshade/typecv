import test from 'node:test';
import assert from 'node:assert/strict';
import { cameraProjection, unprojectPixel, validateCamera } from '../src/geometry/camera.ts';
import {
  cameraCenter,
  identityPose,
  observationJacobian,
  projectPoint,
  rotationExp,
  updatePose,
  validatePose,
} from '../src/geometry/pose.ts';
import { inverse3, nullspace, solveSystem } from '../src/geometry/numerics.ts';
import { triangulateTrack } from '../src/geometry/multiview.ts';
import { buildTracks } from '../src/reconstruction/tracks.ts';
import { scene } from './scene.ts';
const K = { fx: 800, fy: 1000, cx: 400, cy: 300, k1: -0.08, k2: 0.01 };

test('radial camera round trips and projection derivatives agree with central differences', () => {
  for (const X of [
    [0, 0, 3],
    [-1.3, 0.9, 3],
    [2, -1, 4],
  ]) {
    const projection = cameraProjection(K, X)!;
    assert.ok(projection);
    const p = unprojectPixel(K, projection.pixel)!;
    assert.ok(Math.hypot(p[0] - X[0]! / X[2]!, p[1] - X[1]! / X[2]!) < 1e-11);
    for (let k = 0; k < 3; k++) {
      const a = [...X],
        b = [...X];
      a[k]! += 1e-5;
      b[k]! -= 1e-5;
      const pa = cameraProjection(K, a)!.pixel,
        pb = cameraProjection(K, b)!.pixel;
      for (let r = 0; r < 2; r++)
        assert.ok(Math.abs((pa[r]! - pb[r]!) / 2e-5 - projection.jacobian[r * 3 + k]!) < 1e-6);
    }
  }
  assert.equal(cameraProjection(K, [0, 0, -1]), undefined);
  assert.equal(cameraProjection({ ...K, k1: -1, k2: 0 }, [1, 0, 1]), undefined);
  assert.equal(unprojectPixel({ ...K, k1: -1, k2: 0 }, [2000, 300]), undefined);
  assert.throws(() => validateCamera({ ...K, fx: 0 }), RangeError);
});

test('world-point and left-pose analytic Jacobians match central differences', () => {
  const pose = { R: rotationExp([0.05, -0.1, 0.02]), t: Float64Array.of(0.2, -0.1, 0.05) },
    X = Float64Array.of(0.6, -0.3, 5);
  const j = observationJacobian(K, pose, X)!;
  for (let k = 0; k < 6; k++) {
    const d = new Float64Array(6);
    d[k] = 1e-6;
    const a = projectPoint(K, updatePose(pose, d), X)!;
    d[k] = -1e-6;
    const b = projectPoint(K, updatePose(pose, d), X)!;
    for (let r = 0; r < 2; r++)
      assert.ok(Math.abs((a[r]! - b[r]!) / 2e-6 - j.pose[r * 6 + k]!) < 1e-5, `pose ${r},${k}`);
  }
  for (let k = 0; k < 3; k++) {
    const a = X.slice(),
      b = X.slice();
    a[k] += 1e-6;
    b[k] -= 1e-6;
    const pa = projectPoint(K, pose, a)!,
      pb = projectPoint(K, pose, b)!;
    for (let r = 0; r < 2; r++)
      assert.ok(Math.abs((pa[r]! - pb[r]!) / 2e-6 - j.point[r * 3 + k]!) < 1e-5);
  }
  validatePose(updatePose(pose, Float64Array.of(0.01, -0.02, 0.03, 0.1, 0, 0)));
  assert.throws(() => validatePose({ R: new Float64Array(9), t: new Float64Array(3) }), /SO\(3\)/);
});

test('direct projective SVD finds nullspace without forming AtA', () => {
  const rows = [
    [1, 0, 0, -2],
    [0, 1, 0, 3],
    [0, 0, 1, -4],
    [2, 1, 3, -13],
  ];
  const result = nullspace(rows, 4);
  for (const row of rows)
    assert.ok(Math.abs(row.reduce((s, v, i) => s + v * result.vector[i]!, 0)) < 1e-10);
  assert.ok(result.singularValues[1]! > 0.1);
  const minimal = nullspace(rows.slice(0, 3), 4);
  for (const row of rows.slice(0, 3))
    assert.ok(Math.abs(row.reduce((s, v, i) => s + v * minimal.vector[i]!, 0)) < 1e-10);
  assert.deepEqual(solveSystem([2, 0, 0, 4], [6, 8]), Float64Array.of(3, 2));
  assert.equal(inverse3(new Float64Array(9)), undefined);
});

test('tracks are order-independent and reject transitive same-image conflicts', () => {
  const { inputs } = scene({ count: 4, views: 3 });
  const pairs = [
    {
      first: 10,
      second: 20,
      matches: Uint32Array.of(0, 0, 1, 1),
      distances: Float64Array.of(1, 1),
    },
    {
      first: 20,
      second: 30,
      matches: Uint32Array.of(0, 0, 1, 1),
      distances: Float64Array.of(1, 1),
    },
    { first: 10, second: 30, matches: Uint32Array.of(0, 1), distances: Float64Array.of(9) },
  ];
  const a = buildTracks(inputs, pairs),
    b = buildTracks([...inputs].reverse(), [...pairs].reverse());
  assert.deepEqual(a, b);
  assert.equal(a.tracks.length, 2);
  assert.equal(a.conflictingEdges, 1);
  assert.ok(
    a.tracks.every(
      (t) => new Set(t.observations.map((o) => o.viewId)).size === t.observations.length,
    ),
  );
  assert.throws(
    () => buildTracks(inputs, [{ first: 10, second: 20, matches: Uint32Array.of(999, 0) }]),
    /index/,
  );
  assert.throws(() => buildTracks([inputs[0]!, inputs[0]!], []), /unique/);
});

test('multi-view triangulation rejects a corrupt observation and recovers common-frame XYZ', () => {
  const { registered, points } = scene({ views: 5 });
  const X = points[0]!;
  const observations = registered.map((v) => ({
    camera: v.camera,
    pose: v.pose,
    pixel: projectPoint(v.camera, v.pose, X)!,
  }));
  observations[2] = { ...observations[2]!, pixel: [10, 20] };
  const result = triangulateTrack(observations)!;
  assert.ok(result);
  assert.deepEqual(Array.from(result.inliers), [1, 1, 0, 1, 1]);
  assert.ok(Math.hypot(...result.xyz.map((v, i) => v - X[i]!)) < 1e-8);
  assert.ok(result.maxAngleDeg > 1);
  const same = { camera: K, pose: identityPose(), pixel: [400, 300] as const };
  assert.equal(triangulateTrack([same, same]), undefined);
  assert.ok(cameraCenter(registered[4]!.pose)[0]! > 1);
});

test('radial projection rejects outer positive-derivative branches after a lens fold', () => {
  const folded = { fx: 100, fy: 100, cx: 0, cy: 0, k1: -1, k2: 0.2 };
  // At r=2 the local radial derivative is positive again, but inversion is globally ambiguous.
  assert.equal(cameraProjection(folded, [2, 0, 1]), undefined);
  assert.equal(cameraProjection(folded, [0.8, 0, 1]), undefined);
  assert.ok(cameraProjection(folded, [0.2, 0.1, 1]));
});
