import test from 'node:test';
import assert from 'node:assert/strict';
import { bundleAdjust } from '../src/reconstruction/bundle-adjustment.ts';
import {
  bundleObservationJacobian,
  projectBundlePoint,
} from '../src/reconstruction/bundle-directions.ts';
import { updatePose, cameraCenter } from '../src/geometry/pose.ts';
import {
  bundleDirectionsFixture,
  disconnectedBundleDirectionsFixture,
} from './fixtures/bundle-directions.ts';

test('direction factors have exactly zero translation derivative and finite-difference rotation', () => {
  const { truth, directions } = bundleDirectionsFixture(),
    v = truth.views[4]!,
    d = directions[0]!.direction,
    pose = { R: v.pose.R, t: Float64Array.of(1e8, -2e8, 3e8) },
    jac = bundleObservationJacobian(v.camera, pose, d, true)!;
  assert.ok(jac);
  for (let k = 0; k < 6; k++) {
    const step = new Float64Array(6);
    step[k] = 1e-6;
    const pixel = projectBundlePoint(v.camera, updatePose(pose, step), d, true)!;
    for (let r = 0; r < 2; r++) {
      if (k >= 3) {
        assert.equal(jac.pose[r * 6 + k], 0);
        assert.equal(pixel[r], jac.pixel[r]);
      } else assert.ok(Math.abs((pixel[r]! - jac.pixel[r]!) / 1e-6 - jac.pose[r * 6 + k]!) < 0.002);
    }
  }
});

test('fixed and joint-focal BA keep bearings separate, normalize them and preserve the finite gauge', () => {
  for (const refineSharedFocal of [false, true]) {
    const f = bundleDirectionsFixture(refineSharedFocal ? 1.06 : 1),
      before = structuredClone(f),
      r = bundleAdjust(f.input, {
        directionalConstraints: f.directions,
        refineSharedFocal,
        iterations: 40,
      });
    assert.ok(r.diagnostics.finalRmsePx < 0.01, `${r.diagnostics.finalRmsePx}`);
    assert.equal(r.reconstruction.landmarks.length, 60);
    assert.equal(r.directionalConstraints!.length, 60);
    assert.equal(r.diagnostics.directionalObservations, 300);
    assert.equal(r.diagnostics.observations, 600);
    assert.ok(r.reconstruction.landmarks.every((p) => p.trackId < 60));
    const pointRmse = Math.sqrt(
      r.reconstruction.landmarks.reduce(
        (sum, p, i) =>
          sum + p.xyz.reduce((s, x, k) => s + (x - f.truth.landmarks[i]!.xyz[k]!) ** 2, 0),
        0,
      ) / 60,
    );
    assert.ok(pointRmse < 0.01, `point error ${pointRmse} in one fixed map gauge`);
    for (const [i, v] of r.reconstruction.views.entries()) {
      const trace = v.pose.R.reduce((sum, x, k) => sum + x * f.truth.views[i]!.pose.R[k]!, 0),
        angle = (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI;
      assert.ok(angle < 0.01, `rotation error ${angle} degrees`);
    }
    for (const d of r.directionalConstraints!)
      assert.ok(Math.abs(Math.hypot(...d.direction) - 1) < 1e-12);
    const a = cameraCenter(r.reconstruction.views[0]!.pose),
      b = cameraCenter(r.reconstruction.views[1]!.pose);
    assert.ok(Math.abs(Math.hypot(...a.map((x, i) => x - b[i]!)) - r.diagnostics.baseline) < 1e-10);
    assert.deepEqual(f, before, 'solver must own both finite and directional input snapshots');
    if (refineSharedFocal)
      for (const v of r.reconstruction.views) assert.ok(Math.abs(v.camera.fx / 480 - 1) < 0.005);
  }
});

test('direction constraints cannot supply a finite gauge, connect finite components or own the same feature twice', () => {
  const f = bundleDirectionsFixture();
  for (const refineSharedFocal of [false, true]) {
    const options = { directionalConstraints: f.directions, refineSharedFocal };
    assert.throws(() => bundleAdjust({ views: f.input.views, landmarks: [] }, options), RangeError);
    const disconnected = disconnectedBundleDirectionsFixture();
    const bridge = f.directions.map((d) => ({
      ...d,
      observations: [...d.observations.slice(0, 3), { ...d.observations[4]!, viewId: 5 }],
    }));
    assert.throws(
      () => bundleAdjust(disconnected, { ...options, directionalConstraints: bridge }),
      /connected observation graph/,
    );
    for (const bad of [
      { ...f.directions[0]!, direction: new Float64Array(3) },
      { ...f.directions[0]!, trackId: 0 },
      { ...f.directions[0]!, observations: f.input.landmarks[0]!.observations },
    ])
      assert.throws(
        () => bundleAdjust(f.input, { ...options, directionalConstraints: [bad] }),
        RangeError,
      );
  }
});
