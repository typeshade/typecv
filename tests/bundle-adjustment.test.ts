import test from 'node:test';
import assert from 'node:assert/strict';
import { bundleAdjust } from '../src/reconstruction/bundle-adjustment.ts';
import { cameraCenter, updatePose } from '../src/geometry/pose.ts';
import type { SparseReconstruction } from '../src/reconstruction/types.ts';
import { scene } from './scene.ts';
function perturbed() {
  const { reconstruction, rng } = scene({ count: 70, views: 5, noise: 0.2 });
  const input: SparseReconstruction = {
    views: reconstruction.views.map((v, i) =>
      i === 0
        ? v
        : {
            ...v,
            pose: updatePose(
              v.pose,
              Float64Array.from(
                [0.005, 0.005, 0.005, 0.02, 0.02, 0.02],
                (s) => (rng() - 0.5) * s * 2,
              ),
            ),
          },
    ),
    landmarks: reconstruction.landmarks.map((p) => ({
      ...p,
      xyz: p.xyz.map((x) => x + (rng() - 0.5) * 0.08),
    })),
  };
  return input;
}
test('joint Schur BA decreases reprojection objective and preserves gauge/input observations', () => {
  const input = perturbed(),
    snapshot = structuredClone(input),
    output = bundleAdjust(input, { iterations: 25 }),
    d = output.diagnostics;
  assert.ok(d.acceptedSteps > 0);
  assert.ok(d.finalCost < d.initialCost * 0.05, JSON.stringify(d));
  assert.ok(d.finalRmsePx < 0.3, JSON.stringify(d));
  assert.equal(d.observations, 350);
  assert.deepEqual(input, snapshot);
  assert.deepEqual(output.reconstruction.views[0]!.pose, input.views[0]!.pose);
  const a = cameraCenter(output.reconstruction.views[0]!.pose),
    b = cameraCenter(output.reconstruction.views[1]!.pose);
  assert.ok(Math.abs(Math.hypot(...a.map((v, i) => v - b[i]!)) - d.baseline) < 1e-10);
  for (let i = 0; i < input.landmarks.length; i++)
    assert.deepEqual(
      output.reconstruction.landmarks[i]!.observations,
      input.landmarks[i]!.observations,
    );
  console.log('BA', JSON.stringify(d));
});
test('BA with zero iterations returns independent unchanged geometry', () => {
  const input = perturbed(),
    output = bundleAdjust(input, { iterations: 0 });
  assert.deepEqual(output.reconstruction, input);
  assert.notEqual(output.reconstruction.landmarks[0]!.xyz, input.landmarks[0]!.xyz);
  assert.equal(output.diagnostics.initialCost, output.diagnostics.finalCost);
});
test('BA robust objective handles an outlier without silently dropping observations', () => {
  const input = perturbed(),
    landmarks = input.landmarks.map((p, i) =>
      i !== 0
        ? p
        : {
            ...p,
            observations: p.observations.map((o, j) =>
              j !== 2 ? o : { ...o, pixel: [o.pixel[0] + 40, o.pixel[1] - 20] as const },
            ),
          },
    );
  const output = bundleAdjust({ ...input, landmarks }, { iterations: 20, huberPx: 1 });
  assert.ok(output.diagnostics.finalCost < output.diagnostics.initialCost * 0.2);
  assert.equal(output.diagnostics.observations, 350);
  assert.equal(
    output.reconstruction.landmarks[0]!.observations[2]!.pixel[0],
    landmarks[0]!.observations[2]!.pixel[0],
  );
});
test('BA rejects invalid ownership, cheirality and scale gauges', () => {
  const input = perturbed();
  assert.throws(
    () =>
      bundleAdjust({
        ...input,
        landmarks: [
          input.landmarks[0]!,
          { ...input.landmarks[1]!, observations: input.landmarks[0]!.observations },
        ],
      }),
    /owned/,
  );
  assert.throws(() => bundleAdjust(input, { anchorViewId: 999 }), /gauge/);
  assert.throws(
    () =>
      bundleAdjust({
        ...input,
        views: input.views.map((v, i) => (i !== 1 ? v : { ...v, pose: input.views[0]!.pose })),
      }),
    /baseline/,
  );
  assert.throws(
    () =>
      bundleAdjust({
        ...input,
        landmarks: input.landmarks.map((p, i) =>
          i !== 0 ? p : { ...p, xyz: Float64Array.of(0, 0, -10) },
        ),
      }),
    /in front/,
  );
});

test('gauge normalization works when anchor is translated and rotated in the world', () => {
  const { reconstruction } = scene({ count: 30, views: 4, noise: 0.1 });
  // Translate every world point by D and replace t by t - R D; image observations remain unchanged.
  const D = Float64Array.of(7, -3, 2);
  const shifted: SparseReconstruction = {
    views: reconstruction.views.map((v) => ({
      ...v,
      pose: {
        R: v.pose.R,
        t: v.pose.t.map(
          (t, i) =>
            t -
            v.pose.R[i * 3]! * D[0]! -
            v.pose.R[i * 3 + 1]! * D[1]! -
            v.pose.R[i * 3 + 2]! * D[2]!,
        ),
      },
    })),
    landmarks: reconstruction.landmarks.map((p) => ({ ...p, xyz: p.xyz.map((x, i) => x + D[i]!) })),
  };
  const out = bundleAdjust(shifted, { iterations: 15, anchorViewId: 20, scaleViewId: 40 });
  assert.ok(out.diagnostics.finalCost <= out.diagnostics.initialCost);
  assert.deepEqual(
    out.reconstruction.views.find((v) => v.id === 20)!.pose,
    shifted.views.find((v) => v.id === 20)!.pose,
  );
  const a = cameraCenter(out.reconstruction.views.find((v) => v.id === 20)!.pose),
    b = cameraCenter(out.reconstruction.views.find((v) => v.id === 40)!.pose);
  assert.ok(Math.abs(Math.hypot(...a.map((v, i) => v - b[i]!)) - out.diagnostics.baseline) < 1e-9);
});

test('local BA holds boundary cameras fixed while decreasing the joint objective', () => {
  const input = perturbed(),
    out = bundleAdjust(input, { iterations: 15, variableViewIds: [30, 40, 50] });
  assert.ok(out.diagnostics.finalCost < out.diagnostics.initialCost);
  assert.deepEqual(out.reconstruction.views[0]!.pose, input.views[0]!.pose);
  assert.deepEqual(out.reconstruction.views[1]!.pose, input.views[1]!.pose);
  assert.throws(() => bundleAdjust(input, { variableViewIds: [10] }), /window/);
  assert.throws(() => bundleAdjust(input, { variableViewIds: [20] }), /scale camera/);
});

test('BA rejects an unobserved camera rather than optimizing a disconnected gauge', () => {
  const input = perturbed();
  const landmarks = input.landmarks.map((p) => ({
    ...p,
    observations: p.observations.filter((o) => o.viewId !== 50),
  }));
  assert.throws(() => bundleAdjust({ ...input, landmarks }), /connected observation graph/);
});
