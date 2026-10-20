import test from 'node:test';
import assert from 'node:assert/strict';
import { bundleAdjust, validateReconstruction } from '../src/reconstruction/bundle-adjustment.ts';
import { extractLocalBundle, mergeLocalBundle } from '../src/reconstruction/local-bundle.ts';
import { incrementalSteps } from '../src/reconstruction/mapper.ts';
import { identityPose, projectPoint, updatePose } from '../src/geometry/pose.ts';
import type { SparseReconstruction } from '../src/reconstruction/types.ts';
import { scene } from './scene.ts';

function localScene(): SparseReconstruction {
  const { reconstruction, rng } = scene({ count: 48, views: 7, noise: 0.1 }),
    variable = new Set([40, 50]);
  return {
    views: reconstruction.views.map((v) =>
      variable.has(v.id)
        ? {
            ...v,
            pose: updatePose(v.pose, Float64Array.of(0.001, -0.002, 0.001, 0.02, -0.01, 0.01)),
          }
        : v,
    ),
    landmarks: reconstruction.landmarks.map((p, i) => ({
      ...p,
      xyz: i < 24 ? p.xyz.map((x) => x + (rng() - 0.5) * 0.08) : p.xyz,
      observations: p.observations.filter((o) =>
        (i < 24 ? [20, 30, 40, 50] : [10, 20, 60, 70]).includes(o.viewId),
      ),
    })),
  };
}

test('local extraction retains every selected observation and only connected observing cameras', () => {
  const input = localScene(),
    snapshot = structuredClone(input),
    local = extractLocalBundle(input, [40, 50]);
  assert.ok(local);
  validateReconstruction(local.reconstruction);
  assert.deepEqual(
    local.reconstruction.views.map((v) => v.id),
    [20, 30, 40, 50],
  );
  assert.equal(local.reconstruction.landmarks.length, 24);
  assert.equal(local.anchorViewId, 20);
  assert.equal(local.scaleViewId, 30);
  assert.ok(!local.variableViewIds.includes(local.anchorViewId));
  assert.ok(!local.variableViewIds.includes(local.scaleViewId));
  for (const p of local.reconstruction.landmarks) {
    assert.equal(p, input.landmarks[p.trackId]);
    assert.equal(p.observations.length, 4);
  }
  const result = bundleAdjust(local.reconstruction, {
    iterations: 0,
    variableViewIds: local.variableViewIds,
    anchorViewId: local.anchorViewId,
    scaleViewId: local.scaleViewId,
  });
  assert.equal(result.diagnostics.observations, 96);
  assert.deepEqual(input, snapshot);
});

test('local CPU BA improves the same objective and merging preserves unrelated geometry and observations', () => {
  const input = localScene(),
    snapshot = structuredClone(input),
    local = extractLocalBundle(input, [40, 50])!;
  const result = bundleAdjust(local.reconstruction, {
      iterations: 20,
      huberPx: 2,
      variableViewIds: local.variableViewIds,
      anchorViewId: local.anchorViewId,
      scaleViewId: local.scaleViewId,
    }),
    merged = mergeLocalBundle(input, local, result.reconstruction);
  assert.ok(result.diagnostics.acceptedSteps > 0);
  assert.ok(result.diagnostics.finalCost < result.diagnostics.initialCost);
  assert.equal(merged.views.length, input.views.length);
  assert.equal(merged.landmarks.length, input.landmarks.length);
  for (let i = 0; i < input.views.length; i++) {
    assert.equal(merged.views[i]!.camera, input.views[i]!.camera);
    if (![40, 50].includes(input.views[i]!.id)) assert.equal(merged.views[i], input.views[i]);
  }
  for (let i = 0; i < input.landmarks.length; i++) {
    assert.equal(merged.landmarks[i]!.observations, input.landmarks[i]!.observations);
    if (i >= 24) assert.equal(merged.landmarks[i], input.landmarks[i]);
  }
  validateReconstruction(merged);
  assert.deepEqual(input, snapshot);
  const views = new Map(merged.views.map((v) => [v.id, v]));
  let before = 0,
    after = 0;
  for (let i = 0; i < input.landmarks.length; i++)
    for (const o of input.landmarks[i]!.observations) {
      const oldView = input.views.find((v) => v.id === o.viewId)!,
        newView = views.get(o.viewId)!,
        a = projectPoint(oldView.camera, oldView.pose, input.landmarks[i]!.xyz)!,
        b = projectPoint(newView.camera, newView.pose, merged.landmarks[i]!.xyz)!;
      const huber = (e: number) => (e <= 2 ? e * e : 4 * e - 4);
      before += huber(Math.hypot(a[0] - o.pixel[0], a[1] - o.pixel[1]));
      after += huber(Math.hypot(b[0] - o.pixel[0], b[1] - o.pixel[1]));
    }
  assert.ok(after <= before);
});

test('merging accepts only selected coordinates and variable poses from a local result', () => {
  const input = localScene(),
    local = extractLocalBundle(input, [40, 50])!,
    changed: SparseReconstruction = {
      views: input.views.map((v) => ({
        ...v,
        camera: { ...v.camera, fx: v.camera.fx * 2 },
        pose: identityPose(),
      })),
      landmarks: input.landmarks.map((p) => ({
        ...p,
        xyz: Float64Array.of(1, 2, 3),
        observations: [],
      })),
    },
    merged = mergeLocalBundle(input, local, changed);
  for (let i = 0; i < input.views.length; i++) {
    const old = input.views[i]!,
      current = merged.views[i]!;
    assert.equal(current.camera, old.camera);
    assert.deepEqual(current.pose, [40, 50].includes(old.id) ? identityPose() : old.pose);
  }
  for (let i = 0; i < input.landmarks.length; i++) {
    assert.equal(merged.landmarks[i]!.observations, input.landmarks[i]!.observations);
    assert.deepEqual(
      merged.landmarks[i]!.xyz,
      i < 24 ? Float64Array.of(1, 2, 3) : input.landmarks[i]!.xyz,
    );
  }
});

test('unsafe local gauges, disconnected subsets and unconstrained cameras fall back', () => {
  const input = localScene();
  assert.equal(extractLocalBundle(input, []), undefined);
  assert.equal(extractLocalBundle(input, [40, 40]), undefined);
  assert.equal(extractLocalBundle(input, [40, 999]), undefined);
  // Include both landmark groups while leaving only view 10 fixed.
  const allViewIds = input.views.map((v) => v.id);
  assert.equal(extractLocalBundle(input, allViewIds.slice(1)), undefined);
  assert.equal(extractLocalBundle(input, allViewIds), undefined);
  const coincident = {
    ...input,
    views: input.views.map((v) => (v.id === 30 ? { ...v, pose: input.views[1]!.pose } : v)),
  };
  assert.equal(extractLocalBundle(coincident, [40, 50]), undefined);
  const disconnected = {
    ...input,
    landmarks: input.landmarks.map((p, i) => ({
      ...p,
      observations: p.observations.filter((o) =>
        i < 12 ? [20, 40].includes(o.viewId) : i < 24 ? [30, 50].includes(o.viewId) : true,
      ),
    })),
  };
  validateReconstruction(disconnected);
  assert.equal(extractLocalBundle(disconnected, [40, 50]), undefined);
});

test('mapper skips observation filtering only after a successfully extracted local problem', () => {
  // A scripted numerical driver isolates mapper control flow from numerical convergence.
  // Initial tracks connect the full map; a later local track need not retain both seed views.
  // In the unsafe script, tracks 0..11 triangulate only once view 40 is registered, with
  // observations from views 30 and 40 alone, and tracks 12..23 refuse view 40's observation.
  for (const subset of [false, true])
    for (const safe of [false, true]) {
      const { inputs, pairs } = scene({ count: 24, views: 4 }),
        machine = incrementalSteps(inputs, pairs, {
          initialPair: [10, 20],
          minPnPInliers: 6,
          minInitialPoints: 8,
          localBundleSize: 1,
          localBundleSubset: subset,
          // Every registration takes a local BA: no growth ratio or frequency makes one global.
          globalBundleViewRatio: Infinity,
          globalBundlePointRatio: Infinity,
          finalBundleIterations: 0,
        });
      let step = machine.next(),
        registered = 2,
        checked = 0;
      while (!step.done) {
        const job = step.value;
        let value: unknown;
        switch (job.kind) {
          case 'verifyPair': {
            const count = job.args[0].length / 4;
            value = {
              invertibleCount: count,
              homographyInlierCount: 0,
              inlierRows: Uint32Array.from({ length: count }, (_, i) => i),
              geometry: {
                R: identityPose().R,
                t: Float64Array.of(-1, 0, 0),
                inlierCount: count,
                positiveDepthCount: count,
                diagnostics: { medianParallaxDeg: 5 },
              },
            };
            break;
          }
          case 'pnp':
            registered++;
            value = { pose: { R: identityPose().R, t: Float64Array.of(-registered, 0, 0) } };
            break;
          case 'triangulation':
            value = job.observations.map((observations, i) => {
              const trackId = job.trackIds[i]!;
              if (!safe && trackId < 12 && registered < 4) return undefined;
              return {
                xyz: Float64Array.of(trackId * 0.02, 0, 5),
                inliers: Uint8Array.from(observations, (_, j) =>
                  !safe && trackId < 12 ? Number(j >= 2) : 1,
                ),
              };
            });
            break;
          case 'bundle': {
            value = bundleAdjust(job.args[0], { ...job.args[1], iterations: 0 });
            const local = job.args[1]?.variableViewIds !== undefined;
            step = machine.next(value);
            if (local) {
              const extracted = subset && (safe || registered === 3);
              assert.equal(
                !step.done && step.value.kind === 'observations' && step.value.stage === 'filter',
                !extracted,
              );
              if (registered === 4) checked++;
            }
            continue;
          }
          case 'observations':
            value = {
              masks: job.model.landmarks.map((p) =>
                Uint8Array.from(p.observations, (_, j) =>
                  job.stage === 'continuation' &&
                  !safe &&
                  registered === 4 &&
                  p.trackId >= 12 &&
                  j === p.observations.length - 1
                    ? 0
                    : 1,
                ),
              ),
              meanErrorPx: 0,
            };
            break;
        }
        step = machine.next(value);
      }
      assert.equal(checked, 1);
    }
});

test('mapper validates localBundleSubset values received from JSON options', () => {
  const { inputs, pairs } = scene({ count: 24, views: 2 });
  for (const value of ['false', 'true', 0, 1, null, [], {}]) {
    const options = JSON.parse(JSON.stringify({ localBundleSubset: value }));
    assert.throws(() => incrementalSteps(inputs, pairs, options).next(), /localBundleSubset/);
  }
  for (const localBundleSubset of [undefined, false, true])
    assert.doesNotThrow(() => incrementalSteps(inputs, pairs, { localBundleSubset }).next());
});
