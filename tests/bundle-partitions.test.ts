import test from 'node:test';
import assert from 'node:assert/strict';
import { bundleAdjust } from '../src/reconstruction/bundle-adjustment.ts';
import {
  planConnectedBundles,
  mergeBundleRefinements,
} from '../src/reconstruction/local-bundle.ts';
import { incrementalSteps, executeReconstructionJob } from '../src/reconstruction/mapper.ts';
import { sceneSteps, HostSceneDriver } from '../src/reconstruction/scene-mapper.ts';
import type { PairMatches, SparseReconstruction } from '../src/reconstruction/types.ts';
import { scene } from './scene.ts';

function disconnected() {
  const fixture = scene({ count: 48, views: 7, noise: 0.15 });
  const model: SparseReconstruction = {
    views: fixture.reconstruction.views,
    landmarks: fixture.reconstruction.landmarks.map((p, i) => ({
      ...p,
      xyz: p.xyz.map((x) => x + 0.015),
      observations: p.observations.filter((o) =>
        (i < 24 ? [10, 20, 30] : [40, 50, 60]).includes(o.viewId),
      ),
    })),
  };
  const pairs: PairMatches[] = fixture.pairs.flatMap((p) => {
    const matches = model.landmarks.flatMap((point) => {
      const a = point.observations.find((o) => o.viewId === p.first),
        b = point.observations.find((o) => o.viewId === p.second);
      return a && b ? [a.featureId, b.featureId] : [];
    });
    return matches.length ? [{ ...p, matches: Uint32Array.from(matches) }] : [];
  });
  return { ...fixture, model, pairs };
}

test('connected BA retains the original problem and options', () => {
  const { reconstruction } = scene({ count: 24, views: 3 }),
    options = { iterations: 2, refineSharedFocal: true };
  const plan = planConnectedBundles(reconstruction, options);
  assert.equal(plan.partition, undefined);
  assert.equal(plan.problems.length, 1);
  assert.equal(plan.problems[0]!.reconstruction, reconstruction);
  assert.equal(plan.problems[0]!.options, options);
});

test('disconnected observations refine within carried gauges and preserve isolated cameras', () => {
  const { model } = disconnected(),
    snapshot = structuredClone(model);
  assert.throws(() => bundleAdjust(model), /one connected observation graph/);
  const plan = planConnectedBundles(model, {
    iterations: 8,
    anchorViewId: 10,
    scaleViewId: 20,
    refineSharedFocal: true,
  });
  assert.deepEqual(plan.partition?.components, [[10, 20, 30], [40, 50, 60], [70]]);
  assert.deepEqual(plan.partition?.deferredViewIds, [70]);
  assert.equal(plan.partition?.sharedFocalDeferred, true);
  assert.equal(plan.problems.length, 2);
  const results = plan.problems.map((p) => {
    assert.equal(p.options.refineSharedFocal, false);
    assert.ok(!p.options.variableViewIds!.includes(p.options.anchorViewId!));
    assert.ok(!p.options.variableViewIds!.includes(p.options.scaleViewId!));
    const result = bundleAdjust(p.reconstruction, p.options);
    assert.ok(result.diagnostics.finalCost < result.diagnostics.initialCost);
    return result.reconstruction;
  });
  const merged = mergeBundleRefinements(model, results);
  assert.equal(merged.views.length, 7);
  assert.equal(merged.landmarks.length, 48);
  assert.equal(merged.views[6], model.views[6]);
  for (const id of plan.partition!.fixedViewIds) {
    assert.deepEqual(
      merged.views.find((v) => v.id === id)!.pose,
      model.views.find((v) => v.id === id)!.pose,
    );
  }
  assert.deepEqual(
    merged.views.map((v) => v.camera),
    model.views.map((v) => v.camera),
  );
  assert.deepEqual(
    merged.landmarks.map((p) => p.observations),
    model.landmarks.map((p) => p.observations),
  );
  assert.deepEqual(model, snapshot);
});

test('a local partition without two fixed boundary poses is deferred, as is a zero-baseline component', () => {
  const { model } = disconnected();
  const local = planConnectedBundles(model, { variableViewIds: [30, 40, 50, 60] });
  assert.equal(local.problems.length, 1);
  assert.deepEqual(local.partition?.deferredViewIds, [40, 50, 60, 70]);
  const collapsed = {
    ...model,
    views: model.views.map((v) =>
      [40, 50, 60].includes(v.id) ? { ...v, pose: model.views[3]!.pose } : v,
    ),
  };
  const plan = planConnectedBundles(collapsed, {});
  assert.equal(plan.problems.length, 1);
  assert.deepEqual(plan.partition?.deferredViewIds, [40, 50, 60, 70]);
});

test('host and scene mapper resume a disconnected window through final BA without fabricating links', () => {
  const { inputs, model, pairs } = disconnected(),
    options = {
      initialPair: [10, 20] as const,
      ransacIterations: 128,
      bundleIterations: 2,
      finalBundleIterations: 4,
      refineSharedFocal: true,
      rotationTracking: false as const,
    };
  const machine = incrementalSteps(inputs, pairs, options, model);
  let next = machine.next();
  while (!next.done) next = machine.next(executeReconstructionJob(next.value));
  const driver = new HostSceneDriver(),
    sceneMachine = sceneSteps(inputs, pairs, options, model);
  let sceneNext = sceneMachine.next();
  while (!sceneNext.done) sceneNext = sceneMachine.next(driver.execute(sceneNext.value));
  for (const result of [next.value, sceneNext.value]) {
    assert.equal(result.reconstruction.views.length, 7);
    assert.equal(result.reconstruction.landmarks.length, 48);
    assert.equal(result.diagnostics.bundleAdjustments.length, 2);
    assert.deepEqual(result.diagnostics.bundlePartitions?.[0]?.deferredViewIds, [70]);
    assert.equal(result.diagnostics.bundlePartitions?.[0]?.sharedFocalDeferred, true);
    assert.deepEqual(result.reconstruction.views[6]!.pose, model.views[6]!.pose);
    assert.equal(result.unregistered.length, 0);
  }
  assert.deepEqual(next.value.diagnostics, sceneNext.value.diagnostics);
});
