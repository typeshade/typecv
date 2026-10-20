import test from 'node:test';
import assert from 'node:assert/strict';
import { bundleAdjust } from '../src/reconstruction/bundle-adjustment.ts';
import {
  executeReconstructionJob,
  globalBundleDue,
  incrementalSteps,
  type BundleSchedule,
  type IncrementalOptions,
} from '../src/reconstruction/mapper.ts';
import { updatePose } from '../src/geometry/pose.ts';
import { scene } from './scene.ts';

const colmap = { globalViewRatio: 1.1, globalPointRatio: 1.1, globalFrequency: 0 };

test('a global BA is due when the views or the points grew by the ratio since the last one', () => {
  const last = { views: 20, points: 1000 };
  // Below both ratios: a local BA.
  assert.equal(globalBundleDue(21, 1099, last, colmap), false);
  // 22 >= 20 * 1.1: the view ratio.
  assert.equal(globalBundleDue(22, 1000, last, colmap), true);
  // 1100 >= 1000 * 1.1: the point ratio, with the view count unchanged.
  assert.equal(globalBundleDue(21, 1100, last, colmap), true);
  // Infinity turns a ratio off.
  const viewsOnly = { ...colmap, globalPointRatio: Infinity };
  assert.equal(globalBundleDue(21, 5000, last, viewsOnly), false);
  assert.equal(globalBundleDue(22, 5000, last, viewsOnly), true);
  // The optional frequency adds the multiples of its count.
  const frequency = { globalViewRatio: Infinity, globalPointRatio: Infinity, globalFrequency: 3 };
  assert.deepEqual(
    [3, 4, 5, 6, 7, 8, 9].filter((views) => globalBundleDue(views, 0, last, frequency)),
    [3, 6, 9],
  );
  // A small map grows past the ratio at each registration.
  assert.equal(globalBundleDue(3, 10, { views: 2, points: 10 }, colmap), true);
});

/** Run the mapper and record, for each BA, the registered view count and whether it is global. */
function schedule(options: IncrementalOptions, carried?: BundleSchedule) {
  const { inputs, pairs } = scene({ count: 90, views: 12, noise: 0.1 });
  const machine = incrementalSteps(
    inputs,
    pairs,
    { seed: 42, ransacIterations: 200, bundleIterations: 2, ...options },
    undefined,
    carried,
  );
  const passes: { views: number; global: boolean }[] = [];
  let registered = 2,
    step = machine.next();
  while (!step.done) {
    const job = step.value;
    let value: unknown, error: unknown;
    try {
      value = executeReconstructionJob(job);
    } catch (caught) {
      error = caught;
    }
    if (job.kind === 'pnp' && !error) registered++;
    if (job.kind === 'bundle')
      passes.push({ views: registered, global: job.args[1]?.variableViewIds === undefined });
    step = error ? machine.throw(error) : machine.next(value);
  }
  assert.equal(step.value.reconstruction.views.length, 12);
  return passes;
}

test('the mapper runs a global BA on model growth and a local BA after every other registration', () => {
  const carried = { views: 0, points: 0 };
  // Views only: from 2 views a global BA is due at 3, 5 (>= 4.5), 8 (>= 7.5) and 12 (>= 12).
  const passes = schedule(
    { globalBundleViewRatio: 1.5, globalBundlePointRatio: Infinity },
    carried,
  );
  assert.deepEqual(
    passes.map((p) => `${p.views}${p.global ? 'G' : 'L'}`),
    ['2G', '3G', '4L', '5G', '6L', '7L', '8G', '9L', '10L', '11L', '12G', '12G'],
  );
  // The first pass adjusts the initial pair; the last is the final global BA, which always runs.
  assert.deepEqual(passes.at(-1), { views: 12, global: true });
  // The carried schedule holds the size at the last global BA.
  assert.equal(carried.views, 12);
  assert.ok(carried.points > 0);
});

test('the frequency alone keeps the earlier every-third-view schedule', () => {
  const passes = schedule({
    globalBundleViewRatio: Infinity,
    globalBundlePointRatio: Infinity,
    globalBundleFrequency: 3,
  });
  assert.deepEqual(
    passes.filter((p) => p.global).map((p) => p.views),
    [2, 3, 6, 9, 12, 12],
  );
});

test('the default schedule keeps a local BA between global passes once the model is large', () => {
  const passes = schedule({});
  // Every registration has one BA, plus the initial and the final pass.
  assert.equal(passes.length, 12);
  assert.ok(passes.some((p) => !p.global));
  assert.ok(passes.at(-1)!.global);
});

test('mapper schedule options are validated', () => {
  const { inputs, pairs } = scene({ count: 30, views: 3 });
  for (const options of [
    { globalBundleViewRatio: 1 },
    { globalBundlePointRatio: 0.5 },
    { globalBundleViewRatio: Number.NaN },
    { globalBundleFrequency: 0 },
    { bundleStallTolerance: -1 },
    { bundleStallTolerance: 1 },
  ])
    assert.throws(() => incrementalSteps(inputs, pairs, options).next(), RangeError);
});

test('a stall tolerance ends the LM loop when an accepted step lowers the cost too little', () => {
  const { reconstruction } = scene({ count: 60, views: 4, noise: 0.5 });
  const perturbed = {
    ...reconstruction,
    views: reconstruction.views.map((v, i) =>
      i < 2 ? v : { ...v, pose: updatePose(v.pose, Float64Array.of(0.002, 0, -0.001, 0.02, 0, 0)) },
    ),
  };
  const full = bundleAdjust(perturbed, { iterations: 10 }),
    stalled = bundleAdjust(perturbed, { iterations: 10, stallTolerance: 0.5 });
  assert.ok(full.diagnostics.acceptedSteps > stalled.diagnostics.acceptedSteps);
  assert.ok(stalled.diagnostics.acceptedSteps >= 1);
  assert.ok(stalled.diagnostics.finalCost < stalled.diagnostics.initialCost);
  assert.throws(() => bundleAdjust(perturbed, { stallTolerance: 1 }), RangeError);
});
