import test from 'node:test';
import assert from 'node:assert/strict';
import { estimatePosePnPGpu, ShadeDevice, verifyImagePairGpu } from '../../dist/gpu.js';
import { poseFixture } from './geometry-cases.mjs';
import { radialPairFixture } from './initialization-cases.mjs';

async function runWithSchedule(boundedSubmissions, run, solverCheckpoints = boundedSubmissions) {
  const device = await ShadeDevice.open({
      backend: 'cpu',
      boundedSubmissions,
      solverCheckpoints,
      maxModelsPerBatch: 32,
      maxDispatchesPerSubmission: 64,
    }),
    schedule = [],
    planSteps = device.planSteps.bind(device);
  device.planSteps = (steps) =>
    planSteps(
      (function* () {
        for (const step of steps) {
          schedule.push(step[0]);
          yield step;
        }
      })(),
    );
  try {
    return { result: await run(device), schedule, statistics: device.statistics };
  } finally {
    await device.dispose();
    assert.equal(device.statistics.liveBuffers, 0);
  }
}

function countScheduled(schedule, ...names) {
  return schedule.filter((name) => names.includes(name)).length;
}

test('bounded PnP refinement checkpoints preserve the result and diagnostics', async () => {
  const fixture = poseFixture({ count: 97, noise: 0.3, outliers: 0.25 }),
    inputBefore = structuredClone(fixture),
    options = { iterations: 128, minIterations: 64, refinementIterations: 20 };
  const unbounded = await runWithSchedule(false, (device) =>
      estimatePosePnPGpu(device, fixture.world, fixture.pixels, fixture.camera, options),
    ),
    checkpointed = await runWithSchedule(
      false,
      (device) =>
        estimatePosePnPGpu(device, fixture.world, fixture.pixels, fixture.camera, options),
      true,
    ),
    bounded = await runWithSchedule(true, (device) =>
      estimatePosePnPGpu(device, fixture.world, fixture.pixels, fixture.camera, options),
    );
  assert.deepEqual(bounded.result.pose, unbounded.result.pose);
  assert.deepEqual(checkpointed.result.pose, unbounded.result.pose);
  assert.deepEqual(fixture, inputBefore, 'PnP must not mutate caller-owned inputs');
  assert.deepEqual(bounded.result.inliers, unbounded.result.inliers);
  assert.deepEqual(checkpointed.result.inliers, unbounded.result.inliers);
  for (const key of [
    'inlierCount',
    'meanErrorPx',
    'iterations',
    'candidatesEvaluated',
    'refinementAcceptedSteps',
    'refinementRejectedSteps',
  ]) {
    assert.equal(bounded.result[key], unbounded.result[key], key);
    assert.equal(checkpointed.result[key], unbounded.result[key], key);
  }
  assert.deepEqual(bounded.result.execution.cpuStages, unbounded.result.execution.cpuStages);
  assert.ok(bounded.schedule.includes('snapshotPoseActivity'));
  assert.ok(checkpointed.schedule.includes('snapshotPoseActivity'));
  assert.ok(
    countScheduled(checkpointed.schedule, 'poseNormalsOracle') <
      countScheduled(unbounded.schedule, 'poseNormalsOracle'),
    'unbounded checkpoint mode should skip inactive PnP refinement work',
  );
  assert.ok(
    countScheduled(bounded.schedule, 'poseNormalsOracle') <
      countScheduled(unbounded.schedule, 'poseNormalsOracle'),
    'bounded checkpoints should skip inactive PnP refinement work',
  );
  assert.ok(
    bounded.result.execution.applicationReadbackBytes >
      unbounded.result.execution.applicationReadbackBytes,
  );
});

test('bounded two-view recovery checkpoints preserve geometry and failure outcome', async () => {
  const fixture = radialPairFixture({ n: 65 }),
    options = {
      iterations: 32,
      minIterations: 32,
      homographyIterations: 32,
      threshold: 0.001,
      seed: 71,
      refinementIterations: 12,
    };
  for (const data of [fixture, radialPairFixture({ n: 65, planar: true })]) {
    const unbounded = await runWithSchedule(false, (device) =>
        verifyImagePairGpu(device, data.pixels, data.first, data.second, options),
      ),
      checkpointed = await runWithSchedule(
        false,
        (device) => verifyImagePairGpu(device, data.pixels, data.first, data.second, options),
        true,
      ),
      bounded = await runWithSchedule(true, (device) =>
        verifyImagePairGpu(device, data.pixels, data.first, data.second, options),
      );
    assert.equal(bounded.result.reason, unbounded.result.reason);
    assert.equal(checkpointed.result.reason, unbounded.result.reason);
    assert.equal(bounded.result.invertibleCount, unbounded.result.invertibleCount);
    assert.equal(checkpointed.result.invertibleCount, unbounded.result.invertibleCount);
    assert.equal(bounded.result.homographyInlierCount, unbounded.result.homographyInlierCount);
    assert.equal(checkpointed.result.homographyInlierCount, unbounded.result.homographyInlierCount);
    assert.deepEqual(bounded.result.inlierRows, unbounded.result.inlierRows);
    assert.deepEqual(checkpointed.result.inlierRows, unbounded.result.inlierRows);
    assert.deepEqual(bounded.result.geometry, unbounded.result.geometry);
    assert.deepEqual(checkpointed.result.geometry, unbounded.result.geometry);
    assert.ok(bounded.schedule.includes('snapshotTwoViewActivity'));
    assert.ok(checkpointed.schedule.includes('snapshotTwoViewActivity'));
    assert.ok(
      bounded.result.execution.applicationReadbackBytes >
        unbounded.result.execution.applicationReadbackBytes,
    );
  }
});

test('pair model generation stops launching RANSAC batches after the adaptive budget', async () => {
  const options = {
    iterations: 512,
    minIterations: 32,
    homographyIterations: 48,
    threshold: 0.001,
    seed: 71,
    refinementIterations: 4,
  };
  // The radial pair converges, so its adaptive budget stops the search short of the requested
  // iterations; the planar pair finds no geometry, so its search runs them all.
  for (const [input, converges] of [
    [radialPairFixture({ n: 65, noise: 0.05, outliers: false }), true],
    [radialPairFixture({ n: 65, planar: true, noise: 0, outliers: false }), false],
  ]) {
    const full = await runWithSchedule(
        false,
        (device) =>
          verifyImagePairGpu(device, input.pixels, input.first, input.second, {
            ...options,
            minIterations: options.iterations,
          }),
        false,
      ),
      ordinary = await runWithSchedule(
        false,
        (device) => verifyImagePairGpu(device, input.pixels, input.first, input.second, options),
        false,
      ),
      checkpointed = await runWithSchedule(
        false,
        (device) => verifyImagePairGpu(device, input.pixels, input.first, input.second, options),
        true,
      ),
      boundedWithoutSolverCheckpoints = await runWithSchedule(
        true,
        (device) => verifyImagePairGpu(device, input.pixels, input.first, input.second, options),
        false,
      );
    for (const key of [
      'reason',
      'invertibleCount',
      'homographyInlierCount',
      'homographyIterations',
    ])
      assert.equal(checkpointed.result[key], ordinary.result[key], key);
    assert.deepEqual(checkpointed.result.inlierRows, ordinary.result.inlierRows);
    assert.deepEqual(checkpointed.result.geometry, ordinary.result.geometry);
    assert.equal(boundedWithoutSolverCheckpoints.result.reason, ordinary.result.reason);
    assert.deepEqual(boundedWithoutSolverCheckpoints.result.inlierRows, ordinary.result.inlierRows);
    assert.deepEqual(boundedWithoutSolverCheckpoints.result.geometry, ordinary.result.geometry);
    assert.equal(
      checkpointed.result.execution.hypothesisBatches,
      ordinary.result.execution.hypothesisBatches,
    );
    assert.equal(
      checkpointed.result.execution.hypothesesScored,
      ordinary.result.execution.hypothesesScored,
    );
    // The model generation is chunked on every device (the time budget), so the adaptive budget
    // is read between the batches with or without solver checkpoints, and both stop launching
    // batches once the budget is met.
    for (const run of [ordinary, checkpointed, boundedWithoutSolverCheckpoints])
      assert.ok(run.schedule.includes('snapshotPairModelBudget'));
    assert.equal(
      countScheduled(checkpointed.schedule, 'scorePairModelsOracle'),
      countScheduled(ordinary.schedule, 'scorePairModelsOracle'),
    );
    const counts = {
      full: countScheduled(full.schedule, 'scorePairModelsOracle'),
      ordinary: countScheduled(ordinary.schedule, 'scorePairModelsOracle'),
      batches: ordinary.result.execution.hypothesisBatches,
    };
    assert.equal(ordinary.result.reason === undefined, converges, JSON.stringify(counts));
    if (converges) {
      assert.ok(counts.ordinary < counts.full, JSON.stringify(counts));
      assert.ok(counts.batches < Math.ceil(options.iterations / 32), JSON.stringify(counts));
    } else assert.equal(counts.ordinary, counts.full, JSON.stringify(counts));
  }
});

test('bounded pair allocation includes model progress even without solver checkpoints', async () => {
  const fixture = radialPairFixture({ n: 65 }),
    blockCount = Math.ceil(65 / 32),
    fixedBytes =
      72 * 65 +
      200 +
      (564 * 65 + 964 + 352 * blockCount) +
      (108 * 65 + 16 + Math.ceil(65 / 64) * 8) +
      4,
    perModelBytes = 148 + Math.ceil(65 / 64) * 16 + 52 + 98 * 8,
    workingBytes = fixedBytes + 2 * perModelBytes - 1,
    device = await ShadeDevice.open({
      backend: 'cpu',
      boundedSubmissions: true,
      solverCheckpoints: false,
      maxModelsPerBatch: 2,
      workingBytes,
    });
  try {
    const result = await verifyImagePairGpu(device, fixture.pixels, fixture.first, fixture.second, {
      batchSize: 2,
      iterations: 32,
      minIterations: 32,
      homographyIterations: 32,
      threshold: 0.001,
      seed: 71,
      refinementIterations: 4,
    });
    assert.equal(result.execution.hypothesisBatches, 32);
  } finally {
    await device.dispose();
  }
});
