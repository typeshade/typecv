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
