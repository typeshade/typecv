import test from 'node:test';
import assert from 'node:assert/strict';
import { bundleAdjustGpu, ShadeDevice } from '../../dist/gpu.js';
import { bundleFixture } from './geometry-cases.mjs';

test('bounded BA checkpoints skip guarded no-op work without changing solver results', async () => {
  const input = bundleFixture({ outlier: true }),
    options = { iterations: 15, huberPx: 1 },
    results = [],
    schedules = [];
  for (const settings of [
    { boundedSubmissions: false },
    { boundedSubmissions: true, maxDispatchesPerSubmission: 1 },
    { boundedSubmissions: true, maxDispatchesPerSubmission: 16 },
  ]) {
    const device = await ShadeDevice.open({ backend: 'cpu', ...settings });
    const schedule = [],
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
      results.push(await bundleAdjustGpu(device, input, options));
    } finally {
      await device.dispose();
    }
    assert.equal(device.statistics.liveBuffers, 0);
    schedules.push(schedule);
  }
  const [unbounded, boundedSingle, boundedGrouped] = results;
  for (let i = 1; i < results.length; i++) {
    const bounded = results[i];
    assert.deepEqual(bounded.reconstruction, unbounded.reconstruction);
    assert.deepEqual(bounded.diagnostics, unbounded.diagnostics);
    assert.equal(bounded.cgIterations, unbounded.cgIterations);
    assert.equal(bounded.linearSolverFailures, unbounded.linearSolverFailures);
    assert.deepEqual(bounded.execution.cpuStages, unbounded.execution.cpuStages);
    assert.ok(
      bounded.execution.applicationReadbackBytes >=
        unbounded.execution.applicationReadbackBytes + 8,
    );
    assert.ok(schedules[i].includes('snapshotBundleActivity'));
    assert.ok(
      schedules[i].length < schedules[0].length,
      JSON.stringify({ unbounded: schedules[0].length, bounded: schedules[i].length }),
    );
    assert.ok(
      schedules[i].filter((name) => name === 'advanceBundleCg').length <
        schedules[0].filter((name) => name === 'advanceBundleCg').length,
    );
  }
  assert.deepEqual(boundedGrouped.reconstruction, boundedSingle.reconstruction);
  assert.deepEqual(boundedGrouped.diagnostics, boundedSingle.diagnostics);
  assert.equal(boundedGrouped.cgIterations, boundedSingle.cgIterations);
});
