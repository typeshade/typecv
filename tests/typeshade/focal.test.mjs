import test from 'node:test';
import assert from 'node:assert/strict';
import { bundleAdjust } from '../../dist/index.js';
import { bundleAdjustGpu, ShadeDevice } from '../../dist/gpu.js';
import { bundleFixture, bundleMetrics } from './geometry-cases.mjs';

test('batched focal scoring matches CPU fine search and improves over the old coarse grid', async () => {
  const model = bundleFixture({ outlier: true });
  for (const view of model.views) {
    view.camera.fx /= 1.09;
    view.camera.fy /= 1.09;
  }
  const options = { iterations: 0, huberPx: 1, refineSharedFocal: true };
  const expected = bundleAdjust(model, options);
  const device = await ShadeDevice.open({ backend: 'cpu', solverCheckpoints: true });
  try {
    const actual = await bundleAdjustGpu(device, model, options);
    assert.deepEqual(
      actual.reconstruction.views.map((v) => v.camera),
      expected.reconstruction.views.map((v) => v.camera),
    );
    assert.ok(Math.abs(actual.diagnostics.finalCost - expected.diagnostics.finalCost) < 1e-7);
    let coarseCost = Infinity;
    for (let i = -4; i <= 4; i++) {
      const scale = Math.exp((i * Math.log(2)) / 4);
      coarseCost = Math.min(
        coarseCost,
        bundleMetrics(
          {
            ...model,
            views: model.views.map((v) => ({
              ...v,
              camera: { ...v.camera, fx: v.camera.fx * scale, fy: v.camera.fy * scale },
            })),
          },
          1,
        ).cost,
      );
    }
    assert.ok(actual.diagnostics.finalCost < coarseCost);
    // Two scoring/reduction pairs, then one zero-iteration BA, rather than nine full BA calls.
    assert.ok(actual.execution.report.dispatches < 20);
    assert.deepEqual(actual.execution.cpuStages, []);
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBuffers, 0);
});

test('focal calibration preflights the combined resident allocation', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', workingBytes: 4096 });
  try {
    await assert.rejects(
      bundleAdjustGpu(device, bundleFixture(), { refineSharedFocal: true }),
      /budget/,
    );
    assert.equal(device.statistics.liveBuffers, 0);
  } finally {
    await device.dispose();
  }
});
