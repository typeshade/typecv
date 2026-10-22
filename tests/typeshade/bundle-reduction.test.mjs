import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, bundleAdjustGpu } from '../../dist/gpu.js';
import { bundleFixture } from './geometry-cases.mjs';

test('CPU BA retains the scalar oracle when cooperative reduction is requested', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const shared of [false, true]) {
      const input = bundleFixture(),
        results = [];
      for (const schurReduction of ['serial', 'cooperative'])
        results.push(
          await bundleAdjustGpu(device, input, {
            iterations: 2,
            refineSharedFocal: shared,
            schurReduction,
          }),
        );
      assert.deepEqual(results[1].reconstruction, results[0].reconstruction);
      assert.deepEqual(results[1].diagnostics, results[0].diagnostics);
      assert.equal(results[1].execution.report.tier, 'cpu');
    }
    assert.throws(
      () => bundleAdjustGpu(device, bundleFixture(), { schurReduction: 'typo' }),
      /Schur reduction mode/,
    );
    assert.equal(device.statistics.liveBuffers, 0);
  } finally {
    await device.dispose();
  }
});
