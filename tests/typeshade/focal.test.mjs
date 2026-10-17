import test from 'node:test';
import assert from 'node:assert/strict';
import { bundleAdjust } from '../../dist/index.js';
import { bundleAdjustGpu, ShadeDevice } from '../../dist/gpu.js';
import { jointFocalFixture } from './joint-focal-cases.mjs';

const options = {
  iterations: 40,
  refineSharedFocal: true,
  maxCgIterations: 256,
  cgRelativeTolerance: 1e-10,
};
test('joint shared focal recovers independent nonplanar radial ground truth with poses and points', async () => {
  const { input, truth } = jointFocalFixture();
  const snapshot = JSON.stringify(input);
  const expected = bundleAdjust(input, options);
  const device = await ShadeDevice.open({ backend: 'cpu', solverCheckpoints: true });
  try {
    const actual = await bundleAdjustGpu(device, input, options);
    for (const result of [expected, actual]) {
      assert.ok(result.diagnostics.finalCost < 1e-7);
      assert.ok(result.diagnostics.finalRmsePx < 1e-4);
      const scale = result.reconstruction.views[0].camera.fx / input.views[0].camera.fx;
      assert.ok(
        Math.abs(result.reconstruction.views[0].camera.fx - truth.views[0].camera.fx) < 0.001,
      );
      for (let i = 0; i < input.views.length; i++) {
        const k = result.reconstruction.views[i].camera,
          initial = input.views[i].camera;
        assert.ok(Math.abs(k.fx / initial.fx - scale) < 1e-12);
        assert.ok(Math.abs(k.fy / initial.fy - scale) < 1e-12);
        assert.equal(k.k1, initial.k1);
        assert.equal(k.k2, initial.k2);
      }
      assert.deepEqual(result.reconstruction.views[0].pose, input.views[0].pose);
      assert.ok(result.diagnostics.initialCost > 50000);
    }
    assert.ok(Math.abs(actual.diagnostics.finalCost - expected.diagnostics.finalCost) < 1e-7);
    assert.deepEqual(actual.execution.cpuStages, []);
    assert.equal(JSON.stringify(input), snapshot);
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBuffers, 0);
});

test('joint focal zero iterations is identity and fixed poses still constrain the shared focal', async () => {
  const { input } = jointFocalFixture({ points: 20, views: 4 });
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const backend of ['cpu-reference', 'cpu-shader']) {
      const run =
        backend === 'cpu-reference'
          ? async (settings) => bundleAdjust(input, settings)
          : (settings) => bundleAdjustGpu(device, input, settings);
      const zero = await run({ ...options, iterations: 0 });
      assert.deepEqual(zero.reconstruction, input);
      assert.equal(zero.diagnostics.finalCost, zero.diagnostics.initialCost);
      assert.equal(zero.diagnostics.acceptedSteps, 0);
      const local = await run({ ...options, iterations: 15, variableViewIds: [] });
      assert.ok(local.diagnostics.finalCost < local.diagnostics.initialCost);
      assert.deepEqual(
        local.reconstruction.views.map((v) => v.pose),
        input.views.map((v) => v.pose),
      );
      assert.ok(local.reconstruction.views[0].camera.fx > input.views[0].camera.fx);
    }
  } finally {
    await device.dispose();
  }
});

test('joint focal preflights resident allocation and rejects failed solves without changing intrinsics', async () => {
  const { input } = jointFocalFixture({ points: 20, views: 4 });
  let device = await ShadeDevice.open({ backend: 'cpu', workingBytes: 4096 });
  try {
    await assert.rejects(async () => bundleAdjustGpu(device, input, options), /budget/);
    assert.equal(device.statistics.liveBuffers, 0);
  } finally {
    await device.dispose();
  }
  device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const out = await bundleAdjustGpu(device, input, {
      ...options,
      iterations: 1,
      maxCgIterations: 1,
      cgRelativeTolerance: 1e-12,
    });
    assert.equal(out.diagnostics.acceptedSteps, 0);
    assert.deepEqual(out.reconstruction, input);
  } finally {
    await device.dispose();
  }
});
