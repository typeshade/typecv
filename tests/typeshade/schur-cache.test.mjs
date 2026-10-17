import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, bundleAdjustGpu } from '../../dist/gpu.js';
import { jointFocalFixture } from './joint-focal-cases.mjs';
import {
  captureSchurSystems,
  scalarSchur,
  scalarMatrixFree,
  validSpaceVector,
  multiplySchur,
} from './schur-cache-cases.mjs';

test('independent dense and matrix-free Schur operators agree on nonzero valid-space vectors', async () => {
  for (const settings of [
    { fixture: { noise: 0, focalScale: 0.75 }, options: {} },
    { fixture: { noise: 0.1, focalScale: 1 }, options: { anchorViewId: 20, scaleViewId: 40 } },
    { fixture: { noise: 0.2, radial: false }, options: { variableViewIds: [30, 40] } },
  ]) {
    const { input } = jointFocalFixture({ points: 12, views: 4, ...settings.fixture }),
      device = await ShadeDevice.open({ backend: 'cpu', solverCheckpoints: true, poolBytes: 0 }),
      captured = captureSchurSystems(device, input);
    try {
      await bundleAdjustGpu(device, input, {
        iterations: 1,
        refineSharedFocal: true,
        ...settings.options,
      });
      assert.equal(captured.systems.length, 1);
      const system = captured.systems[0],
        dense = scalarSchur(system);
      for (const seed of [0.3, 1.7, 4.1]) {
        const vector = validSpaceVector(system, seed),
          actual = multiplySchur(dense, vector),
          expected = scalarMatrixFree(system, vector),
          scale = Math.max(1, ...expected.map(Math.abs));
        assert.ok(vector.some((x) => x !== 0));
        assert.ok(Math.max(...actual.map((x, i) => Math.abs(x - expected[i]))) <= 1e-10 * scale);
      }
    } finally {
      await device.dispose();
    }
    assert.equal(device.statistics.liveBuffers, 0);
  }
});

test('Schur cache validates its boolean option and preserves explicit CPU execution', async () => {
  const { input } = jointFocalFixture({ points: 12, views: 4 }),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 }),
    options = { iterations: 2, refineSharedFocal: true };
  try {
    for (const schurCache of ['false', 'true', null, 0, 1, {}, []])
      await assert.rejects(
        async () => bundleAdjustGpu(device, input, { ...options, schurCache }),
        /schurCache/,
      );
    assert.equal(device.statistics.liveBuffers, 0);
    const ordinary = await bundleAdjustGpu(device, input, options),
      requested = await bundleAdjustGpu(device, input, { ...options, schurCache: true });
    assert.deepEqual(requested.reconstruction, ordinary.reconstruction);
    assert.deepEqual(requested.diagnostics, ordinary.diagnostics);
    assert.equal(requested.cgIterations, ordinary.cgIterations);
    assert.ok(
      requested.execution.report.groups
        .flatMap((g) => g.calls)
        .every((name) => !name.includes('Cached') && !name.startsWith('cacheBundle')),
    );
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBuffers, 0);
});
