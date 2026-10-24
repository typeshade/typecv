import test from 'node:test';
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { bundleDirectionsFixture } from '../fixtures/bundle-directions.ts';
import { inBrowser } from './browser.mjs';

test('WebGPU bearing-only BA keeps depth unresolved while refining poses and shared focal', async () => {
  const fixtures = JSON.parse(
    JSON.stringify([bundleDirectionsFixture(), bundleDirectionsFixture(1.06)], (_, v) =>
      ArrayBuffer.isView(v) ? [...v] : v,
    ),
  );
  const { result, errors } = await inBrowser(
    async (fixtures) => {
      const api = await import('/dist/gpu.js'),
        { runBundleDirectionCases } = await import('/tests/typeshade/bundle-directions-cases.mjs'),
        device = await api.ShadeDevice.open({ poolBytes: 0 });
      try {
        return await runBundleDirectionCases(api, device, fixtures);
      } finally {
        await device.dispose();
      }
    },
    fixtures,
    { timeoutMs: 180_000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.backend, 'webgpu');
  for (const r of result.rows) {
    assert.ok(r.diagnostics.finalRmsePx < 0.01, `${r.diagnostics.finalRmsePx}`);
    assert.equal(r.diagnostics.directionalObservations, 300);
    assert.equal(r.diagnostics.observations, 600);
    assert.equal(r.finiteIds.length, 60);
    assert.ok(r.finiteIds.every((id) => id < 60));
    assert.equal(r.directionalIds.length, 60);
    assert.ok(r.directionalIds.every((id) => id >= 60));
    assert.ok(r.norms.every((n) => Math.abs(n - 1) < 1e-6));
    assert.ok(r.focals.every((f) => Math.abs(f / 480 - 1) < 0.005));
    assert.ok(r.pointRmse < 0.01, `point error ${r.pointRmse}`);
    assert.ok(r.rotationMaxDeg < 0.01, `rotation error ${r.rotationMaxDeg} degrees`);
    assert.deepEqual(r.cpuStages, []);
    assert.equal(r.unchanged, true);
    assert.equal(r.liveBytes, 0);
    assert.equal(r.liveBuffers, 0);
  }
  if (process.env.TYPECV_FOCAL_DIRECTION_GPU_REPORT)
    await writeFile(
      process.env.TYPECV_FOCAL_DIRECTION_GPU_REPORT,
      JSON.stringify(result, null, 2) + '\n',
    );
});
