import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/gpu.js';
import { bundleDirectionsFixture } from '../fixtures/bundle-directions.ts';
import { runBundleDirectionCases } from './bundle-directions-cases.mjs';

export function assertBundleDirections(result) {
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
}

for (const workingBytes of [64 * 1024 ** 2, 100_000])
  test(`TypeShade bearing-only BA preserves its finite/directional split (${workingBytes} bytes)`, async () => {
    const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0, workingBytes });
    try {
      const fixtures = [bundleDirectionsFixture(), bundleDirectionsFixture(1.06)],
        result = await runBundleDirectionCases(api, device, fixtures);
      assertBundleDirections(result);
      if (workingBytes === 100_000)
        for (const r of result.rows)
          assert.ok(r.execution.report.groups.some((g) => g.calls.includes('checkBundlePoints')));
    } finally {
      await device.dispose();
    }
  });
