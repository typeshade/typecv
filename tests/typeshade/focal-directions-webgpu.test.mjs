import test from 'node:test';
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { directionObservationFixture } from '../fixtures/direction-observations.ts';
import { compareSeedGauge } from '../../scripts/geometry-comparison.mjs';
import { inBrowser } from './browser.mjs';

test('WebGPU automatic focal calibration preserves mixed-direction depth ambiguity', async () => {
  const f = directionObservationFixture({ noisePx: 0.2 });
  for (const v of f.input.images) v.camera = { ...v.camera, fx: 512, fy: 512 };
  const input = JSON.parse(JSON.stringify(f.input, (_, v) => (ArrayBuffer.isView(v) ? [...v] : v))),
    started = performance.now(),
    { result: r, errors } = await inBrowser(
      async (input) => {
        const api = await import('/dist/gpu.js'),
          { runFocalDirectionCase } = await import('/tests/typeshade/focal-directions-cases.mjs'),
          device = await api.ShadeDevice.open({ poolBytes: 0 });
        try {
          return await runFocalDirectionCase(api, device, input);
        } finally {
          await device.dispose();
        }
      },
      input,
      { timeoutMs: 180_000 },
    );
  assert.deepEqual(errors, []);
  const q = compareSeedGauge(
    { views: f.truth.views, points: f.truth.finitePoints },
    r.model,
    r.initialPair,
  );
  assert.equal(r.backend, 'webgpu');
  assert.equal(r.model.views.length, 5);
  assert.equal(r.model.landmarks.length, 60);
  assert.ok(r.model.landmarks.every((p) => p.observations.every((o) => o.featureId < 60)));
  assert.equal(r.pending, 60);
  assert.equal(r.focalSearch.scoring, 'directional-bic');
  assert.ok(r.model.views.every((v) => Math.abs(v.camera.fx / 480 - 1) < 0.02));
  assert.ok(q.maxRotationErrorDeg < 0.2);
  assert.ok(q.pointRmse < 0.1);
  assert.ok(r.cpuStages.some((s) => s.includes('bearing seed/selection')));
  assert.equal(r.liveBuffers, 0);
  assert.equal(r.liveBytes, 0);
  if (process.env.TYPECV_FOCAL_DIRECTION_MAP_GPU_REPORT)
    await writeFile(
      process.env.TYPECV_FOCAL_DIRECTION_MAP_GPU_REPORT,
      JSON.stringify(
        {
          createdAt: new Date().toISOString(),
          totalMs: performance.now() - started,
          protocol:
            'Geometry only; supplied feature identities; unknown focal; truth only in external evaluator. Includes browser startup/compilation, upload, geometry, synchronization and readback; no RGB decode or extraction. Physical GPU not measured.',
          quality: q,
          result: { ...r, model: undefined },
        },
        null,
        2,
      ) + '\n',
    );
});
