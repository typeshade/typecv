import test from 'node:test';
import assert from 'node:assert/strict';
import { writeFile, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { inBrowser } from './browser.mjs';
import { assertDirectionCases, directionFixtures } from './direction-observations-fixture.mjs';

test('WebGPU preserves unresolved direction tracks and promotes reobserved far finite points', async () => {
  const fixtures = directionFixtures();
  const started = performance.now();
  const { result, errors } = await inBrowser(
    async (fixtures) => {
      const api = await import('/dist/gpu.js'),
        { runDirectionCases } = await import('/tests/typeshade/direction-observations-cases.mjs');
      const device = await api.ShadeDevice.open({ poolBytes: 0 });
      try {
        return {
          ...(await runDirectionCases(api, device, fixtures)),
          backend: device.backend,
          device: device.info,
        };
      } finally {
        await device.dispose();
      }
    },
    fixtures,
    { timeoutMs: 180000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.backend, 'webgpu');
  assertDirectionCases(result, fixtures);
  if (process.env.TYPECV_DIRECTION_GPU_REPORT) {
    const sourceHashes = Object.fromEntries(
      await Promise.all(
        [
          'src/geometry/gpu/recovery.shade.ts',
          'src/geometry/gpu/pair-estimation.ts',
          'src/geometry/two-view.ts',
          'src/geometry/refinement.ts',
          'tests/fixtures/direction-observations.ts',
        ].map(async (path) => [
          path,
          createHash('sha256')
            .update(await readFile(path))
            .digest('hex'),
        ]),
      ),
    );
    await writeFile(
      process.env.TYPECV_DIRECTION_GPU_REPORT,
      JSON.stringify(
        {
          createdAt: new Date().toISOString(),
          elapsedMs: performance.now() - started,
          protocol:
            'Synthetic calibrated keypoints and supplied identities; geometry/upload/readback only, no RGB decode. Includes browser startup and shader compilation. Other tests may run concurrently; not a speed comparison.',
          requestedHardware: process.env.TYPECV_HARDWARE_GPU === '1',
          physicalGpuMeasured: false,
          sourceHashes,
          result,
        },
        null,
        2,
      ) + '\n',
    );
  }
});
