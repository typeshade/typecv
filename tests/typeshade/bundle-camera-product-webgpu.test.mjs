import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('per-camera products keep resident and streamed BA identical across window sizes and scheduling', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { bundleFixture } = await import('/tests/typeshade/geometry-cases.mjs');
      const input = bundleFixture({ count: 80, views: 6 }),
        C = 6,
        P = 80,
        M = 480,
        rows = [];
      for (const shared of [false, true]) {
        const doubles =
          18 * C +
          3 * P +
          2 * M +
          (12 * C + 3 * P) +
          (30 * C + 3 * P + (shared ? 5 + C : 0)) +
          26 +
          3 * Math.ceil(M / 64) +
          (12 * C + 3 * P + (shared ? 27 : 26)) +
          78 * C +
          36 * C +
          (shared ? 6 * C + 6 : 0) +
          (shared ? 7 : 6) * 64 * C;
        const floor = doubles * 8 + 4 * (3 * M + P + 2 * C + 2) + 8;
        const runs = [];
        for (const bounded of [false, true])
          for (const capacity of [undefined, 64, 150]) {
            const workingBytes = capacity
              ? floor + capacity * (shared ? 376 : 336) + Math.ceil(capacity / C) * 104
              : 16 * 1024 ** 2;
            const device = await ShadeDevice.open({
              workingBytes,
              poolBytes: 0,
              boundedSubmissions: bounded,
            });
            try {
              const r = await bundleAdjustGpu(device, input, {
                iterations: 2,
                maxCgIterations: 96,
                refineSharedFocal: shared,
                cameraProduct: 'workgroups',
                schurReduction: 'serial',
              });
              runs.push({
                bounded,
                capacity,
                accepted: r.diagnostics.acceptedSteps,
                failures: r.linearSolverFailures,
                streamed: r.execution.report.groups.some((g) =>
                  g.calls.includes('checkBundlePoints'),
                ),
                text: JSON.stringify([r.reconstruction, r.diagnostics, r.cgIterations]),
                live: device.statistics.liveBuffers,
              });
            } finally {
              await device.dispose();
            }
          }
        rows.push({ shared, runs });
      }
      return rows;
    },
    undefined,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  for (const { shared, runs } of result)
    for (const run of runs) {
      assert.equal(
        run.streamed,
        run.capacity !== undefined,
        JSON.stringify({ shared, ...run, text: undefined }),
      );
      assert.ok(run.accepted > 0);
      assert.equal(run.failures, 0);
      assert.equal(run.live, 0);
      assert.equal(
        run.text,
        runs[0].text,
        `shared ${shared}, bounded ${run.bounded}, capacity ${run.capacity}`,
      );
    }
});

test('per-camera dispatch splits respect the workgroup limit and bounded queue fences', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { ringBundleFixture } = await import('/tests/typeshade/geometry-cases.mjs');
      const input = ringBundleFixture({ points: 640, globalPoints: 16 }),
        rows = [];
      for (const shared of [false, true]) {
        const runs = [];
        for (const limit of [undefined, 128]) {
          const device = await ShadeDevice.open({
            workingBytes: 64 * 1024 ** 2,
            poolBytes: 0,
            boundedSubmissions: true,
          });
          if (limit) Object.defineProperty(device, 'maxWorkgroups', { value: limit });
          const plan = device.plan.bind(device);
          let maximum = 0;
          device.plan = async (...args) => {
            const report = await plan(...args);
            maximum = Math.max(maximum, report.dispatches);
            return report;
          };
          try {
            const r = await bundleAdjustGpu(device, input, {
              iterations: 1,
              maxCgIterations: 256,
              cameraProduct: 'workgroups',
              schurReduction: 'serial',
              refineSharedFocal: shared,
            });
            const calls = r.execution.report.groups.flatMap((g) => g.calls);
            runs.push({
              maximum,
              text: JSON.stringify([r.reconstruction, r.diagnostics, r.cgIterations]),
              camera: calls.filter((c) => c === 'bundleSchurCameraWorkgroup').length,
              advance: calls.filter((c) => c === 'advanceBundleCg').length,
              accepted: r.diagnostics.acceptedSteps,
              finish: calls.filter((c) => c === 'finishBundleSchurFocal').length,
              live: device.statistics.liveBuffers,
            });
          } finally {
            await device.dispose();
          }
        }
        rows.push({ shared, runs });
      }
      return rows;
    },
    undefined,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  for (const {
    shared,
    runs: [whole, split],
  } of result) {
    assert.equal(split.text, whole.text, `shared ${shared}`);
    assert.ok(whole.accepted > 0);
    assert.equal(whole.maximum, 1);
    assert.equal(split.maximum, 1);
    assert.equal(whole.camera, whole.advance);
    assert.equal(split.camera, 3 * split.advance);
    assert.equal(split.finish, shared ? split.advance : 0);
    assert.equal(whole.live, 0);
    assert.equal(split.live, 0);
  }
});

async function checkProductFlavor(integer) {
  const { buildComputeProbe } = await import('./compute-probe-build.mjs');
  const file = await buildComputeProbe('bundle-camera-product-probe');
  const { result, errors } = await inBrowser(
    async (options) => {
      const { checkCameraProductFlavors } = await import('/probe/product.js');
      return checkCameraProductFlavors(options);
    },
    { integer, hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
    { timeoutMs: 300000, serve: { '/probe/product.js': file } },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.validation, []);
  assert.equal(result.rows.length, 8);
  for (const row of result.rows)
    for (const value of row.errors) {
      assert.equal(value.flavor, value.owner);
      assert.ok(Number.isFinite(value.error) && value.error < 1e-9, JSON.stringify({ row, value }));
    }
}

test('per-camera products match the CPU in isolated resident, carry and inactive float kernels', async () =>
  checkProductFlavor(false));

test(
  'native integer-fp64 per-camera products match the CPU',
  {
    skip: process.env.TYPECV_HARDWARE_GPU !== '1' && process.env.TYPECV_INTEGER_BA_GPU !== '1',
  },
  async () => checkProductFlavor(true),
);
