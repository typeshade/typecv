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

async function checkProductFlavor(integer, specializations = {}) {
  const { buildComputeProbe } = await import('./compute-probe-build.mjs');
  const file = await buildComputeProbe('bundle-camera-product-probe');
  const { result, errors } = await inBrowser(
    async (options) => {
      const { checkCameraProductFlavors } = await import('/probe/product.js');
      return checkCameraProductFlavors(options);
    },
    { integer, hardware: process.env.TYPECV_HARDWARE_GPU === '1', ...specializations },
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

test('all per-camera widths and observation tiles match CPU products through nonempty streamed windows', async () =>
  checkProductFlavor(false, { sizes: [32, 64, 128, 256], tiles: [undefined, 137] }));

async function checkBaCameraWidths(widths) {
  const { result, errors } = await inBrowser(
    async ({ widths }) => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { bundleFixture } = await import('/tests/typeshade/geometry-cases.mjs');
      const C = 4,
        P = 193,
        M = C * P,
        input = bundleFixture({ count: P, views: C }),
        rows = [];
      const policyDevice = await ShadeDevice.open({ poolBytes: 0 });
      const occupancyFloors = policyDevice.occupancyFloors;
      await policyDevice.dispose();
      for (const shared of [false, true])
        for (const width of widths)
          for (const tile of [undefined, 73]) {
            const selected = width === 'auto' ? (occupancyFloors ? 64 : 32) : width;
            // Auto targets four observations per lane, capped by each task's observation count.
            const actualWidth = width === 'auto' && tile ? 32 : selected;
            const tasks = C * (tile ? Math.ceil(P / tile) : 1);
            const taskWords = tile ? C + 1 + 3 * tasks : 0;
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
              (shared ? 7 : 6) * actualWidth * tasks +
              (tile ? (shared ? 7 : 6) * tasks : 0);
            const floor = doubles * 8 + 4 * (3 * M + P + 2 * C + 2) + 8 + taskWords * 4;
            const runs = [];
            for (const streamed of [false, true]) {
              const device = await ShadeDevice.open({
                workingBytes: streamed
                  ? floor + 150 * (shared ? 376 : 336) + Math.ceil(150 / C) * 104
                  : 16 * 1024 ** 2,
                poolBytes: 0,
                boundedSubmissions: streamed,
              });
              // Each segmented product needs three separate dispatches when the limit is four.
              Object.defineProperty(device, 'maxWorkgroups', { value: 4 });
              const plan = device.plan.bind(device);
              let maximum = 0;
              device.plan = async (...args) => {
                const report = await plan(...args);
                maximum = Math.max(maximum, report.dispatches);
                return report;
              };
              try {
                const options = {
                  iterations: 1,
                  maxCgIterations: 64,
                  refineSharedFocal: shared,
                  cameraProduct: 'workgroups',
                  schurReduction: 'serial',
                  cameraProductWorkgroupSize: width,
                  cameraProductTileSize: tile,
                };
                const r = await bundleAdjustGpu(device, input, options);
                const groups = r.execution.report.groups;
                const kernel = tile ? 'bundleSchurCameraSegments' : 'bundleSchurCameraWorkgroup';
                const shapes = groups.flatMap((g) =>
                  g.calls.flatMap((call, i) =>
                    call === kernel ? [g.workgroupShapes?.[i]?.[0]] : [],
                  ),
                );
                runs.push({
                  streamed,
                  maximum,
                  accepted: r.diagnostics.acceptedSteps,
                  failures: r.linearSolverFailures,
                  live: device.statistics.liveBuffers,
                  usesWindows: groups.some((g) => g.calls.includes('checkBundlePoints')),
                  shapes,
                  text: JSON.stringify([r.reconstruction, r.diagnostics, r.cgIterations]),
                });
                if (streamed && width === 256 && tile) {
                  Object.defineProperty(device, 'workingBytes', { value: floor - 1 });
                  let rejected = false;
                  try {
                    await bundleAdjustGpu(device, input, options);
                  } catch (e) {
                    rejected = /floor|budget/i.test(e.message);
                  }
                  if (!rejected || device.statistics.liveBuffers)
                    throw new Error('Tiled lane storage escaped residency admission');
                }
              } finally {
                await device.dispose();
              }
            }
            rows.push({ shared, width, tile, actualWidth, runs });
          }
      return rows;
    },
    { widths },
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.length, widths.length * 4);
  for (const row of result) {
    const [resident, streamed] = row.runs;
    assert.equal(resident.text, streamed.text, JSON.stringify({ ...row, runs: undefined }));
    assert.equal(streamed.maximum, 1);
    for (const run of row.runs) {
      assert.equal(run.usesWindows, run.streamed);
      assert.ok(run.accepted > 0);
      assert.equal(run.failures, 0);
      assert.equal(run.live, 0);
      assert.ok(run.shapes.length > 0);
      assert.ok(
        run.shapes.every((s) => s === row.actualWidth),
        JSON.stringify({ row: { ...row, runs: undefined }, shapes: run.shapes }),
      );
    }
  }
}

test('camera widths and tiled products preserve streamed BA, bounded fences and resource admission', async () =>
  checkBaCameraWidths([32, 64, 128, 256]));
test('automatic camera width retains streamed results and bounded fences', async () =>
  checkBaCameraWidths(['auto']));

test('specialized segmented products certify cached shared-focal solves and changing widths on one owner', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { bundleFixture } = await import('/tests/typeshade/geometry-cases.mjs');
      const input = bundleFixture({ count: 193, views: 4 });
      const device = await ShadeDevice.open({ poolBytes: 0 });
      try {
        const rows = [];
        for (const width of [32, 128, 64]) {
          const r = await bundleAdjustGpu(device, input, {
            iterations: 2,
            maxCgIterations: 96,
            refineSharedFocal: true,
            schurCache: true,
            schurReduction: 'serial',
            cameraProduct: 'workgroups',
            cameraProductWorkgroupSize: width,
            cameraProductTileSize: 73,
          });
          const calls = r.execution.report.groups.flatMap((g) => g.calls);
          rows.push({
            width,
            cost: r.diagnostics.finalCost,
            accepted: r.diagnostics.acceptedSteps,
            failures: r.linearSolverFailures,
            live: device.statistics.liveBuffers,
            cached: calls.includes('bundleCachedSchurProduct'),
            certified: calls.includes('certifyCachedBundleCg'),
            segmented: calls.includes('bundleSchurCameraSegments'),
            finished: calls.includes('finishBundleSchurCameraSegments'),
          });
        }
        // Width changes must prepare each pipeline before a timed matrix-free chunk as well.
        for (const width of [32, 128, 64]) {
          const r = await bundleAdjustGpu(device, input, {
            iterations: 1,
            maxCgIterations: 64,
            cameraProduct: 'workgroups',
            cameraProductWorkgroupSize: width,
            cameraProductTileSize: 73,
            schurReduction: 'serial',
          });
          const keys = Object.keys(r.execution.report.chunks ?? {});
          rows.push({
            width,
            cost: r.diagnostics.finalCost,
            accepted: r.diagnostics.acceptedSteps,
            failures: r.linearSolverFailures,
            live: device.statistics.liveBuffers,
            keys,
          });
        }
        return rows;
      } finally {
        await device.dispose();
      }
    },
    undefined,
    { timeoutMs: 300000 },
  );
  assert.deepEqual(errors, []);
  for (const row of result) {
    assert.ok(row.accepted > 0);
    assert.equal(row.failures, 0);
    assert.equal(row.live, 0);
    if (row.keys)
      assert.ok(
        row.keys.some((key) => key.includes(`Camera${row.width}Tile73`)),
        JSON.stringify(row),
      );
    else {
      assert.ok(row.cached && row.certified && row.segmented && row.finished, JSON.stringify(row));
      assert.ok(Math.abs(row.cost - result[0].cost) < 1e-7 * Math.max(1, Math.abs(result[0].cost)));
    }
  }
});
