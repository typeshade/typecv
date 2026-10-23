import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('cooperative BA camera products, LM costs and geometry agree with scalar reduction', async () => {
  const { result, errors } = await inBrowser(
    async (hardware) => {
      const api = await import('/dist/gpu.js'),
        { bundleReductionCases } = await import('/tests/typeshade/bundle-reduction-cases.mjs'),
        adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter');
      const description = [
        adapter.info?.vendor,
        adapter.info?.architecture,
        adapter.info?.device,
        adapter.info?.description,
      ]
        .filter(Boolean)
        .join(' ');
      if (
        hardware &&
        (/swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter)
      )
        throw new Error(`Hardware requested, software adapter selected: ${description}`);
      const gpu = await adapter.requestDevice(),
        validation = [];
      gpu.addEventListener('uncapturederror', (event) => validation.push(event.error.message));
      try {
        return { description, rows: await bundleReductionCases(api, gpu), validation };
      } finally {
        gpu.destroy();
      }
    },
    process.env.TYPECV_HARDWARE_GPU === '1',
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.validation, []);
  console.log(JSON.stringify(result));
  for (const row of result.rows) {
    assert.ok(row.maxCoordinateError < 1e-5, `${row.name}: geometry ${row.maxCoordinateError}`);
    for (const run of row.runs) {
      assert.equal(run.tier, 'webgpu');
      assert.equal(run.cooperative, run.reduction === 'cooperative', row.name);
      assert.ok(run.diagnostics.acceptedSteps > 0, row.name);
      assert.equal(run.failures, 0, row.name);
      assert.equal(run.streamed, row.name.startsWith('streamed'));
      if (run.capture) {
        assert.ok(run.capture.checked > 0, row.name);
        if (row.expectIdle)
          assert.ok(run.capture.idle > 0, 'inactive PCG workgroups also complete');
        assert.ok(
          run.capture.maxRelativeError < 2e-10,
          `${row.name}: product ${run.capture.maxRelativeError}`,
        );
      }
      if (row.bounded) assert.equal(run.maxDispatches, 1, `${row.name}: bounded dispatches`);
    }
    assert.ok(Math.abs(row.runs[0].metrics.cost - row.runs[1].metrics.cost) < 1e-6, row.name);
    if (row.runs[0].capture) {
      const a = row.runs[0].capture.costs,
        b = row.runs[1].capture.costs;
      assert.equal(a.length, b.length);
      for (let i = 0; i < a.length; i++)
        assert.ok(Math.abs(a[i] - b[i]) < 1e-6 * Math.max(1, a[i]), `${row.name}: LM ${i}`);
    }
  }
  assert.equal(result.rows.filter((row) => row.cameras === 320).length, 2);
});

test('automatic BA reduction keeps short lists, CPU-style scheduling and dispatch limits on the serial path', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js'),
        { bundleFixture } = await import('/tests/typeshade/geometry-cases.mjs'),
        gpu = await (await navigator.gpu.requestAdapter()).requestDevice(),
        rows = [];
      try {
        for (const config of [
          { count: 131, floor: true, mode: 'auto', expected: true, width: 32 },
          { count: 40, floor: true, mode: 'auto', expected: false, width: 8 },
          { count: 131, floor: false, mode: 'auto', expected: false, width: 64 },
          { count: 40, floor: false, mode: 'cooperative', expected: true, width: 32 },
          { count: 131, floor: true, mode: 'cooperative', limit: 20, expected: false, width: 8 },
          { count: 300, floor: true, mode: 'auto', expected: true, width: 64 },
          { views: 8, count: 40, floor: true, mode: 'auto', expected: false, width: 16 },
          { views: 16, count: 40, floor: true, mode: 'auto', expected: false, width: 32 },
          { views: 24, count: 40, floor: true, mode: 'auto', expected: false, width: 64 },
        ]) {
          const device = await ShadeDevice.open({
            gpuDevice: gpu,
            poolBytes: 0,
            boundedSubmissions: true,
          });
          device.occupancyFloors = config.floor;
          if (config.limit) device.maxKernelElements = config.limit * 64;
          try {
            const run = await bundleAdjustGpu(
                device,
                bundleFixture({ count: config.count, views: config.views }),
                {
                  iterations: 1,
                  maxCgIterations: 2,
                  schurReduction: config.mode,
                },
              ),
              calls = run.execution.report.groups.flatMap((g) => g.calls);
            rows.push({
              ...config,
              assemblyShapes: run.execution.report.groups
                .filter((g) =>
                  g.calls.some(
                    (c) => c === 'bundleCameraRows' || c === 'bundleCameraRowsCooperative',
                  ),
                )
                .flatMap((g) => g.workgroupShapes ?? []),
              cooperative: calls.includes('bundleSchurCameraCooperative'),
              serial: calls.includes('bundleSchurCamera'),
              cooperativeAssembly: calls.includes('bundleCameraRowsCooperative'),
              serialAssembly: calls.includes('bundleCameraRows'),
            });
          } finally {
            await device.dispose();
          }
        }
        return rows;
      } finally {
        gpu.destroy();
      }
    },
    undefined,
    { timeoutMs: 120000 },
  );
  assert.deepEqual(errors, []);
  for (const row of result) {
    assert.ok(row.assemblyShapes.length > 0);
    for (const shape of row.assemblyShapes) assert.deepEqual(shape, [row.width, 1, 1]);
    assert.equal(row.cooperative, row.expected, JSON.stringify(row));
    assert.equal(row.serial, !row.expected, JSON.stringify(row));
    assert.equal(row.cooperativeAssembly, row.expected, JSON.stringify(row));
    assert.equal(row.serialAssembly, !row.expected, JSON.stringify(row));
  }
});
