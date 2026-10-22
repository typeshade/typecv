import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('parallel GPU camera rows and Schur rows match the scalar CPU BA oracle', async () => {
  const hardware = process.env.TYPECV_HARDWARE_GPU === '1';
  const { result, errors } = await inBrowser(
    async ({ hardware }) => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { bundleFixture } = await import('/tests/typeshade/geometry-cases.mjs');
      const cases = [
        { settings: {}, options: {} },
        { settings: { outlier: true }, options: {} },
        {
          settings: { shift: [7, -3, 2] },
          options: { anchorViewId: 20, scaleViewId: 40 },
        },
        { settings: {}, options: { variableViewIds: [30, 40] } },
      ];
      const run = async (device) => {
        const results = [];
        for (const { settings, options } of cases) {
          const input = bundleFixture(settings),
            runOptions = { iterations: 12, huberPx: 1, ...options };
          results.push(await bundleAdjustGpu(device, input, runOptions));
        }
        return results;
      };
      // This is the previous CPU working-set threshold; the GPU-only scratch must not inflate it.
      const cpuDevice = await ShadeDevice.open({ backend: 'cpu', workingBytes: 71736 });
      let cpu;
      try {
        cpu = await run(cpuDevice);
      } finally {
        await cpuDevice.dispose();
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter available');
      const gpu = await adapter.requestDevice(),
        gpuErrors = [];
      gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      const info = adapter.info,
        adapterDescription = [info?.vendor, info?.architecture, info?.device, info?.description]
          .filter(Boolean)
          .join(' '),
        software =
          /swiftshader|software|llvmpipe/i.test(adapterDescription) ||
          adapter.isFallbackAdapter === true;
      if (hardware && software)
        throw new Error(
          `Hardware requested, but a software adapter was selected: ${adapterDescription}`,
        );
      const gpuDevice = await ShadeDevice.open({ gpuDevice: gpu });
      let gpuResults;
      try {
        gpuResults = await run(gpuDevice);
      } finally {
        await gpuDevice.dispose();
      }
      const legacyDevice = await ShadeDevice.open({
        gpuDevice: gpu,
        boundedSubmissions: true,
        maxDispatchesPerSubmission: 16,
      });
      let legacyOutlier;
      try {
        legacyOutlier = await bundleAdjustGpu(legacyDevice, bundleFixture({ outlier: true }), {
          iterations: 12,
          huberPx: 1,
        });
      } finally {
        await legacyDevice.dispose();
      }
      if (gpuErrors.length) throw new Error(`WebGPU validation errors: ${gpuErrors.join('; ')}`);
      // C=4/P=40/M=160: the CPU working-set total is 71,736 bytes (the LM state holds 26
      // doubles and the metric partials 3 per block of 64 observations); the GPU camera
      // scratch adds exactly 1,152 bytes. Both sides of that boundary complete: the larger budget holds the whole problem, the smaller one streams.
      const exactBudgetDevice = await ShadeDevice.open({
        gpuDevice: gpu,
        workingBytes: 72888,
        poolBytes: 0,
      });
      let exactBudgetLive;
      try {
        await bundleAdjustGpu(exactBudgetDevice, bundleFixture(), { iterations: 0 });
      } finally {
        exactBudgetLive = exactBudgetDevice.statistics.liveBuffers;
        await exactBudgetDevice.dispose();
      }
      const shortBudgetDevice = await ShadeDevice.open({
        gpuDevice: gpu,
        workingBytes: 71736,
        poolBytes: 0,
      });
      let shortBudgetError = '',
        shortBudgetLive;
      try {
        await bundleAdjustGpu(shortBudgetDevice, bundleFixture(), { iterations: 0 });
      } catch (error) {
        shortBudgetError = String(error);
      } finally {
        shortBudgetLive = shortBudgetDevice.statistics.liveBuffers;
        await shortBudgetDevice.dispose();
      }
      // Force only the inverse input to be singular after rows are produced. This checks
      // that each failed column's diagonal marker survives finalization and rejects LM.
      const failureErrors = [];
      gpu.addEventListener('uncapturederror', (event) => failureErrors.push(event.error.message));
      const failureDevice = await ShadeDevice.open({
        gpuDevice: gpu,
        poolBytes: 0,
        solverCheckpoints: true,
      });
      let cameraScratch;
      const allocateF64 = failureDevice.allocateF64.bind(failureDevice),
        planSteps = failureDevice.planSteps.bind(failureDevice);
      failureDevice.allocateF64 = (length, values) => {
        const buffer = allocateF64(length, values);
        if (length === 36 * 4) cameraScratch = buffer;
        return buffer;
      };
      failureDevice.planSteps = (steps) =>
        planSteps(
          (function* () {
            for (const step of steps) {
              if (step[0] !== 'bundleCameraRows') {
                yield step;
                continue;
              }
              yield [
                step[0],
                step[1],
                async () => {
                  await step[2]?.();
                  if (!cameraScratch) throw new Error('Camera scratch was not captured');
                  await cameraScratch.write(new Float64Array(36 * 4));
                },
              ];
            }
          })(),
        );
      const failureInput = bundleFixture(),
        failureSnapshot = JSON.stringify(failureInput);
      let failureResult, failureLive;
      try {
        failureResult = await bundleAdjustGpu(failureDevice, failureInput, { iterations: 1 });
      } finally {
        failureLive = failureDevice.statistics.liveBuffers;
        await failureDevice.dispose();
      }
      if (failureErrors.length)
        throw new Error(`WebGPU inverse-failure validation errors: ${failureErrors.join('; ')}`);
      return {
        comparisons: cases.map(({ settings }, index) => {
          const cpuResult = cpu[index],
            gpuResult = gpuResults[index],
            coords = (result) => [
              ...result.reconstruction.views.flatMap((view) => [...view.pose.R, ...view.pose.t]),
              ...result.reconstruction.landmarks.flatMap((point) => [...point.xyz]),
            ],
            a = coords(cpuResult),
            b = coords(gpuResult);
          let maxCoordinateError = 0;
          for (let i = 0; i < a.length; i++)
            maxCoordinateError = Math.max(maxCoordinateError, Math.abs(a[i] - b[i]));
          return {
            settings,
            maxCoordinateError,
            cpuCost: cpuResult.diagnostics.finalCost,
            gpuCost: gpuResult.diagnostics.finalCost,
            cpuRmse: cpuResult.diagnostics.finalRmsePx,
            gpuRmse: gpuResult.diagnostics.finalRmsePx,
            dispatchGroups: gpuResult.execution.report.groups,
            chunks: gpuResult.execution.report.chunks,
            cpuDispatchGroups: cpuResult.execution.report.groups,
          };
        }),
        legacyOutlier: {
          cpu: cpu[1],
          gpu: legacyOutlier,
        },
        budget: { exactBudgetLive, shortBudgetError, shortBudgetLive },
        inverseFailure: {
          acceptedSteps: failureResult.diagnostics.acceptedSteps,
          rejectedSteps: failureResult.diagnostics.rejectedSteps,
          linearSolverFailures: failureResult.linearSolverFailures,
          unchanged: JSON.stringify(failureResult.reconstruction) === failureSnapshot,
          inputUnchanged: JSON.stringify(failureInput) === failureSnapshot,
          liveBuffers: failureLive,
        },
      };
    },
    { hardware },
  );
  assert.deepEqual(errors, []);
  for (const outcome of result.comparisons) {
    const gpuCalls = outcome.dispatchGroups.flatMap((group) => group.calls),
      cpuCalls = outcome.cpuDispatchGroups.flatMap((group) => group.calls);
    assert.ok(outcome.maxCoordinateError < 1e-7, JSON.stringify(outcome));
    assert.ok(Math.abs(outcome.cpuCost - outcome.gpuCost) < 1e-8, JSON.stringify(outcome));
    assert.ok(Math.abs(outcome.cpuRmse - outcome.gpuRmse) < 1e-8, JSON.stringify(outcome));
    assert.ok(gpuCalls.includes('bundleCameraRows'));
    assert.ok(gpuCalls.includes('invertBundleCameraColumns'));
    assert.ok(gpuCalls.includes('finalizeBundleCameraInverse'));
    // The PCG loop runs as chunks of the time budget, the per-iteration kernels on every tier.
    for (const kernel of ['bundleSchurPoint', 'bundleSchurCamera', 'advanceBundleCg'])
      assert.ok(gpuCalls.includes(kernel), kernel);
    assert.ok(!gpuCalls.includes('bundlePersistentCg'));
    assert.ok(outcome.chunks?.bundleCg?.chunks > 0, JSON.stringify(outcome.chunks));
    assert.ok(cpuCalls.includes('bundleCameraBlocks'));
  }
  const legacyCalls = result.legacyOutlier.gpu.execution.report.groups.flatMap(
      (group) => group.calls,
    ),
    legacyCoords = (run) => [
      ...run.reconstruction.views.flatMap((view) => [...view.pose.R, ...view.pose.t]),
      ...run.reconstruction.landmarks.flatMap((point) => [...point.xyz]),
    ],
    legacyCpuCoords = legacyCoords(result.legacyOutlier.cpu),
    legacyGpuCoords = legacyCoords(result.legacyOutlier.gpu);
  assert.ok(legacyCalls.includes('bundleSchurPoint'));
  assert.ok(legacyCalls.includes('bundleSchurCamera'));
  assert.ok(legacyCalls.includes('advanceBundleCg'));
  assert.ok(
    Math.max(...legacyCpuCoords.map((value, index) => Math.abs(value - legacyGpuCoords[index]))) <
      1e-7,
  );
  assert.equal(result.budget.exactBudgetLive, 0);
  // docs/gpu-budget.md: a budget below the whole problem no longer throws; the BA streams.
  assert.equal(result.budget.shortBudgetError, '');
  assert.equal(result.budget.shortBudgetLive, 0);
  assert.equal(result.inverseFailure.acceptedSteps, 0);
  assert.equal(result.inverseFailure.rejectedSteps, 1);
  assert.equal(result.inverseFailure.linearSolverFailures, 1);
  assert.equal(result.inverseFailure.unchanged, true);
  assert.equal(result.inverseFailure.inputUnchanged, true);
  assert.equal(result.inverseFailure.liveBuffers, 0);
});

test('chunked GPU BA is repeatable and matches the CPU oracle on a larger fixture', async () => {
  const hardware = process.env.TYPECV_HARDWARE_GPU === '1';
  const { result, errors } = await inBrowser(
    async ({ hardware }) => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { bundleFixture } = await import('/tests/typeshade/geometry-cases.mjs');
      const input = bundleFixture({ count: 256, views: 8 });
      const options = { iterations: 12, huberPx: 1 },
        fallbackOptions = { ...options, maxCgIterations: 129 };
      const coordinates = (run) => [
        ...run.reconstruction.views.flatMap((view) => [...view.pose.R, ...view.pose.t]),
        ...run.reconstruction.landmarks.flatMap((point) => [...point.xyz]),
      ];

      const cpuDevice = await ShadeDevice.open({ backend: 'cpu' });
      let cpu, cpuFallback;
      try {
        cpu = await bundleAdjustGpu(cpuDevice, input, options);
        cpuFallback = await bundleAdjustGpu(cpuDevice, input, fallbackOptions);
      } finally {
        await cpuDevice.dispose();
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter available');
      const info = adapter.info,
        adapterDescription = [info?.vendor, info?.architecture, info?.device, info?.description]
          .filter(Boolean)
          .join(' '),
        software =
          /swiftshader|software|llvmpipe/i.test(adapterDescription) ||
          adapter.isFallbackAdapter === true;
      if (hardware && software)
        throw new Error(
          `Hardware requested, but a software adapter was selected: ${adapterDescription}`,
        );
      const gpu = await adapter.requestDevice(),
        validationErrors = [];
      gpu.addEventListener('uncapturederror', (event) =>
        validationErrors.push(event.error.message),
      );
      const gpuDevice = await ShadeDevice.open({ gpuDevice: gpu });
      let repeated, fallback;
      try {
        repeated = [];
        for (let i = 0; i < 3; i++) {
          repeated.push(await bundleAdjustGpu(gpuDevice, structuredClone(input), options));
        }
        fallback = await bundleAdjustGpu(gpuDevice, structuredClone(input), fallbackOptions);
      } finally {
        await gpuDevice.dispose();
      }
      if (validationErrors.length)
        throw new Error(`WebGPU validation errors: ${validationErrors.join('; ')}`);

      const reference = coordinates(cpu),
        fallbackCpuCoordinates = coordinates(cpuFallback),
        fallbackGpuCoordinates = coordinates(fallback),
        runs = repeated.map((run) => {
          const values = coordinates(run);
          let maxCoordinateError = 0;
          for (let i = 0; i < reference.length; i++)
            maxCoordinateError = Math.max(maxCoordinateError, Math.abs(reference[i] - values[i]));
          const calls = run.execution.report.groups.flatMap((group) => group.calls);
          return {
            maxCoordinateError,
            acceptedSteps: run.diagnostics.acceptedSteps,
            rejectedSteps: run.diagnostics.rejectedSteps,
            linearSolverFailures: run.linearSolverFailures,
            cgIterations: run.cgIterations,
            finalCost: run.diagnostics.finalCost,
            finalRmsePx: run.diagnostics.finalRmsePx,
            calls,
          };
        });
      return {
        fixture: { views: input.views.length, landmarks: input.landmarks.length },
        cpu: {
          acceptedSteps: cpu.diagnostics.acceptedSteps,
          rejectedSteps: cpu.diagnostics.rejectedSteps,
          linearSolverFailures: cpu.linearSolverFailures,
          cgIterations: cpu.cgIterations,
          finalCost: cpu.diagnostics.finalCost,
          finalRmsePx: cpu.diagnostics.finalRmsePx,
        },
        fallback: {
          cpu: {
            acceptedSteps: cpuFallback.diagnostics.acceptedSteps,
            rejectedSteps: cpuFallback.diagnostics.rejectedSteps,
            linearSolverFailures: cpuFallback.linearSolverFailures,
            cgIterations: cpuFallback.cgIterations,
            finalCost: cpuFallback.diagnostics.finalCost,
            finalRmsePx: cpuFallback.diagnostics.finalRmsePx,
          },
          gpu: {
            acceptedSteps: fallback.diagnostics.acceptedSteps,
            rejectedSteps: fallback.diagnostics.rejectedSteps,
            linearSolverFailures: fallback.linearSolverFailures,
            cgIterations: fallback.cgIterations,
            finalCost: fallback.diagnostics.finalCost,
            finalRmsePx: fallback.diagnostics.finalRmsePx,
            maxCoordinateError: Math.max(
              ...fallbackCpuCoordinates.map((value, i) =>
                Math.abs(value - fallbackGpuCoordinates[i]),
              ),
            ),
            calls: fallback.execution.report.groups.flatMap((group) => group.calls),
          },
        },
        runs,
      };
    },
    { hardware },
  );

  assert.deepEqual(errors, []);
  assert.deepEqual(result.fixture, { views: 8, landmarks: 256 });
  for (const run of result.runs) {
    assert.ok(run.calls.includes('advanceBundleCg'), JSON.stringify(result));
    assert.ok(!run.calls.includes('bundlePersistentCg'), JSON.stringify(result));
    assert.equal(run.acceptedSteps, result.cpu.acceptedSteps, JSON.stringify(result));
    assert.equal(run.rejectedSteps, result.cpu.rejectedSteps, JSON.stringify(result));
    assert.equal(run.linearSolverFailures, result.cpu.linearSolverFailures, JSON.stringify(result));
    assert.equal(run.cgIterations, result.cpu.cgIterations, JSON.stringify(result));
    assert.ok(run.maxCoordinateError < 1e-7, JSON.stringify(result));
    assert.ok(Math.abs(run.finalCost - result.cpu.finalCost) < 1e-8, JSON.stringify(result));
    assert.ok(Math.abs(run.finalRmsePx - result.cpu.finalRmsePx) < 1e-8, JSON.stringify(result));
  }
  for (const field of [
    'acceptedSteps',
    'rejectedSteps',
    'linearSolverFailures',
    'cgIterations',
    'finalCost',
    'finalRmsePx',
    'maxCoordinateError',
  ]) {
    assert.equal(
      new Set(result.runs.map((run) => run[field])).size,
      1,
      `Chunked GPU BA varied across repeated runs for ${field}: ${JSON.stringify(result)}`,
    );
  }
  assert.ok(result.fallback.gpu.calls.includes('bundleSchurPoint'), JSON.stringify(result));
  assert.ok(result.fallback.gpu.calls.includes('bundleSchurCamera'), JSON.stringify(result));
  assert.ok(result.fallback.gpu.calls.includes('advanceBundleCg'), JSON.stringify(result));
  assert.ok(!result.fallback.gpu.calls.includes('bundlePersistentCg'), JSON.stringify(result));
  assert.equal(
    result.fallback.gpu.acceptedSteps,
    result.fallback.cpu.acceptedSteps,
    JSON.stringify(result),
  );
  assert.equal(
    result.fallback.gpu.rejectedSteps,
    result.fallback.cpu.rejectedSteps,
    JSON.stringify(result),
  );
  assert.equal(
    result.fallback.gpu.linearSolverFailures,
    result.fallback.cpu.linearSolverFailures,
    JSON.stringify(result),
  );
  assert.equal(
    result.fallback.gpu.cgIterations,
    result.fallback.cpu.cgIterations,
    JSON.stringify(result),
  );
  assert.ok(result.fallback.gpu.maxCoordinateError < 1e-7, JSON.stringify(result));
  assert.ok(Math.abs(result.fallback.gpu.finalCost - result.fallback.cpu.finalCost) < 1e-8);
  assert.ok(Math.abs(result.fallback.gpu.finalRmsePx - result.fallback.cpu.finalRmsePx) < 1e-8);
});

// docs/gpu-throughput.md, item 5: the camera product runs one workgroup per camera on WebGPU. It
// is held to the row kernel it replaced (`cameraProduct: 'rows'`, the CPU tier's oracle) on the
// same device: the first S p within 1e-10 of its largest entry, the same LM counts, PCG counts
// within one a solve, the final cost and error within 1e-6 relative.
test('one workgroup per camera holds the row kernel on the first PCG product and in the LM result', async () => {
  const hardware = process.env.TYPECV_HARDWARE_GPU === '1';
  const { result, errors } = await inBrowser(
    async ({ hardware }) => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { bundleFixture, ringBundleFixture } =
        await import('/tests/typeshade/geometry-cases.mjs');
      const { jointFocalFixture } = await import('/tests/typeshade/joint-focal-cases.mjs');
      const cases = [
        { name: 'fixed', input: bundleFixture(), options: { iterations: 6, huberPx: 1 } },
        {
          name: 'fixed outlier',
          input: bundleFixture({ outlier: true }),
          options: { iterations: 6, huberPx: 1 },
        },
        {
          name: 'fixed 8 views',
          input: bundleFixture({ count: 256, views: 8 }),
          options: { iterations: 6, huberPx: 1 },
        },
        {
          name: 'fixed ring 320',
          input: ringBundleFixture(),
          options: { iterations: 2, huberPx: 1 },
        },
        {
          name: 'shared',
          input: jointFocalFixture({ noise: 0.1, focalScale: 0.9 }).input,
          options: { iterations: 6, huberPx: 1, refineSharedFocal: true },
        },
        {
          name: 'shared window',
          input: jointFocalFixture({ points: 20, views: 4, noise: 0.2, radial: false }).input,
          options: { iterations: 6, refineSharedFocal: true, variableViewIds: [30, 40] },
        },
        {
          name: 'shared ring 320',
          input: ringBundleFixture(),
          options: { iterations: 2, huberPx: 1, refineSharedFocal: true },
        },
      ];
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter available');
      const info = adapter.info,
        description = [info?.vendor, info?.architecture, info?.device, info?.description]
          .filter(Boolean)
          .join(' ');
      if (
        hardware &&
        (/swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter)
      )
        throw new Error(`Hardware requested, but a software adapter was selected: ${description}`);
      const gpu = await adapter.requestDevice(),
        gpuErrors = [];
      gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      // One step per dispatch: after each `advanceBundleCg` of the first products, read S p
      // (`vectors[3m, 4m)`, which the step reads and does not write).
      const run = async (input, options) => {
        const C = input.views.length,
          P = input.landmarks.length,
          shared = options.refineSharedFocal === true,
          m = 6 * C + (shared ? 1 : 0),
          tiled = options.cameraProduct !== 'rows',
          vectorLength = 5 * m + 3 * P + (shared && tiled ? C : 0),
          device = await ShadeDevice.open({
            gpuDevice: gpu,
            boundedSubmissions: true,
            maxDispatchesPerSubmission: 64,
            solverCheckpoints: false,
            poolBytes: 0,
            workingBytes: 1024 ** 3,
          }),
          products = [];
        let vectors;
        const allocate = device.allocateF64.bind(device),
          planSteps = device.planSteps.bind(device);
        device.allocateF64 = (length, values) => {
          const buffer = allocate(length, values);
          if (length === vectorLength) vectors = buffer;
          return buffer;
        };
        device.planSteps = (steps) =>
          planSteps(
            (function* () {
              for (const step of steps) {
                if (step[0] !== 'advanceBundleCg' || products.length >= 16) {
                  yield step;
                  continue;
                }
                yield [
                  step[0],
                  step[1],
                  async () => {
                    await step[2]?.();
                    const values = await device.read(vectors, 4 * m);
                    products.push([...values.subarray(3 * m, 4 * m)]);
                  },
                ];
              }
            })(),
          );
        try {
          const r = await bundleAdjustGpu(device, structuredClone(input), options);
          return {
            products,
            accepted: r.diagnostics.acceptedSteps,
            rejected: r.diagnostics.rejectedSteps,
            failures: r.linearSolverFailures,
            cgIterations: r.cgIterations,
            finalCost: r.diagnostics.finalCost,
            finalRmsePx: r.diagnostics.finalRmsePx,
            calls: [...new Set(r.execution.report.groups.flatMap((g) => g.calls))],
            live: device.statistics.liveBuffers,
          };
        } finally {
          await device.dispose();
        }
      };
      const out = [];
      for (const { name, input, options } of cases) {
        const rows = await run(input, { ...options, cameraProduct: 'rows' }),
          tiles = await run(input, options);
        const errors = [];
        for (let k = 0; k < Math.min(rows.products.length, tiles.products.length); k++) {
          const a = tiles.products[k],
            b = rows.products[k],
            scale = Math.max(...b.map(Math.abs));
          errors.push(scale > 0 ? Math.max(...a.map((x, i) => Math.abs(x - b[i]))) / scale : 0);
        }
        out.push({
          name,
          cameras: input.views.length,
          worst: Math.max(...errors),
          errors,
          rows: { ...rows, products: rows.products.length },
          tiles: { ...tiles, products: tiles.products.length },
        });
      }
      if (gpuErrors.length) throw new Error(`WebGPU validation errors: ${gpuErrors.join('; ')}`);
      return out;
    },
    { hardware },
    { timeoutMs: 1800000 },
  );
  assert.deepEqual(errors, []);
  for (const row of result) {
    const text = JSON.stringify(row);
    // Bounded submissions without checkpoints run every PCG iteration of the budget.
    assert.equal(row.tiles.products, 16, text);
    assert.equal(row.rows.products, row.tiles.products, text);
    // The first product reads the same p and the same blocks in both runs: the difference is the
    // order of the sum alone. The later products drift apart as PCG amplifies the last bits
    // (to 1e-5 relative by the sixteenth on the shared fixture), so `errors` only reports them.
    assert.ok(row.errors[0] <= 1e-10, text);
    for (const field of ['accepted', 'rejected', 'failures'])
      assert.equal(row.tiles[field], row.rows[field], `${field}: ${text}`);
    // A PCG solve may cross its tolerance one iteration sooner or later: at most one a solve.
    assert.ok(
      Math.abs(row.tiles.cgIterations - row.rows.cgIterations) <=
        row.rows.accepted + row.rows.rejected,
      `cgIterations: ${text}`,
    );
    for (const field of ['finalCost', 'finalRmsePx'])
      assert.ok(
        Math.abs(row.tiles[field] - row.rows[field]) <= 1e-6 * Math.abs(row.rows[field]),
        `${field}: ${text}`,
      );
    assert.ok(row.tiles.calls.includes('bundleSchurCamera'), text);
    assert.ok(!row.tiles.calls.includes('bundleSchurCameraOracle'), text);
    assert.ok(row.rows.calls.includes('bundleSchurCameraOracle'), text);
    assert.ok(!row.rows.calls.includes('bundleSchurCamera'), text);
    assert.equal(
      row.tiles.calls.includes('finishBundleSchurFocal'),
      row.name.startsWith('shared'),
      text,
    );
    assert.equal(row.tiles.live, 0);
    assert.equal(row.rows.live, 0);
  }
  assert.ok(result.some((row) => row.cameras >= 300));
});

// The camera product dispatches one workgroup per camera, so a dispatch may not hold more cameras
// than `maxComputeWorkgroupsPerDimension`: past it, the host splits the dispatch (`control.base`).
// A device that reports a limit of 128 runs 320 cameras as three dispatches, bit for bit the one.
test('the camera product splits at the workgroup limit with unchanged results', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { ringBundleFixture } = await import('/tests/typeshade/geometry-cases.mjs');
      const input = ringBundleFixture(),
        out = [];
      for (const shared of [false, true]) {
        const options = { iterations: 2, huberPx: 1, refineSharedFocal: shared },
          runs = [];
        for (const limit of [undefined, 128]) {
          const device = await ShadeDevice.open({ workingBytes: 1024 ** 3, poolBytes: 0 });
          if (limit) Object.defineProperty(device, 'maxWorkgroups', { value: limit });
          try {
            const r = await bundleAdjustGpu(device, structuredClone(input), options),
              calls = r.execution.report.groups.flatMap((g) => g.calls);
            runs.push({
              maxWorkgroups: device.maxWorkgroups,
              text: JSON.stringify([r.reconstruction, r.diagnostics, r.cgIterations]),
              camera: calls.filter((c) => c === 'bundleSchurCamera').length,
              advance: calls.filter((c) => c === 'advanceBundleCg').length,
              finish: calls.filter((c) => c === 'finishBundleSchurFocal').length,
            });
          } finally {
            await device.dispose();
          }
        }
        out.push({ shared, cameras: input.views.length, runs });
      }
      return out;
    },
    undefined,
    { timeoutMs: 1800000 },
  );
  assert.deepEqual(errors, []);
  for (const { shared, cameras, runs } of result) {
    const [whole, split] = runs;
    assert.equal(cameras, 320);
    assert.ok(whole.maxWorkgroups >= cameras);
    assert.equal(split.maxWorkgroups, 128);
    assert.equal(split.text, whole.text, `shared ${shared}`);
    assert.ok(whole.advance > 0);
    assert.equal(whole.camera, whole.advance);
    assert.equal(split.camera, 3 * split.advance);
    assert.equal(split.finish, shared ? split.advance : 0);
  }
});
