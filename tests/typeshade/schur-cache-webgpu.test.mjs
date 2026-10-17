import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { bundleAdjust } from '../../dist/index.js';
import { jointFocalFixture } from './joint-focal-cases.mjs';
import {
  scalarSchur,
  scalarMatrixFree,
  multiplySchur,
  validSpaceVector,
} from './schur-cache-cases.mjs';

function assertOperator(system) {
  const dense = scalarSchur(system),
    m = 6 * system.C + 1;
  assert.equal(system.schur.length, m * m);
  for (let r = 0; r < m; r++)
    for (let c = 0; c < m; c++) {
      assert.equal(system.schur[c * m + r], system.schur[r * m + c]);
      assert.ok(
        Math.abs(system.schur[c * m + r] - dense[r * m + c]) <=
          1e-10 * Math.max(1, Math.abs(dense[r * m + c])),
      );
    }
  for (const seed of [0.3, 1.7, 4.1]) {
    const vector = validSpaceVector(system, seed),
      expected = scalarMatrixFree(system, vector),
      actual = multiplySchur(system.schur, vector, true),
      scale = Math.max(1, ...expected.map(Math.abs));
    assert.ok(vector.some((x) => x !== 0));
    assert.ok(Math.max(...actual.map((x, i) => Math.abs(x - expected[i]))) <= 1e-10 * scale);
  }
}

test('cached WebGPU Schur operators and certificates match independent scalar algebra', async () => {
  const { result, errors } = await inBrowser(
    async ({ hardware }) => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js'),
        { jointFocalFixture } = await import('/tests/typeshade/joint-focal-cases.mjs'),
        { captureSchurSystems } = await import('/tests/typeshade/schur-cache-cases.mjs'),
        adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('WebGPU adapter unavailable');
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
        (/swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter === true)
      )
        throw new Error('Hardware requested but software adapter selected');
      const gpu = await adapter.requestDevice(),
        gpuErrors = [],
        rows = [];
      gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      try {
        for (const settings of [
          { fixture: { noise: 0, focalScale: 0.75 }, options: {} },
          {
            fixture: { noise: 0.1, focalScale: 1 },
            options: { anchorViewId: 20, scaleViewId: 40 },
          },
          { fixture: { noise: 0.2, radial: false }, options: { variableViewIds: [30, 40] } },
        ]) {
          const { input } = jointFocalFixture({ points: 12, views: 4, ...settings.fixture }),
            device = await ShadeDevice.open({
              gpuDevice: gpu,
              solverCheckpoints: true,
              poolBytes: 0,
            }),
            captured = captureSchurSystems(device, input),
            snapshot = JSON.stringify(input);
          try {
            const run = await bundleAdjustGpu(device, input, {
              iterations: 1,
              refineSharedFocal: true,
              schurCache: true,
              maxCgIterations: 256,
              cgRelativeTolerance: 1e-10,
              ...settings.options,
            });
            rows.push({
              ...captured,
              diagnostics: run.diagnostics,
              cgIterations: run.cgIterations,
              calls: run.execution.report.groups.flatMap((g) => g.calls),
              tier: run.execution.report.tier,
              inputPreserved: JSON.stringify(input) === snapshot,
            });
          } finally {
            await device.dispose();
          }
          if (device.statistics.liveBuffers) throw new Error('Leaked Schur cache buffers');
        }
      } finally {
        gpu.destroy();
      }
      return { rows, gpuErrors };
    },
    { hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
    { timeoutMs: 300000 },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  for (const row of result.rows) {
    assert.equal(row.tier, 'webgpu');
    assert.equal(row.inputPreserved, true);
    assert.ok(row.calls.includes('bundleCachedPersistentCg'));
    assert.ok(row.calls.includes('certifyCachedBundleCg'));
    assert.equal(row.systems.length, 1);
    const system = row.systems[0],
      m = 6 * system.C + 1;
    assertOperator(system);
    assert.ok(row.certificates.length >= 1);
    for (const certificate of row.certificates) assert.ok(certificate.state[23] <= 256);
    const last = row.certificates.at(-1),
      actual = scalarMatrixFree(system, last.vectors.slice(0, m));
    let residual = 0;
    for (let i = 0; i < m; i++) {
      const rhs =
        i === m - 1
          ? system.focal[6 * system.C + 2]
          : system.blocks[
              45 * system.M + 13 * system.P + 72 * system.C + Math.floor(i / 6) * 6 + (i % 6)
            ];
      residual += (rhs - actual[i]) ** 2;
    }
    assert.ok(Math.abs(residual - last.state[22]) <= 1e-10 * Math.max(1, residual));
    if (row.diagnostics.acceptedSteps) assert.ok(residual <= 1e-20 * last.state[21]);
    assert.ok(row.cgIterations <= 256);
  }
});

test('cached WebGPU focal refinement preserves quality, rejects exhaustion and honors eligibility and memory budgets', async () => {
  const { input: oracleInput, truth } = jointFocalFixture(),
    options = {
      iterations: 40,
      refineSharedFocal: true,
      maxCgIterations: 256,
      cgRelativeTolerance: 1e-10,
    },
    reference = bundleAdjust(oracleInput, options),
    { result, errors } = await inBrowser(
      async ({ hardware, options }) => {
        const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js'),
          { jointFocalFixture } = await import('/tests/typeshade/joint-focal-cases.mjs'),
          { captureSchurSystems } = await import('/tests/typeshade/schur-cache-cases.mjs'),
          adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) throw new Error('WebGPU adapter unavailable');
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
          (/swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter === true)
        )
          throw new Error('Hardware requested but software adapter selected');
        const gpu = await adapter.requestDevice(),
          gpuErrors = [];
        gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
        const run = async (
          input,
          controls,
          settings = {},
          corruptCache = false,
          capture = false,
        ) => {
          const device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0, ...settings }),
            captured =
              corruptCache || capture
                ? captureSchurSystems(device, input, { corruptCache })
                : undefined,
            snapshot = JSON.stringify(input);
          try {
            const result = await bundleAdjustGpu(device, input, controls);
            return {
              diagnostics: result.diagnostics,
              cgIterations: result.cgIterations,
              failures: result.linearSolverFailures,
              focal: result.reconstruction.views[0].camera.fx,
              coordinates: [
                ...result.reconstruction.views.flatMap((v) => [...v.pose.R, ...v.pose.t]),
                ...result.reconstruction.landmarks.flatMap((p) => [...p.xyz]),
              ],
              calls: result.execution.report.groups.flatMap((g) => g.calls),
              readback: result.execution.applicationReadbackBytes,
              unchanged: JSON.stringify(result.reconstruction) === snapshot,
              inputPreserved: JSON.stringify(input) === snapshot,
              observationsPreserved: result.reconstruction.landmarks.every(
                (p, i) =>
                  JSON.stringify(p.observations) ===
                  JSON.stringify(input.landmarks[i].observations),
              ),
              fixedPreserved:
                JSON.stringify(result.reconstruction.views[0].pose) ===
                JSON.stringify(input.views[0].pose),
              certificates: captured?.certificates,
              systems: captured?.systems,
            };
          } finally {
            await device.dispose();
            if (device.statistics.liveBuffers) throw new Error('Leaked Schur cache buffers');
          }
        };
        let output;
        try {
          const { input } = jointFocalFixture(),
            quality = await run(
              input,
              { ...options, schurCache: true },
              { solverCheckpoints: false },
            ),
            small = jointFocalFixture({ points: 12, views: 4 }).input,
            failed = await run(
              small,
              {
                iterations: 1,
                refineSharedFocal: true,
                schurCache: true,
                maxCgIterations: 1,
                cgRelativeTolerance: 1e-12,
              },
              { solverCheckpoints: true },
              true,
            ),
            recovered = await run(
              small,
              {
                iterations: 1,
                refineSharedFocal: true,
                schurCache: true,
                maxCgIterations: 256,
                cgRelativeTolerance: 1e-10,
              },
              { solverCheckpoints: true },
              true,
            ),
            boundedOptions = {
              iterations: 1,
              refineSharedFocal: true,
              maxCgIterations: 1,
              cgRelativeTolerance: 1e-12,
            },
            bounded = await run(
              small,
              { ...boundedOptions, schurCache: true },
              { boundedSubmissions: true, maxDispatchesPerSubmission: 1 },
            ),
            boundedControl = await run(small, boundedOptions, {
              boundedSubmissions: true,
              maxDispatchesPerSubmission: 1,
            });
          const expanded = structuredClone(small);
          while (expanded.views.length < 17) {
            const id = 10 + expanded.views.length * 10,
              last = expanded.views.at(-1);
            expanded.views.push({ ...structuredClone(last), id });
            for (const p of expanded.landmarks)
              p.observations.push({ ...structuredClone(p.observations.at(-1)), viewId: id });
          }
          const large = await run(expanded, { ...boundedOptions, schurCache: true }),
            largeControl = await run(expanded, boundedOptions),
            atLimit = {
              ...expanded,
              views: expanded.views.slice(0, 16),
              landmarks: expanded.landmarks.map((p) => ({
                ...p,
                observations: p.observations.slice(0, 16),
              })),
            },
            limit = await run(
              atLimit,
              { ...boundedOptions, schurCache: true },
              { solverCheckpoints: true },
              false,
              true,
            );
          const C = 4,
            P = 12,
            M = 48,
            lengths = [
              18 * C + 3 * P + 2 * M,
              12 * C + 3 * P,
              45 * M + 13 * P + 78 * C,
              36 * C,
              5 * (6 * C + 1) + 3 * P,
              26 + 2 * M,
              6 * C + 6 + 2 * M,
              12 * C + 3 * P + 27,
              (6 * C + 1) ** 2,
              18 * M + 3 * P,
            ],
            exactBytes = lengths.reduce((sum, n) => sum + n * 8, (3 * M + P + 2 * C + 2) * 4 + 12);
          const budget = await run(
            small,
            { ...boundedOptions, schurCache: true },
            { workingBytes: exactBytes },
          );
          const short = await ShadeDevice.open({
            gpuDevice: gpu,
            poolBytes: 0,
            workingBytes: exactBytes - 1,
          });
          let budgetError = '',
            live;
          try {
            await bundleAdjustGpu(short, small, { ...boundedOptions, schurCache: true });
          } catch (error) {
            budgetError = String(error);
          } finally {
            live = short.statistics.liveBuffers;
            await short.dispose();
          }
          // Exercise the assembly-work boundary entirely in host preflight. The
          // allocation interceptor prevents either large problem reaching a GPU kernel.
          const preflightRows = [];
          for (const observations of [10485, 10486]) {
            const data = jointFocalFixture({ points: 2622, views: 4 }).input;
            data.landmarks.at(-1).observations.splice(2);
            if (observations === 10485) data.landmarks.at(-2).observations.splice(3);
            const C = 4,
              P = 2622,
              M = observations,
              base = [
                18 * C + 3 * P + 2 * M,
                12 * C + 3 * P,
                45 * M + 13 * P + 78 * C,
                36 * C,
                5 * (6 * C + 1) + 3 * P,
                26 + 2 * M,
                6 * C + 6 + 2 * M,
                12 * C + 3 * P + 27,
              ],
              oldBytes = base.reduce((sum, n) => sum + n * 8, (3 * M + P + 2 * C + 2) * 4 + 8),
              preflight = await ShadeDevice.open({
                gpuDevice: gpu,
                poolBytes: 0,
                workingBytes: oldBytes,
              });
            let allocations = 0,
              error = '';
            preflight.allocateF64 = () => {
              allocations++;
              throw new Error('Stopped at first allocation before GPU work');
            };
            try {
              await bundleAdjustGpu(preflight, data, { ...boundedOptions, schurCache: true });
            } catch (cause) {
              error = String(cause);
            } finally {
              await preflight.dispose();
            }
            preflightRows.push({
              observations,
              allocations,
              error,
              live: preflight.statistics.liveBuffers,
            });
          }
          const pooledDevice = await ShadeDevice.open({
              gpuDevice: gpu,
              poolBytes: 8 * 1024 * 1024,
            }),
            inputA = jointFocalFixture({ points: 12, views: 4, noise: 0.1, focalScale: 0.9 }).input,
            inputB = jointFocalFixture({
              points: 12,
              views: 4,
              noise: 0.2,
              focalScale: 0.75,
              radial: false,
            }).input,
            pooled = [];
          try {
            for (const data of [inputA, inputB, inputA]) {
              const result = await bundleAdjustGpu(pooledDevice, data, {
                ...options,
                iterations: 3,
                schurCache: true,
              });
              pooled.push({
                diagnostics: result.diagnostics,
                cgIterations: result.cgIterations,
                coordinates: [
                  ...result.reconstruction.views.flatMap((v) => [...v.pose.R, ...v.pose.t]),
                  ...result.reconstruction.landmarks.flatMap((p) => [...p.xyz]),
                ],
                allocations: pooledDevice.statistics.allocations,
              });
            }
          } finally {
            await pooledDevice.dispose();
          }
          output = {
            quality,
            failed,
            recovered,
            bounded,
            boundedControl,
            large,
            largeControl,
            limit,
            budget,
            budgetError,
            budgetLive: live,
            preflightRows,
            pooled,
            pooledLive: pooledDevice.statistics.liveBuffers,
          };
        } finally {
          gpu.destroy();
        }
        return { ...output, gpuErrors };
      },
      { hardware: process.env.TYPECV_HARDWARE_GPU === '1', options },
      { timeoutMs: 300000 },
    );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  assert.ok(result.quality.diagnostics.finalCost < 1e-7);
  assert.ok(result.quality.diagnostics.finalRmsePx < 1e-4);
  assert.ok(Math.abs(result.quality.focal - truth.views[0].camera.fx) < 0.001);
  assert.ok(
    Math.abs(result.quality.diagnostics.finalCost - reference.diagnostics.finalCost) < 1e-7,
  );
  const referenceCoordinates = [
    ...reference.reconstruction.views.flatMap((v) => [...v.pose.R, ...v.pose.t]),
    ...reference.reconstruction.landmarks.flatMap((p) => [...p.xyz]),
  ];
  assert.equal(result.quality.coordinates.length, referenceCoordinates.length);
  for (let i = 0; i < referenceCoordinates.length; i++)
    assert.ok(
      Math.abs(result.quality.coordinates[i] - referenceCoordinates[i]) <
        (i < 12 * oracleInput.views.length ? 1e-6 : 1e-5),
    );
  assert.ok(
    result.quality.inputPreserved &&
      result.quality.observationsPreserved &&
      result.quality.fixedPreserved,
  );
  assert.ok(result.quality.calls.includes('bundleCachedPersistentCg'));
  assert.ok(result.quality.calls.includes('certifyCachedBundleCg'));
  assert.equal(result.failed.diagnostics.acceptedSteps, 0);
  assert.equal(result.failed.diagnostics.rejectedSteps, 1);
  assert.equal(result.failed.failures, 1);
  assert.equal(result.failed.unchanged, true);
  assert.equal(result.failed.certificates.at(-1).state[23], 1);
  assert.ok(!result.failed.calls.includes('bundlePersistentCg'));
  assert.ok(result.recovered.calls.includes('bundlePersistentCg'));
  assert.ok(result.recovered.certificates.length >= 2);
  assert.ok(result.recovered.certificates.at(-1).state[23] <= 256);
  assert.ok(result.recovered.diagnostics.acceptedSteps > 0);
  for (const [requested, control] of [
    [result.bounded, result.boundedControl],
    [result.large, result.largeControl],
  ]) {
    assert.deepEqual(requested.diagnostics, control.diagnostics);
    assert.equal(requested.cgIterations, control.cgIterations);
    assert.ok(!requested.calls.includes('bundleCachedPersistentCg'));
  }
  assert.ok(result.limit.calls.includes('bundleCachedPersistentCg'));
  assert.equal(result.limit.systems[0].C, 16);
  assertOperator(result.limit.systems[0]);
  assert.ok(result.budget.calls.includes('bundleCachedPersistentCg'));
  assert.match(result.budgetError, /working\/storage\/dispatch budget/);
  assert.equal(result.budgetLive, 0);
  assert.deepEqual(
    result.preflightRows.map((row) => row.observations),
    [10485, 10486],
  );
  assert.equal(result.preflightRows[0].allocations, 0);
  assert.match(result.preflightRows[0].error, /working\/storage\/dispatch budget/);
  assert.equal(result.preflightRows[1].allocations, 1);
  assert.match(result.preflightRows[1].error, /Stopped at first allocation/);
  assert.ok(result.preflightRows.every((row) => row.live === 0));
  assert.deepEqual(result.pooled[0], result.pooled[2]);
  assert.equal(result.pooled[0].allocations, result.pooled[1].allocations);
  assert.equal(result.pooledLive, 0);
});
