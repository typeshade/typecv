import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { jointFocalFixture } from './joint-focal-cases.mjs';

test('WebGPU jointly recovers shared focal with radial projection and preserves zero-step and failed trials', async () => {
  const fixture = JSON.parse(
    JSON.stringify(jointFocalFixture(), (_key, value) =>
      ArrayBuffer.isView(value) ? Array.from(value) : value,
    ),
  );
  const { result, errors } = await inBrowser(
    async ({ fixture, hardware }) => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { bundleAdjust } = await import('/dist/index.js');
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter');
      const info = adapter.info,
        identity = [info.vendor, info.architecture, info.device, info.description].join(' ');
      const software = adapter.isFallbackAdapter || /software|swiftshader|llvmpipe/i.test(identity);
      if (hardware && software) throw new Error('Hardware requested, software selected');
      const gpu = await adapter.requestDevice(),
        gpuErrors = [];
      gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      const device = await ShadeDevice.open({ gpuDevice: gpu, solverCheckpoints: true });
      const input = {
          views: fixture.input.views.map((v) => ({
            ...v,
            pose: { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) },
          })),
          landmarks: fixture.input.landmarks.map((p) => ({ ...p, xyz: Float64Array.from(p.xyz) })),
        },
        options = {
          iterations: 40,
          refineSharedFocal: true,
          maxCgIterations: 256,
          cgRelativeTolerance: 1e-10,
        };
      try {
        const snapshot = JSON.stringify(input);
        const started = performance.now(),
          actual = await bundleAdjustGpu(device, input, options),
          elapsedMs = performance.now() - started;
        const expected = bundleAdjust(input, options);
        const zero = await bundleAdjustGpu(device, input, { ...options, iterations: 0 });
        const rejected = await bundleAdjustGpu(device, input, {
          ...options,
          iterations: 1,
          maxCgIterations: 1,
          cgRelativeTolerance: 1e-12,
        });
        return {
          identity,
          software,
          gpuErrors,
          elapsedMs,
          actual: actual.diagnostics,
          expected: expected.diagnostics,
          actualCameras: actual.reconstruction.views.map((v) => v.camera),
          truthCameras: fixture.truth.views.map((v) => v.camera),
          inputCameras: input.views.map((v) => v.camera),
          zeroUnchanged: JSON.stringify(zero.reconstruction) === snapshot,
          rejectedUnchanged: JSON.stringify(rejected.reconstruction) === snapshot,
          rejectedAccepted: rejected.diagnostics.acceptedSteps,
          anchorUnchanged:
            JSON.stringify(actual.reconstruction.views[0].pose) ===
            JSON.stringify(input.views[0].pose),
          inputUnchanged: JSON.stringify(input) === snapshot,
          tier: actual.execution.report.tier,
          cpuStages: actual.execution.cpuStages,
        };
      } finally {
        await device.dispose();
        gpu.destroy();
      }
    },
    { fixture, hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
    { timeoutMs: 300000 },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  assert.equal(result.tier, 'webgpu');
  assert.deepEqual(result.cpuStages, []);
  assert.ok(
    result.zeroUnchanged &&
      result.rejectedUnchanged &&
      result.anchorUnchanged &&
      result.inputUnchanged,
    JSON.stringify(result),
  );
  assert.equal(result.rejectedAccepted, 0);
  assert.ok(result.actual.finalCost < 1e-7, JSON.stringify(result));
  assert.ok(result.actual.finalRmsePx < 1e-4);
  assert.ok(Math.abs(result.actual.finalCost - result.expected.finalCost) < 1e-7);
  for (let i = 0; i < result.actualCameras.length; i++) {
    assert.ok(Math.abs(result.actualCameras[i].fx - result.truthCameras[i].fx) < 0.001);
    assert.ok(Math.abs(result.actualCameras[i].fy - result.truthCameras[i].fy) < 0.001);
    assert.equal(result.actualCameras[i].k1, result.inputCameras[i].k1);
    assert.equal(result.actualCameras[i].k2, result.inputCameras[i].k2);
  }
  console.log(JSON.stringify({ jointFocal: result }));
});

test('WebGPU global focal row beyond lane64 works with 64 fixed cameras in chunked and bounded schedules', async () => {
  const fixture = JSON.parse(
    JSON.stringify(jointFocalFixture({ views: 4, points: 8 }), (_k, v) =>
      ArrayBuffer.isView(v) ? Array.from(v) : v,
    ),
  );
  const { result, errors } = await inBrowser(
    async ({ fixture, hardware }) => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { bundleAdjust } = await import('/dist/index.js');
      const adapter = await navigator.gpu.requestAdapter();
      const info = adapter.info,
        identity = [info.vendor, info.architecture, info.device, info.description].join(' ');
      if (
        hardware &&
        (adapter.isFallbackAdapter || /software|swiftshader|llvmpipe/i.test(identity))
      )
        throw new Error('Physical adapter required');
      const gpu = await adapter.requestDevice(),
        gpuErrors = [];
      gpu.addEventListener('uncapturederror', (e) => gpuErrors.push(e.error.message));
      const input = {
        views: Array.from({ length: 64 }, (_, i) => {
          const v = fixture.input.views[i % 4];
          return {
            ...v,
            id: i + 100,
            camera: { ...v.camera },
            pose: { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) },
          };
        }),
        landmarks: fixture.input.landmarks.map((p) => ({
          ...p,
          xyz: Float64Array.from(p.xyz),
          observations: Array.from({ length: 64 }, (_, i) => ({
            ...p.observations[i % 4],
            viewId: i + 100,
          })),
        })),
      };
      const options = {
        iterations: 12,
        refineSharedFocal: true,
        variableViewIds: [],
        maxCgIterations: 8,
        cgRelativeTolerance: 1e-10,
      };
      const cpu = bundleAdjust(input, options),
        rows = [];
      try {
        for (const boundedSubmissions of [false, true]) {
          const device = await ShadeDevice.open({
            gpuDevice: gpu,
            boundedSubmissions,
            solverCheckpoints: true,
            poolBytes: 0,
          });
          try {
            const out = await bundleAdjustGpu(device, input, options);
            rows.push({
              boundedSubmissions,
              tier: out.execution.report.tier,
              cost: out.diagnostics.finalCost,
              initial: out.diagnostics.initialCost,
              expectedCost: cpu.diagnostics.finalCost,
              fx: out.reconstruction.views[0].camera.fx,
              expectedFx: cpu.reconstruction.views[0].camera.fx,
              fixed: out.reconstruction.views.every(
                (v, i) => JSON.stringify(v.pose) === JSON.stringify(input.views[i].pose),
              ),
              shared: out.reconstruction.views.every(
                (v) => Math.abs(v.camera.fx / out.reconstruction.views[0].camera.fx - 1) < 1e-12,
              ),
              observations: out.diagnostics.observations,
            });
          } finally {
            await device.dispose();
          }
        }
      } finally {
        gpu.destroy();
      }
      return { rows, gpuErrors };
    },
    { fixture, hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
    { timeoutMs: 300000 },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  for (const r of result.rows) {
    assert.equal(r.tier, 'webgpu');
    assert.ok(r.fixed && r.shared);
    assert.equal(r.observations, 512);
    assert.ok(r.cost < r.initial);
    assert.ok(Math.abs(r.cost - r.expectedCost) < 1e-6, JSON.stringify(r));
    assert.ok(Math.abs(r.fx - r.expectedFx) < 1e-5, JSON.stringify(r));
  }
  console.log(JSON.stringify({ focal64: result }));
});

test('WebGPU joint focal carries PCG across timed chunks and bounded submissions', async () => {
  const fixture = JSON.parse(
    JSON.stringify(jointFocalFixture({ views: 6, points: 200, focalScale: 0.95 }), (_key, value) =>
      ArrayBuffer.isView(value) ? Array.from(value) : value,
    ),
  );
  const { result, errors } = await inBrowser(
    async ({ fixture, hardware }) => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { bundleAdjust } = await import('/dist/index.js');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter');
      const info = adapter.info;
      const identity = [info.vendor, info.architecture, info.device, info.description].join(' ');
      if (
        hardware &&
        (adapter.isFallbackAdapter || /software|swiftshader|llvmpipe/i.test(identity))
      )
        throw new Error('Physical adapter required');
      const gpu = await adapter.requestDevice(),
        gpuErrors = [];
      gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      const input = {
        views: fixture.input.views.map((v) => ({
          ...v,
          pose: { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) },
        })),
        landmarks: fixture.input.landmarks.map((p) => ({ ...p, xyz: Float64Array.from(p.xyz) })),
      };
      const snapshot = JSON.stringify(input);
      // 1200 observations and up to 255 PCG iterations in one LM iteration: the first chunk of the
      // key `bundleCg` is one iteration, so a solve of more than one iteration spans chunks.
      const options = {
        iterations: 1,
        refineSharedFocal: true,
        maxCgIterations: 255,
        cgRelativeTolerance: 1e-10,
      };
      const expected = bundleAdjust(input, options),
        rows = [];
      let reference;
      let maxDifference = 0;
      try {
        for (const schedule of [
          { boundedSubmissions: false, solverCheckpoints: true },
          { boundedSubmissions: false, solverCheckpoints: false },
          { boundedSubmissions: true, solverCheckpoints: true },
        ]) {
          const device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0, ...schedule });
          try {
            const out = await bundleAdjustGpu(device, input, options);
            const calls = out.execution.report.groups.flatMap((g) => g.calls);
            const packed = out.reconstruction.views
              .flatMap((v) => [...v.pose.R, ...v.pose.t, v.camera.fx, v.camera.fy])
              .concat(out.reconstruction.landmarks.flatMap((p) => [...p.xyz]));
            if (!reference) reference = packed;
            else
              for (let i = 0; i < packed.length; i++)
                maxDifference = Math.max(maxDifference, Math.abs(packed[i] - reference[i]));
            rows.push({
              ...schedule,
              diagnostics: out.diagnostics,
              cgIterations: out.cgIterations,
              maxCgIterations: options.maxCgIterations,
              failures: out.linearSolverFailures,
              tier: out.execution.report.tier,
              persistentCalls: calls.filter((n) => n === 'bundlePersistentCg').length,
              distributedCalls: calls.filter((n) => n === 'bundleSchurPoint').length,
              advanceCalls: calls.filter((n) => n === 'advanceBundleCg').length,
              activityReads: calls.filter((n) => n === 'snapshotBundleActivity').length,
              chunks: out.execution.report.chunks,
              submissions: out.execution.report.submissions,
              anchorUnchanged:
                JSON.stringify(out.reconstruction.views[0].pose) ===
                JSON.stringify(input.views[0].pose),
              cpuStages: out.execution.cpuStages,
            });
          } finally {
            await device.dispose();
          }
        }
      } finally {
        gpu.destroy();
      }
      return {
        rows,
        expected: expected.diagnostics,
        maxDifference,
        inputUnchanged: JSON.stringify(input) === snapshot,
        gpuErrors,
      };
    },
    { fixture, hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
    { timeoutMs: 300000 },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  assert.ok(result.inputUnchanged);
  for (const row of result.rows) {
    assert.equal(row.tier, 'webgpu');
    assert.deepEqual(row.cpuStages, []);
    // One LM step proves continuation occurred inside a solve, not only across LM steps.
    assert.ok(row.cgIterations > 8 && row.cgIterations <= 255, JSON.stringify(row));
    assert.equal(row.failures, 0);
    assert.equal(row.diagnostics.acceptedSteps, 1);
    assert.ok(row.anchorUnchanged);
    assert.ok(
      Math.abs(row.diagnostics.finalCost - result.expected.finalCost) < 1e-6,
      JSON.stringify(row),
    );
  }
  // Every schedule runs the per-iteration kernels: the persistent kernel is removed.
  for (const row of result.rows) {
    assert.equal(row.persistentCalls, 0);
    assert.equal(row.distributedCalls, row.advanceCalls);
    // Each completed PCG iteration is one dispatch of each kernel; the iterations after
    // convergence that a chunk or a checkpoint stride still issues are guarded no-ops. Each
    // unbounded row's fresh device first dispatches the iteration's kernels over zero workgroups,
    // the warm-up before the first timed chunk of `bundleCg`.
    const iterations = row.advanceCalls - (row.boundedSubmissions ? 0 : 1);
    assert.ok(iterations >= row.cgIterations && iterations <= 255, JSON.stringify(row));
  }
  // Unbounded, the loop is chunks of the key `bundleCg`: the host reads the flags after each
  // chunk and, with checkpoints, once after the LM step. The first chunk also observes any
  // inactive initial PCG state through its guarded kernels, without a pre-chunk read. The
  // first chunk is one iteration, so a solve of more than eight iterations spans chunks.
  for (const row of result.rows.slice(0, 2)) {
    assert.ok(row.chunks.bundleCg.chunks >= 2, JSON.stringify(row));
    assert.equal(row.activityReads, row.chunks.bundleCg.chunks + Number(row.solverCheckpoints));
  }
  // Bounded submissions retain the initial and final checkpoints, and a checkpoint every four
  // issued PCG iterations except at the iteration cap, where the final checkpoint suffices.
  assert.equal(result.rows[2].chunks, undefined);
  assert.ok(result.rows[2].distributedCalls > 8);
  const bounded = result.rows[2];
  assert.equal(
    bounded.activityReads,
    2 + Math.floor(Math.min(bounded.advanceCalls, bounded.maxCgIterations - 1) / 4),
  );
  assert.equal(result.rows[0].cgIterations, result.rows[1].cgIterations);
  assert.equal(result.rows[0].cgIterations, result.rows[2].cgIterations);
  assert.equal(result.rows[0].diagnostics.acceptedSteps, result.rows[2].diagnostics.acceptedSteps);
  assert.equal(result.rows[0].diagnostics.rejectedSteps, result.rows[2].diagnostics.rejectedSteps);
  assert.ok(result.maxDifference < 1e-8, JSON.stringify(result));
  console.log(JSON.stringify({ jointFocalChunks: result }));
});

test('WebGPU joint focal chunk remainder rejects exhausted solves and includes the global row at camera64', async () => {
  const fixtures = JSON.parse(
    JSON.stringify(
      {
        exhausted: jointFocalFixture({ views: 6, points: 5000, focalScale: 0.95 }),
        wide: jointFocalFixture({ views: 4, points: 17, focalScale: 0.95 }),
      },
      (_key, value) => (ArrayBuffer.isView(value) ? Array.from(value) : value),
    ),
  );
  const { result, errors } = await inBrowser(
    async ({ fixtures, hardware }) => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { bundleAdjust } = await import('/dist/index.js');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter');
      const info = adapter.info;
      const identity = [info.vendor, info.architecture, info.device, info.description].join(' ');
      if (
        hardware &&
        (adapter.isFallbackAdapter || /software|swiftshader|llvmpipe/i.test(identity))
      )
        throw new Error('Physical adapter required');
      const gpu = await adapter.requestDevice(),
        gpuErrors = [];
      gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      const hydrate = (input) => ({
        views: input.views.map((v) => ({
          ...v,
          pose: { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) },
        })),
        landmarks: input.landmarks.map((p) => ({ ...p, xyz: Float64Array.from(p.xyz) })),
      });
      const input = hydrate(fixtures.exhausted.input);
      const snapshot = JSON.stringify(input);
      const options = {
        iterations: 1,
        refineSharedFocal: true,
        maxCgIterations: 9,
        cgRelativeTolerance: 1e-12,
      };
      let device;
      let exhausted, wide;
      try {
        device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0, solverCheckpoints: true });
        try {
          const out = await bundleAdjustGpu(device, input, options);
          const calls = out.execution.report.groups.flatMap((g) => g.calls);
          exhausted = {
            diagnostics: out.diagnostics,
            cgIterations: out.cgIterations,
            failures: out.linearSolverFailures,
            unchanged: JSON.stringify(out.reconstruction) === snapshot,
            inputUnchanged: JSON.stringify(input) === snapshot,
            persistentCalls: calls.filter((n) => n === 'bundlePersistentCg').length,
            distributedCalls: calls.filter((n) => n === 'bundleSchurPoint').length,
            chunks: out.execution.report.chunks,
            tier: out.execution.report.tier,
            cpuStages: out.execution.cpuStages,
          };
        } finally {
          await device.dispose();
          device = undefined;
        }
        const base = hydrate(fixtures.wide.input);
        const wideInput = {
          views: Array.from({ length: 64 }, (_, i) => {
            const view = base.views[i % 4];
            return {
              ...view,
              id: 100 + i,
              camera: { ...view.camera },
              pose: { R: view.pose.R.slice(), t: view.pose.t.slice() },
            };
          }),
          landmarks: base.landmarks.map((point) => ({
            ...point,
            xyz: point.xyz.slice(),
            observations: Array.from({ length: 64 }, (_, i) => ({
              ...point.observations[i % 4],
              viewId: 100 + i,
            })),
          })),
        };
        const wideSnapshot = JSON.stringify(wideInput);
        const wideOptions = {
          iterations: 3,
          refineSharedFocal: true,
          maxCgIterations: 255,
          cgRelativeTolerance: 1e-10,
          variableViewIds: [],
        };
        const expected = bundleAdjust(wideInput, wideOptions);
        device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0, solverCheckpoints: true });
        try {
          const out = await bundleAdjustGpu(device, wideInput, wideOptions);
          const calls = out.execution.report.groups.flatMap((g) => g.calls);
          wide = {
            diagnostics: out.diagnostics,
            expected: expected.diagnostics,
            cgIterations: out.cgIterations,
            failures: out.linearSolverFailures,
            fx: out.reconstruction.views[0].camera.fx,
            expectedFx: expected.reconstruction.views[0].camera.fx,
            persistentCalls: calls.filter((n) => n === 'bundlePersistentCg').length,
            distributedCalls: calls.filter((n) => n === 'bundleSchurPoint').length,
            chunks: out.execution.report.chunks,
            fixedPoses: out.reconstruction.views.every(
              (view, i) => JSON.stringify(view.pose) === JSON.stringify(wideInput.views[i].pose),
            ),
            sharedScale: out.reconstruction.views.every(
              (view, i) =>
                Math.abs(
                  view.camera.fx / wideInput.views[i].camera.fx -
                    out.reconstruction.views[0].camera.fx / wideInput.views[0].camera.fx,
                ) < 1e-12,
            ),
            inputUnchanged: JSON.stringify(wideInput) === wideSnapshot,
            tier: out.execution.report.tier,
            cpuStages: out.execution.cpuStages,
          };
        } finally {
          await device.dispose();
          device = undefined;
        }
      } finally {
        if (device) await device.dispose();
        gpu.destroy();
      }
      return { exhausted, wide, gpuErrors };
    },
    { fixtures, hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
    { timeoutMs: 300000 },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  const { exhausted, wide } = result;
  assert.equal(exhausted.tier, 'webgpu');
  assert.deepEqual(exhausted.cpuStages, []);
  assert.equal(exhausted.diagnostics.observations, 30000);
  assert.equal(exhausted.cgIterations, 9);
  // The solve never converges, so the chunks run all nine iterations and no more: the last chunk
  // holds what remains of the budget. The tenth `bundleSchurPoint` is the warm-up before the first
  // timed chunk of the fresh device, over zero workgroups.
  assert.equal(exhausted.persistentCalls, 0);
  assert.equal(exhausted.distributedCalls, 9 + 1);
  assert.ok(exhausted.chunks.bundleCg.chunks >= 2);
  assert.equal(exhausted.failures, 1);
  assert.equal(exhausted.diagnostics.acceptedSteps, 0);
  assert.equal(exhausted.diagnostics.rejectedSteps, 1);
  assert.equal(exhausted.diagnostics.finalCost, exhausted.diagnostics.initialCost);
  assert.ok(exhausted.unchanged && exhausted.inputUnchanged);
  assert.equal(wide.tier, 'webgpu');
  assert.deepEqual(wide.cpuStages, []);
  assert.equal(wide.diagnostics.observations, 1088);
  assert.equal(wide.persistentCalls, 0);
  assert.ok(wide.distributedCalls >= wide.cgIterations && wide.cgIterations > 0);
  assert.ok(wide.chunks.bundleCg.chunks > 0);
  assert.equal(wide.failures, 0);
  assert.ok(wide.fixedPoses && wide.sharedScale && wide.inputUnchanged);
  assert.ok(wide.diagnostics.finalCost < wide.diagnostics.initialCost);
  assert.ok(
    Math.abs(wide.diagnostics.finalCost - wide.expected.finalCost) < 1e-6,
    JSON.stringify(wide),
  );
  assert.ok(Math.abs(wide.fx - wide.expectedFx) < 1e-5, JSON.stringify(wide));
  console.log(JSON.stringify({ jointFocalChunkBoundaries: result }));
});
