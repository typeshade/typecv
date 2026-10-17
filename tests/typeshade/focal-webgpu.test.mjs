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

test('WebGPU global focal row beyond lane64 works with 64 fixed cameras in persistent and bounded schedules', async () => {
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
