import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('WebGPU focal scoring matches the CPU fine search on a non-grid focal scale', async () => {
  const hardware = process.env.TYPECV_HARDWARE_GPU === '1';
  const { result, errors } = await inBrowser(
    async ({ hardware }) => {
      const [{ ShadeDevice, bundleAdjustGpu }, { bundleAdjust }, { bundleFixture }] =
        await Promise.all([
          import('/dist/gpu.js'),
          import('/dist/reconstruction/bundle-adjustment.js'),
          import('/tests/typeshade/geometry-cases.mjs'),
        ]);
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('WebGPU adapter unavailable');
      const info = adapter.info;
      const adapterDescription = [info?.vendor, info?.architecture, info?.device, info?.description]
        .filter(Boolean)
        .join(' ');
      const software =
        /swiftshader|software|llvmpipe/i.test(adapterDescription) ||
        adapter.isFallbackAdapter === true;
      if (hardware && software)
        throw new Error('Hardware requested, but a software adapter was selected');

      const gpu = await adapter.requestDevice();
      const gpuErrors = [];
      gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      const model = bundleFixture({ outlier: true });
      for (const view of model.views) {
        view.camera.fx /= 1.09;
        view.camera.fy /= 1.09;
      }
      const options = { iterations: 0, huberPx: 1, refineSharedFocal: true };
      const cpu = bundleAdjust(model, options);
      const device = await ShadeDevice.open({ gpuDevice: gpu });
      let actual;
      try {
        actual = await bundleAdjustGpu(device, model, options);
      } finally {
        await device.dispose();
      }
      // The device intentionally retains reusable buffers in its pool until disposal.
      const liveAfterSuccess = device.statistics.liveBuffers;

      // The score-only search must reject its full resident working-set budget before allocation.
      const constrained = await ShadeDevice.open({ gpuDevice: gpu, workingBytes: 4096 });
      let budgetError = '';
      try {
        await bundleAdjustGpu(constrained, model, options);
      } catch (error) {
        budgetError = String(error);
      } finally {
        await constrained.dispose();
      }
      const liveAfterBudgetFailure = constrained.statistics.liveBuffers;
      gpu.destroy();
      return {
        adapterDescription,
        software,
        backend: actual.execution.backend,
        tier: actual.execution.report.tier,
        dispatches: actual.execution.report.dispatches,
        cpuStages: actual.execution.cpuStages,
        actualCameras: actual.reconstruction.views.map((view) => view.camera),
        expectedCameras: cpu.reconstruction.views.map((view) => view.camera),
        actualCost: actual.diagnostics.finalCost,
        expectedCost: cpu.diagnostics.finalCost,
        expectedScale: cpu.reconstruction.views[0].camera.fx / model.views[0].camera.fx,
        liveAfterSuccess,
        liveAfterBudgetFailure,
        budgetError,
        gpuErrors,
      };
    },
    { hardware },
    { timeoutMs: 300000 },
  );

  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  assert.equal(result.backend, 'webgpu');
  assert.equal(result.tier, 'webgpu');
  assert.deepEqual(result.cpuStages, []);
  assert.ok(result.dispatches < 20);
  assert.equal(result.liveAfterSuccess, 0);
  assert.equal(result.liveAfterBudgetFailure, 0);
  assert.match(result.budgetError, /budget/i);
  assert.ok(result.expectedScale > 1.05 && result.expectedScale < 1.13);
  assert.equal(result.actualCameras.length, result.expectedCameras.length);
  for (let i = 0; i < result.actualCameras.length; i++) {
    assert.ok(Math.abs(result.actualCameras[i].fx - result.expectedCameras[i].fx) < 1e-7);
    assert.ok(Math.abs(result.actualCameras[i].fy - result.expectedCameras[i].fy) < 1e-7);
  }
  assert.ok(Math.abs(result.actualCost - result.expectedCost) < 1e-7);
});
