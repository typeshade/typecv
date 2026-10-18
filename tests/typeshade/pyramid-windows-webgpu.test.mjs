import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('WebGPU image windows preserve full-image pyramid outputs and obey each dispatch limit', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, extractFeaturesGpuPyramidResident } = await import('/dist/gpu.js');
    const adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice(),
      device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0 }),
      width = 101,
      height = 99,
      pixels = new Uint8Array(width * height * 4),
      output = [];
    let seed = 0x7f42ac13;
    for (let i = 0; i < pixels.length; i++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      pixels[i] = seed >>> 24;
    }
    const originalLimit = device.maxKernelElements,
      oldDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups,
      dispatches = [];
    let capture = false;
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      if (capture) dispatches.push(args);
      return oldDispatch.apply(this, args);
    };
    gpu.pushErrorScope('validation');
    try {
      for (const maxFeatures of [4, 80, 2000])
        for (const smoothDescriptors of [false, true]) {
          const options = { maxFeatures, threshold: 0.045, smoothDescriptors },
            reference = await extractFeaturesGpuPyramidResident(
              device,
              pixels,
              width,
              height,
              options,
            );
          const expected = await reference.read();
          await reference.dispose();
          device.maxKernelElements = 1088;
          capture = true;
          const chunked = await extractFeaturesGpuPyramidResident(
            device,
            pixels,
            width,
            height,
            options,
          );
          const actual = await chunked.read();
          await chunked.dispose();
          capture = false;
          device.maxKernelElements = originalLimit;
          output.push({
            maxFeatures,
            count: chunked.count,
            expectedCount: expected.scores.length,
            capacity: chunked.capacity,
            expectedCapacity: expected.capacity,
            tier: actual.report.tier,
            equalKeypoints: actual.keypoints.every((value, i) => value === expected.keypoints[i]),
            equalScores: actual.scores.every((value, i) => value === expected.scores[i]),
            equalDescriptors: actual.descriptors.every(
              (value, i) => value === expected.descriptors[i],
            ),
            equalLengths:
              actual.keypoints.length === expected.keypoints.length &&
              actual.scores.length === expected.scores.length &&
              actual.descriptors.length === expected.descriptors.length,
            referenceDispatches: expected.report.dispatches,
            windowDispatches: actual.report.dispatches,
          });
        }
      await gpu.queue.onSubmittedWorkDone();
      const validationError = await gpu.popErrorScope();
      return {
        output,
        dispatches,
        validationError: validationError?.message ?? null,
        liveBytes: device.statistics.liveBytes,
      };
    } finally {
      GPUComputePassEncoder.prototype.dispatchWorkgroups = oldDispatch;
      await device.dispose();
      gpu.destroy();
    }
  });
  assert.deepEqual(errors, []);
  assert.equal(result.validationError, null);
  assert.equal(result.liveBytes, 0);
  assert.ok(result.dispatches.length > 0);
  for (const [x, y = 1, z = 1] of result.dispatches) {
    assert.ok(x <= 17, `window dispatch ${x} exceeds 17 workgroups`);
    assert.equal(y, 1);
    assert.equal(z, 1);
  }
  for (const row of result.output) {
    assert.ok(row.count > 0 && row.count <= row.maxFeatures);
    assert.equal(row.count, row.expectedCount);
    assert.equal(row.capacity, row.expectedCapacity);
    assert.equal(row.tier, 'webgpu');
    assert.equal(row.equalLengths, true);
    assert.equal(row.equalKeypoints, true);
    assert.equal(row.equalScores, true);
    assert.equal(row.equalDescriptors, true);
    assert.ok(row.windowDispatches > row.referenceDispatches);
  }
});
