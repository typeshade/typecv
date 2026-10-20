import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('WebGPU CLAHE agrees with the CPU reference and SIFT with clahe finds more features on a bright low-contrast image', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, equalizeGrayCpu, equalizeGrayGpu, extractSiftGpuResident } =
      await import('/dist/gpu.js');
    const adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice(),
      device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0 }),
      width = 131,
      height = 97,
      gray = new Float32Array(width * height);
    let state = 99;
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        gray[y * width + x] =
          0.82 +
          0.03 * Math.sin(x / 9) * Math.cos(y / 7) +
          0.012 * (((state >>> 8) % 1000) / 1000 - 0.5);
      }
    try {
      const diffs = [];
      for (const options of [true, { tiles: [5, 3], clipLimit: 0.02 }]) {
        const cpu = equalizeGrayCpu(gray, width, height, options),
          out = await equalizeGrayGpu(device, gray, width, height, options);
        let worst = 0;
        for (let i = 0; i < cpu.length; i++) worst = Math.max(worst, Math.abs(cpu[i] - out[i]));
        diffs.push(worst);
      }
      const rgba = new Uint8Array(width * height * 4);
      for (let i = 0; i < gray.length; i++) {
        rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = Math.round(gray[i] * 255);
        rgba[i * 4 + 3] = 255;
      }
      const counts = [];
      for (const clahe of [false, true]) {
        const sift = await extractSiftGpuResident(device, rgba, width, height, {
          maxFeatures: 2048,
          clahe,
        });
        counts.push(sift.count);
        await sift.dispose();
      }
      return { diffs, counts };
    } finally {
      await device.dispose();
    }
  });
  assert.deepEqual(errors, []);
  for (const worst of result.diffs) assert.ok(worst < 2.5e-4, `max |gpu - cpu| = ${worst}`);
  assert.ok(result.counts[1] > result.counts[0], `features ${result.counts}`);
});
