import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('WebGPU SLIC agrees with the CPU reference and keeps every label one component', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, slic, slicGpu } = await import('/dist/gpu.js');
    const adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice(),
      device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0 }),
      width = 192,
      height = 144,
      rgba = new Uint8Array(width * height * 4),
      boxes = [
        [0, 0, 0.45, 0.5, [200, 40, 40]],
        [0.45, 0, 1, 0.3, [40, 90, 210]],
        [0.45, 0.3, 0.8, 0.75, [230, 200, 50]],
        [0.8, 0.3, 1, 1, [60, 170, 80]],
        [0, 0.5, 0.45, 1, [150, 60, 190]],
        [0.45, 0.75, 0.8, 1, [30, 30, 40]],
      ];
    let state = 7;
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const box = boxes.find(
          ([x0, y0, x1, y1]) =>
            x >= x0 * width && x < x1 * width && y >= y0 * height && y < y1 * height,
        );
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        const noise = ((state >>> 8) % 5) - 2;
        for (let c = 0; c < 3; c++) rgba[(y * width + x) * 4 + c] = box[4][c] + noise;
        rgba[(y * width + x) * 4 + 3] = 255;
      }
    try {
      const rows = [];
      for (const options of [{ nSegments: 30 }, {}]) {
        const cpu = slic(rgba, width, height, options),
          out = await slicGpu(device, rgba, width, height, options);
        let same = 0;
        for (let i = 0; i < cpu.labels.length; i++) if (cpu.labels[i] === out.labels[i]) same++;
        // Components per label.
        const seen = new Uint8Array(out.labels.length);
        let components = 0;
        for (let s = 0; s < seen.length; s++) {
          if (seen[s]) continue;
          components++;
          const stack = [s];
          seen[s] = 1;
          while (stack.length) {
            const p = stack.pop(),
              x = p % width;
            for (const q of [
              x > 0 && p - 1,
              x < width - 1 && p + 1,
              p >= width && p - width,
              p + width < seen.length && p + width,
            ])
              if (q !== false && !seen[q] && out.labels[q] === out.labels[p]) {
                seen[q] = 1;
                stack.push(q);
              }
          }
        }
        rows.push({ agreement: same / cpu.labels.length, count: out.count, components });
      }
      return rows;
    } finally {
      await device.dispose();
    }
  });
  assert.deepEqual(errors, []);
  for (const row of result) {
    assert.ok(row.agreement >= 0.99, `agreement ${row.agreement}`);
    assert.equal(row.components, row.count, 'one component per label');
  }
});
