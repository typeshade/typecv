import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('WebGPU Hough lines agree with the CPU reference', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, houghLines, houghLinesGpu } = await import('/dist/gpu.js');
    const size = 256,
      image = new Uint8Array(size * size * 4),
      strokes = [
        [0, 40],
        [30, 150],
        [90, 200],
        [135, -70],
      ];
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        let v = 20;
        for (const [degrees, rho] of strokes) {
          const t = (degrees * Math.PI) / 180,
            d = x * Math.cos(t) + y * Math.sin(t) - rho;
          if (d >= 0 && d < 12) v = 220;
        }
        image.set([v, v, v, 255], (y * size + x) * 4);
      }
    const adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice(),
      device = await ShadeDevice.open({ gpuDevice: gpu, backend: 'webgpu', poolBytes: 0 });
    try {
      const rows = [],
        absolute = { thresholdMode: 'absolute', threshold: 64 };
      for (const hysteresis of ['gpu', 'cpu']) {
        const cpu = houghLines(image, size, size, absolute),
          out = await houghLinesGpu(device, image, size, size, { ...absolute, hysteresis });
        let same = 0;
        for (let i = 0; i < cpu.edges.length; i++) if (cpu.edges[i] === out.edges[i]) same++;
        // The device accumulator must be the accumulator of its own edge map, exactly.
        const expected = new Uint32Array(out.accumulator.length);
        for (let p = 0; p < out.edges.length; p++) {
          if (!out.edges[p]) continue;
          const x = p % size,
            y = (p - x) / size;
          for (let t = 0; t < out.thetaBins; t++) {
            const c = Math.round(Math.cos((Math.PI * t) / out.thetaBins) * 65536),
              s = Math.round(Math.sin((Math.PI * t) / out.thetaBins) * 65536);
            expected[t * out.rhoCount + ((x * c + y * s + 32768) >> 16) + out.maxRho]++;
          }
        }
        rows.push({
          hysteresis,
          edgeAgreement: same / cpu.edges.length,
          accumulatorExact: expected.every((v, i) => v === out.accumulator[i]),
          sameLines: JSON.stringify(cpu.lines) === JSON.stringify(out.lines),
          sameSegments: JSON.stringify(cpu.segments) === JSON.stringify(out.segments),
          lines: out.lines.length,
          segments: out.segments.length,
        });
      }
      return rows;
    } finally {
      await device.dispose();
    }
  });
  assert.deepEqual(errors, []);
  for (const row of result) {
    assert.ok(row.edgeAgreement >= 0.999, `edges ${row.edgeAgreement} (${row.hysteresis})`);
    assert.equal(row.accumulatorExact, true, row.hysteresis);
    assert.ok(row.lines >= 8 && row.segments > 0, JSON.stringify(row));
    if (row.edgeAgreement === 1) {
      assert.equal(row.sameLines, true);
      assert.equal(row.sameSegments, true);
    }
  }
});
