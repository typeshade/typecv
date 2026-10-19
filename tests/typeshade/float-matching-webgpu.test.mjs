import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { floatDescriptorFixture, floatMatchOracle } from './float-matching-cases.mjs';

test('WebGPU exhaustive float matcher preserves L2 neighbours, tails and strict mutual ratio', async () => {
  const fixtures = [];
  for (const [n, m, dimensions] of [
    [2, 3, 1],
    [7, 9, 15],
    [9, 7, 17],
    [17, 19, 33],
    [65, 67, 128],
  ]) {
    const a = floatDescriptorFixture(n, dimensions),
      b = floatDescriptorFixture(m, dimensions, 29);
    for (let i = 0; i < Math.min(n, m); i++)
      for (let d = 0; d < dimensions; d++) b[i * dimensions + d] = a[i * dimensions + d] + 0.001;
    for (const mutual of [false, true])
      fixtures.push({ a: [...a], b: [...b], dimensions, options: { mutual } });
  }
  fixtures.push(
    { a: [0, 10], b: [0, 0, 10], dimensions: 1, options: {} },
    { a: [0, 10], b: [1, 2], dimensions: 1, options: { ratio: 0.5, mutual: false } },
    { a: [0, 4], b: [1, 9], dimensions: 1, options: { maxDistance: 1, mutual: false } },
    { a: [3e38, 3e38], b: [3e38, 3e38], dimensions: 1, options: {} },
  );
  // 4,004,001 pairs: exceeds the legacy 2,000-feature / four-million-pair cap.
  // All distances and the 0.25 offset are exactly representable in f32.
  fixtures.push({
    a: Array.from({ length: 2001 }, (_, i) => i * 4),
    b: Array.from({ length: 2001 }, (_, i) => i * 4 + 0.25),
    dimensions: 1,
    options: {},
  });
  const { result, errors } = await inBrowser(
    async ({ fixtures, hardware }) => {
      const { ShadeDevice, matchFloatDescriptorsGpu } = await import('/dist/gpu.js');
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('WebGPU adapter unavailable');
      const info = adapter.info;
      const adapterDescription = [info?.vendor, info?.architecture, info?.device, info?.description]
        .filter(Boolean)
        .join(' ');
      const software =
        /swiftshader|software|llvmpipe/i.test(adapterDescription) ||
        adapter.isFallbackAdapter === true;
      if (hardware && software) throw new Error('Hardware requested but software adapter selected');
      const gpu = await adapter.requestDevice(),
        gpuErrors = [];
      gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      const device = await ShadeDevice.open({ gpuDevice: gpu });
      const rows = [],
        invalid = [];
      let aliasError = '';
      try {
        for (const fixture of fixtures) {
          const a = Float32Array.from(fixture.a),
            b = Float32Array.from(fixture.b),
            left = device.allocate(a.length, a),
            right = device.allocate(b.length, b);
          try {
            if (!aliasError) {
              const descriptor = {
                descriptors: left,
                count: a.length / fixture.dimensions,
                dimensions: fixture.dimensions,
                elements: a.length,
              };
              try {
                await matchFloatDescriptorsGpu(device, descriptor, descriptor);
              } catch (error) {
                aliasError = String(error);
              }
            }
            const r = await matchFloatDescriptorsGpu(
              device,
              {
                descriptors: left,
                count: a.length / fixture.dimensions,
                dimensions: fixture.dimensions,
                elements: a.length,
              },
              {
                descriptors: right,
                count: b.length / fixture.dimensions,
                dimensions: fixture.dimensions,
                elements: b.length,
              },
              fixture.options,
            );
            rows.push({
              matches: [...r.matches],
              distances: [...r.distances],
              tier: r.report.tier,
              dispatches: r.report.dispatches,
              execution: r.execution,
            });
          } finally {
            device.drop(left);
            device.drop(right);
          }
        }
        for (const value of [NaN, Infinity, -Infinity, 3e38]) {
          const buffer = device.allocate(2, Float32Array.of(value, 0)),
            otherBuffer = device.allocate(2, Float32Array.of(value, 0)),
            descriptor = { descriptors: buffer, count: 2, dimensions: 1, elements: 2 };
          try {
            await matchFloatDescriptorsGpu(device, descriptor, {
              ...descriptor,
              descriptors: otherBuffer,
            });
            invalid.push('accepted');
          } catch (error) {
            invalid.push(String(error));
          } finally {
            device.drop(buffer);
            device.drop(otherBuffer);
          }
        }
      } finally {
        await device.dispose();
        gpu.destroy();
      }
      return {
        rows,
        invalid,
        aliasError,
        adapterDescription,
        software,
        gpuErrors,
        liveBuffers: device.statistics.liveBuffers,
      };
    },
    { fixtures, hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
    { timeoutMs: 300000 },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  assert.equal(result.liveBuffers, 0);
  assert.match(result.aliasError, /distinct source and target Resident/);
  result.rows.forEach((row, i) => {
    const fixture = fixtures[i],
      expected = floatMatchOracle(
        Float32Array.from(fixture.a),
        Float32Array.from(fixture.b),
        fixture.dimensions,
        fixture.options,
      );
    assert.deepEqual(row.matches, expected.matches);
    row.distances.forEach((distance, j) =>
      assert.ok(
        Math.abs(distance - expected.distances[j]) <= 2e-5 * Math.max(1, expected.distances[j]),
      ),
    );
    assert.equal(row.tier, 'webgpu');
    assert.equal(row.dispatches, 6);
    assert.equal(row.execution.applicationUploadBytes, 0);
    assert.equal(row.execution.applicationReadbackBytes, 8 + 12 * expected.distances.length);
  });
  result.invalid.forEach((error) => assert.match(error, /Nonfinite|overflow/));
});
