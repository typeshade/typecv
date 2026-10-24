import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { gridPoints, renderScene } from '../klt-cases.ts';

test('WebGPU pyramid reuse preserves tracking while reducing image uploads and record snapshots', async () => {
  const width = 128,
    height = 96,
    points = Array.from(gridPoints(width, height, 30, 25, 3, 3)),
    frames = Array.from({ length: 4 }, (_, id) => ({
      id,
      width,
      height,
      gray: Array.from(renderScene({ width, height }, { dx: 2 * id, dy: -id })),
    }));
  const { result, errors } = await inBrowser(
    async (input) => {
      const { ShadeDevice, createKltPairCache, kltOptionsFor, trackKltGpu } =
        await import('/dist/internal/gpu.js');
      const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      const frames = input.frames.map((f) => ({ ...f, gray: Float32Array.from(f.gray) })),
        options = kltOptionsFor(input.width, { levels: 3, halfWindow: 4 }),
        pairs = [
          [0, 1],
          [1, 0],
          [1, 2],
          [2, 1],
          [2, 3],
          [3, 2],
        ],
        before = device.statistics,
        baseline = [];
      try {
        for (const [a, b] of pairs) {
          const r = await trackKltGpu(
            device,
            frames[a].gray,
            frames[b].gray,
            input.width,
            input.height,
            input.points,
            options,
          );
          baseline.push({ records: Array.from(r.records), forward: Array.from(r.forward) });
        }
        const middle = device.statistics,
          cache = createKltPairCache(device),
          cached = [];
        try {
          for (const [a, b] of pairs) {
            const r = await cache.track(frames[a], frames[b], input.points, options, {
              readBack: false,
            });
            cached.push({
              records: Array.from(r.records),
              forward: Array.from(r.forward),
              backLength: r.back.length,
            });
          }
          const after = device.statistics;
          return {
            baseline,
            cached,
            statistics: cache.statistics(),
            baselineUpload: middle.applicationUploadBytes - before.applicationUploadBytes,
            cachedUpload: after.applicationUploadBytes - middle.applicationUploadBytes,
            baselineSnapshot: middle.residentSnapshotBytes - before.residentSnapshotBytes,
            cachedSnapshot: after.residentSnapshotBytes - middle.residentSnapshotBytes,
            liveAfterDispose: await cache.dispose().then(() => device.statistics.liveBytes),
          };
        } finally {
          await cache.dispose();
        }
      } finally {
        await device.dispose();
      }
    },
    { width, height, frames, points },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.liveAfterDispose, 0);
  assert.equal(result.statistics.pyramidBuilds, 4);
  assert.equal(result.statistics.peakFrames, 3);
  const total = width * height + (width / 2) * (height / 2) + (width / 4) * (height / 4);
  assert.equal(
    result.cachedUpload,
    result.statistics.pyramidBuilds * total * 4 +
      result.statistics.pyramidAllocations * total * 8 +
      6 * (points.length / 2) * 16,
  );
  assert.ok(result.cachedUpload < result.baselineUpload);
  assert.equal(result.cachedSnapshot * 3, result.baselineSnapshot * 2);
  for (let k = 0; k < result.baseline.length; k++) {
    assert.equal(result.cached[k].backLength, 0);
    for (const key of ['records', 'forward']) {
      assert.equal(result.cached[k][key].length, result.baseline[k][key].length);
      for (let i = 0; i < result.baseline[k][key].length; i++) {
        if (i % 4 === 3) assert.equal(result.cached[k][key][i], result.baseline[k][key][i]);
        else assert.ok(Math.abs(result.cached[k][key][i] - result.baseline[k][key][i]) < 1e-4);
      }
    }
  }
  console.log(
    JSON.stringify({
      ...result.statistics,
      baselineUpload: result.baselineUpload,
      cachedUpload: result.cachedUpload,
      baselineSnapshot: result.baselineSnapshot,
      cachedSnapshot: result.cachedSnapshot,
    }),
  );
});
