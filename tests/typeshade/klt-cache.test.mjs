import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  createKltPairCache,
  kltLayout,
  kltOptionsFor,
  trackKltGpu,
} from '../../dist/internal/gpu.js';
import { gridPoints, renderScene } from '../klt-cases.ts';

const width = 128,
  height = 96,
  options = kltOptionsFor(width, { levels: 3, halfWindow: 4 }),
  points = gridPoints(width, height, 30, 25, 3, 3),
  frames = Array.from({ length: 4 }, (_, id) => ({
    id,
    width,
    height,
    gray: renderScene({ width, height }, { dx: 2 * id, dy: -id }),
  })),
  pairs = [
    [0, 1],
    [1, 0],
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],
  ];

async function withDevice(run, extra = {}) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, ...extra });
  try {
    await run(device);
  } finally {
    await device.dispose();
  }
}

test('bounded KLT frame reuse preserves measurements and cuts image uploads/readbacks', async () => {
  await withDevice(async (device) => {
    const before = device.statistics,
      baseline = [];
    for (const [a, b] of pairs)
      baseline.push(
        await trackKltGpu(device, frames[a].gray, frames[b].gray, width, height, points, options),
      );
    const middle = device.statistics;
    assert.equal(middle.liveBytes, 0);
    const cache = createKltPairCache(device);
    try {
      for (let i = 0; i < pairs.length; i++) {
        const [a, b] = pairs[i],
          actual = await cache.track(frames[a], frames[b], points, options, { readBack: false });
        assert.deepEqual(actual.records, baseline[i].records);
        assert.deepEqual(actual.forward, baseline[i].forward);
        assert.equal(actual.back.length, 0);
      }
      const after = device.statistics,
        stats = cache.statistics(),
        layout = kltLayout(width, height, options.levels),
        baselineUpload = middle.applicationUploadBytes - before.applicationUploadBytes,
        cachedUpload = after.applicationUploadBytes - middle.applicationUploadBytes,
        baselineReadback = middle.applicationReadbackBytes - before.applicationReadbackBytes,
        cachedReadback = after.applicationReadbackBytes - middle.applicationReadbackBytes;
      assert.equal(stats.pyramidBuilds, 4);
      assert.equal(stats.misses, 4);
      assert.equal(stats.hits, 8);
      assert.equal(stats.evictions, 1);
      assert.equal(stats.frames, 3);
      assert.equal(stats.peakFrames, 3);
      assert.equal(stats.peakResidentBytes, 3 * 12 * layout.total);
      assert.equal(after.liveBytes, stats.residentBytes);
      assert.equal(
        cachedUpload,
        stats.pyramidBuilds * layout.total * 4 +
          stats.pyramidAllocations * layout.total * 8 +
          pairs.length * (points.length / 2) * 16,
        'gray refreshes, initial gradient mirrors and point records are all counted',
      );
      assert.ok(cachedUpload < baselineUpload, `${cachedUpload} vs ${baselineUpload}`);
      assert.equal(cachedReadback * 3, baselineReadback * 2);
      console.log(
        JSON.stringify({
          baselineUpload,
          cachedUpload,
          baselineReadback,
          cachedReadback,
          ...stats,
        }),
      );
    } finally {
      await cache.dispose();
    }
    assert.equal(device.statistics.liveBytes, 0);
    await cache.dispose();
    await assert.rejects(cache.track(frames[0], frames[1], points, options), /disposed/);
  });
});

test('cache rebuilds replaced evidence and changed pyramid levels, without stale measurements', async () => {
  await withDevice(async (device) => {
    const cache = createKltPairCache(device, { capacity: 2 });
    try {
      await cache.track(frames[0], frames[1], points, options);
      const replacement = { ...frames[1], gray: renderScene({ width, height }, { dx: 3, dy: -2 }) },
        expected = await trackKltGpu(
          device,
          frames[0].gray,
          replacement.gray,
          width,
          height,
          points,
          options,
        ),
        actual = await cache.track(frames[0], replacement, points, options);
      assert.deepEqual(actual.records, expected.records);
      assert.deepEqual(actual.back, expected.back);
      assert.equal(cache.statistics().pyramidBuilds, 3);
      const nextOptions = { ...options, levels: 2 },
        expectedNext = await trackKltGpu(
          device,
          frames[0].gray,
          replacement.gray,
          width,
          height,
          points,
          nextOptions,
        ),
        actualNext = await cache.track(frames[0], replacement, points, nextOptions);
      assert.deepEqual(actualNext.records, expectedNext.records);
      assert.equal(cache.statistics().pyramidBuilds, 5);
      await assert.rejects(
        cache.track(frames[0], { ...frames[1], width: width + 1 }, points, options),
        /frame|coordinates/,
      );
    } finally {
      await cache.dispose();
    }
    assert.equal(device.statistics.liveBytes, 0);
  });
});

test('a partial pyramid allocation failure releases its buffers', async () => {
  const total = kltLayout(width, height, options.levels).total;
  await withDevice(
    async (device) => {
      const cache = createKltPairCache(device);
      try {
        await assert.rejects(cache.track(frames[0], frames[1], points, options), /working budget/);
        assert.equal(cache.statistics().frames, 0);
        assert.equal(device.statistics.liveBytes, 0);
      } finally {
        await cache.dispose();
      }
    },
    { workingBytes: total * 16 - 4 },
  );
});

test('queued cache calls finish before idempotent disposal', async () => {
  await withDevice(async (device) => {
    const cache = createKltPairCache(device),
      a = cache.track(frames[0], frames[1], points, options),
      b = cache.track(frames[1], frames[2], points, options),
      disposed = cache.dispose();
    const results = await Promise.all([a, b]);
    await disposed;
    assert.equal(results.length, 2);
    assert.equal(cache.statistics().pyramidBuilds, 3);
    assert.equal(cache.statistics().frames, 0);
    assert.equal(device.statistics.liveBytes, 0);
  });
});

test('cache evicts an unprotected frame to respect the device budget below capacity', async () => {
  const pyramidBytes = kltLayout(width, height, options.levels).total * 12;
  await withDevice(
    async (device) => {
      const cache = createKltPairCache(device, { capacity: 3 });
      try {
        const first = await cache.track(frames[0], frames[1], points, options);
        const second = await cache.track(frames[1], frames[2], points, options);
        assert.equal(first.records.length, second.records.length);
        const stats = cache.statistics();
        assert.equal(stats.frames, 2);
        assert.equal(stats.peakFrames, 2);
        assert.equal(stats.evictions, 1);
        assert.equal(stats.pyramidBuilds, 3);
        assert.ok(device.statistics.liveBytes <= pyramidBytes * 2.5);
      } finally {
        await cache.dispose();
      }
      assert.equal(device.statistics.liveBytes, 0);
    },
    { workingBytes: Math.floor(pyramidBytes * 2.5) },
  );
});
