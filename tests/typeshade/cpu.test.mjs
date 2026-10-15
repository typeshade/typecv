import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  EpipolarSession,
  estimateEssentialRansacGpu,
  reconstructTwoViewGpu,
} from '../../dist/gpu.js';
import { fixture, reference, reconstructionFixture, poseMetrics } from './cases.mjs';

test('compiled TypeShade CPU tier agrees with independent f64 scoring and tie order', async () => {
  const data = fixture(),
    expected = reference(data),
    d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const s = await EpipolarSession.create(d, data.pixels, data.first, data.second);
    const r = await s.score(data.models, data.threshold, { readMask: true, readStatistics: true });
    assert.equal(r.report.tier, 'cpu');
    assert.equal(r.inlierCount, expected.counts[2]);
    assert.equal(r.bestIndex, 2);
    assert.deepEqual(Array.from(r.mask), expected.masks[2]);
    const h = data.models.length / 9;
    for (let i = 0; i < h; i++) {
      assert.equal(r.modelStatistics[i], expected.counts[i]);
      assert.ok(Math.abs(r.modelStatistics[h + i] - expected.costs[i]) < 0.002);
    }
    await s.dispose();
  } finally {
    await d.dispose();
  }
});

test('resident observations, reuse, serialized queries and concurrent disposal have defined lifetimes', async () => {
  const data = fixture(31),
    d = await ShadeDevice.open({ backend: 'cpu' });
  const s = await EpipolarSession.create(d, data.pixels, data.first, data.second);
  const before = d.statistics;
  const a = await s.score(data.models, data.threshold),
    mid = d.statistics;
  const [b, c] = await Promise.all([
    s.score(data.models, data.threshold),
    s.score(data.models, data.threshold),
  ]);
  assert.deepEqual([a.bestIndex, b.bestIndex, c.bestIndex], [2, 2, 2]);
  assert.equal(
    d.statistics.applicationUploadBytes - before.applicationUploadBytes,
    3 * data.models.length * 4,
  );
  assert.equal(d.statistics.applicationReadbackBytes - before.applicationReadbackBytes, 3 * 12);
  assert.equal(d.statistics.allocations, mid.allocations);
  assert.ok(d.statistics.reuses > 0);
  const pending = s.score(data.models, data.threshold);
  const closing = s.dispose();
  await pending;
  await closing;
  await assert.rejects(s.score(data.models, data.threshold), /disposed/);
  const dispose = d.dispose();
  assert.equal(d.dispose(), dispose);
  await dispose;
  assert.equal(d.statistics.liveBuffers, 0);
  await assert.rejects(
    d.enqueue(async () => 0),
    /disposed/,
  );
});

test('strict tier selection and f32/storage bounds reject invalid work', async () => {
  const data = fixture(10),
    d = await ShadeDevice.open({ backend: 'cpu', maxModelsPerBatch: 2 });
  try {
    await assert.rejects(ShadeDevice.open({ backend: 'cpu' }), /active/);
    const s = await EpipolarSession.create(d, data.pixels, data.first, data.second);
    assert.equal(s.capacity, 2);
    assert.throws(() => s.score(data.models, data.threshold), RangeError);
    assert.throws(() => s.score(data.models.slice(0, 9), 1e-30), RangeError);
    assert.throws(() => s.score(data.models.slice(0, 9), 1e-20), RangeError);
    assert.throws(
      () => EpipolarSession.create(d, data.pixels, { ...data.first, k1: 0.1 }, data.second),
      /undistorted/,
    );
    assert.throws(
      () => EpipolarSession.create(d, Float64Array.of(Infinity, 0, 0, 0), data.first, data.second),
      RangeError,
    );
    const huge = { fx: 1e38, fy: 1e38, cx: -3e38, cy: 0 };
    assert.throws(
      () => EpipolarSession.create(d, Float64Array.of(3e38, 0, 0, 0), huge, data.second),
      /f32 range/,
    );
    await s.dispose();
  } finally {
    await d.dispose();
  }
});

test('failed readback destroys temporary buffers and leaves a reusable session', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' }),
    data = fixture(257);
  try {
    const s = await EpipolarSession.create(d, data.pixels, data.first, data.second);
    const original = d.read;
    d.read = async () => {
      throw new Error('injected readback failure');
    };
    await assert.rejects(s.score(data.models, data.threshold), /injected/);
    assert.equal(d.statistics.liveBuffers, 1);
    assert.equal(d.statistics.pooledBytes, 0);
    d.read = original;
    assert.equal((await s.score(data.models, data.threshold)).bestIndex, 2);
    await s.dispose();
  } finally {
    await d.dispose();
  }
});

test('a failed upload cannot strand a buffer removed from the free pool', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const buffer = d.allocate(9, new Float32Array(9));
    d.recycle(buffer);
    buffer.write = () => {
      throw new Error('injected upload failure');
    };
    assert.throws(() => d.allocate(9, new Float32Array(9)), /injected upload/);
    assert.equal(d.statistics.liveBuffers, 0);
    assert.equal(d.statistics.pooledBytes, 0);
  } finally {
    await d.dispose();
  }
});

test('reduction block tails, zero-pool budgets and caller snapshots are deterministic', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, workingBytes: 4096 });
  try {
    for (const n of [1, 255, 256, 257]) {
      const data = fixture(n),
        expected = reference(data),
        a = { ...data.first };
      const creating = EpipolarSession.create(d, data.pixels, a, data.second);
      a.fx = 1;
      data.pixels.fill(0);
      const s = await creating;
      const options = { readMask: true };
      const pending = s.score(data.models.subarray(18, 27), data.threshold, options);
      options.readMask = false;
      const r = await pending;
      assert.deepEqual([...r.mask], expected.masks[2]);
      assert.equal(r.inlierCount, expected.counts[2]);
      assert.equal(d.statistics.pooledBytes, 0);
      await s.dispose();
      assert.equal(d.statistics.liveBuffers, 0);
    }
    const data = fixture(400);
    await assert.rejects(EpipolarSession.create(d, data.pixels, data.first, data.second), /budget/);
    assert.equal(d.statistics.liveBuffers, 0);
  } finally {
    await d.dispose();
  }
});

test('missing WebGPU rejects explicitly and failed opening releases the singleton', async () => {
  await assert.rejects(ShadeDevice.open());
  const d = await ShadeDevice.open({ backend: 'cpu' });
  await d.dispose();
});

test('hybrid RANSAC reconstructs a known noisy pose and obeys adaptive batch/trial caps', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' }),
    data = reconstructionFixture();
  try {
    const options = {
      iterations: 500,
      minIterations: 32,
      threshold: 0.001,
      batchSize: 16,
      seed: 71,
    };
    const result = await reconstructTwoViewGpu(d, data.pixels, data.first, data.second, options);
    const metric = poseMetrics(result, data);
    assert.ok(metric.valid >= 250, JSON.stringify(metric));
    assert.ok(metric.rotationErrorDeg < 0.1, JSON.stringify(metric));
    assert.ok(metric.translationCosine > 0.999, JSON.stringify(metric));
    assert.ok(metric.errorPx < 0.15);
    assert.equal(metric.execution.backend, 'cpu');
    assert.ok(metric.execution.hypothesisBatches > 1);
    assert.ok(metric.execution.cpuStages.includes('triangulation'));
    assert.ok(metric.trials >= 32 && metric.trials <= 500);
    const capped = await estimateEssentialRansacGpu(d, data.pixels, data.first, data.second, {
      ...options,
      iterations: 17,
      minIterations: 17,
      batchSize: 64,
    });
    assert.equal(capped.ransacIterations, 17);
    assert.equal(capped.execution.hypothesisBatches, 1);
    await assert.rejects(
      estimateEssentialRansacGpu(d, data.pixels, data.first, data.second, { batchSize: 0 }),
      RangeError,
    );
    const coincident = new Float64Array(40 * 4).fill(200);
    await assert.rejects(
      estimateEssentialRansacGpu(d, coincident, data.first, data.second, { iterations: 4 }),
      /valid two-view/,
    );
  } finally {
    await d.dispose();
  }
});
