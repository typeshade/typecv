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
  tracking = kltOptionsFor(width, { levels: 3, halfWindow: 4 }),
  layout = kltLayout(width, height, tracking.levels),
  pyramidBytes = layout.total * 12,
  points = gridPoints(width, height, 30, 25, 3, 3),
  frames = Array.from({ length: 4 }, (_, id) => ({
    id,
    width,
    height,
    gray: renderScene({ width, height }, { dx: id * 2, dy: -id }),
  })),
  activeBytes = (device) => device.statistics.liveBytes - device.statistics.pooledBytes;

async function withDevice(run, options = {}) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, ...options });
  try {
    await run(device);
  } finally {
    await device.dispose();
  }
}

test('compatible cache victims reuse buffers with exact measurements and honest upload counts', async () => {
  await withDevice(async (device) => {
    const pairs = [
        [0, 1],
        [2, 1],
        [3, 1],
        [0, 1],
      ],
      baseline = [];
    for (const [from, to] of pairs)
      baseline.push(
        await trackKltGpu(
          device,
          frames[from].gray,
          frames[to].gray,
          width,
          height,
          points,
          tracking,
        ),
      );
    const before = device.statistics,
      cache = createKltPairCache(device, { capacity: 2 });
    try {
      for (let i = 0; i < pairs.length; i++) {
        const [from, to] = pairs[i],
          actual = await cache.track(frames[from], frames[to], points, tracking);
        for (const name of ['records', 'forward', 'back'])
          assert.deepEqual(actual[name], baseline[i][name]);
      }
      const stats = cache.statistics();
      assert.equal(stats.pyramidAllocations, 2);
      assert.equal(stats.bufferReuses, 6);
      assert.equal(stats.pyramidBuilds, 5);
      assert.equal(stats.evictions, 3);
      assert.equal(device.statistics.allocations - before.allocations, 4 + 4 * pairs.length);
      assert.equal(
        device.statistics.applicationUploadBytes - before.applicationUploadBytes,
        5 * layout.total * 4 + (points.length / 2) * 16 * pairs.length,
        'in-place image writes count the same uploads as newly allocated images',
      );
    } finally {
      await cache.dispose();
    }
    assert.equal(activeBytes(device), 0);
  });
});

test('replacing same-id image evidence reuses compatible storage and recomputes gradients', async () => {
  await withDevice(async (device) => {
    const cache = createKltPairCache(device, { capacity: 2 });
    try {
      await cache.track(frames[0], frames[1], points, tracking);
      const replaced = { ...frames[1], gray: renderScene({ width, height }, { dx: 3, dy: -2 }) },
        before = cache.statistics(),
        expected = await trackKltGpu(
          device,
          frames[0].gray,
          replaced.gray,
          width,
          height,
          points,
          tracking,
        ),
        actual = await cache.track(frames[0], replaced, points, tracking),
        after = cache.statistics();
      for (const key of ['records', 'forward', 'back'])
        assert.deepEqual(actual[key], expected[key]);
      assert.equal(after.pyramidAllocations, before.pyramidAllocations);
      assert.equal(after.bufferReuses, before.bufferReuses + 2);
      assert.equal(after.pyramidBuilds, before.pyramidBuilds + 1);
      assert.equal(after.evictions, before.evictions);
    } finally {
      await cache.dispose();
    }
    assert.equal(activeBytes(device), 0);
  });
});

test('idle KLT storage yields to another allocation and rebuilds unchanged with either pool policy', async () => {
  for (const poolBytes of [0, 32 * 1024 ** 2])
    await withDevice(
      async (device) => {
        const sentinel = device.allocate(17, new Float32Array(17).fill(0.375)),
          cache = createKltPairCache(device, { capacity: 2 });
        let blocker;
        try {
          const first = await cache.track(frames[0], frames[1], points, tracking);
          assert.equal(cache.statistics().frames, 2);
          blocker = device.allocate(3 * layout.total);
          assert.equal(cache.statistics().frames, 1, 'idle cache releases one pyramid');
          assert.ok(activeBytes(device) <= device.workingBytes);
          device.recycle(blocker);
          blocker = undefined;
          const rebuilt = await cache.track(frames[0], frames[1], points, tracking);
          for (const key of ['records', 'forward', 'back'])
            assert.deepEqual(rebuilt[key], first[key]);
          assert.equal(cache.statistics().pyramidBuilds, 3);
          assert.deepEqual(Array.from(await device.read(sentinel, 17)), Array(17).fill(0.375));
        } finally {
          if (blocker) device.recycle(blocker);
          await cache.dispose();
          device.recycle(sentinel);
        }
        assert.equal(activeBytes(device), 0);
      },
      { poolBytes, workingBytes: 2 * pyramidBytes + 4096 },
    );
});

test('pending KLT output reads protect their active image pair against allocation pressure', async () => {
  await withDevice(
    async (device) => {
      const cache = createKltPairCache(device, { capacity: 2 }),
        read = device.read.bind(device);
      let release,
        reached,
        first = true;
      const held = new Promise((resolve) => {
          release = resolve;
        }),
        started = new Promise((resolve) => {
          reached = resolve;
        });
      device.read = async (buffer, length) => {
        if (first) {
          first = false;
          reached();
          await held;
        }
        return read(buffer, length);
      };
      const tracked = cache.track(frames[0], frames[1], points, tracking);
      let blocker;
      try {
        await started;
        assert.throws(() => device.allocate(3 * layout.total), /working budget/);
        assert.equal(cache.statistics().frames, 2, 'in-flight images cannot be reclaimed');
        release();
        await tracked;
        blocker = device.allocate(3 * layout.total);
        assert.equal(
          cache.statistics().frames,
          1,
          'same allocation succeeds after pair becomes idle',
        );
      } finally {
        release();
        await tracked;
        device.read = read;
        if (blocker) device.recycle(blocker);
        await cache.dispose();
      }
      assert.equal(activeBytes(device), 0);
    },
    { workingBytes: 2 * pyramidBytes + 4096 },
  );
});

test('a failed accepted in-place write drops dirty buffers before queued cache disposal', async () => {
  await withDevice(async (device) => {
    const cache = createKltPairCache(device, { capacity: 2 }),
      sentinel = device.allocate(17, new Float32Array(17).fill(0.25)),
      write = device.writeF32.bind(device),
      drop = device.drop.bind(device),
      failure = new Error('injected write failure'),
      dropped = [];
    try {
      await cache.track(frames[0], frames[1], points, tracking);
      device.writeF32 = (buffer, values) => {
        write(buffer, values);
        throw failure;
      };
      device.drop = (buffer) => {
        dropped.push(buffer);
        drop(buffer);
      };
      const failed = cache.track(frames[2], frames[1], points, tracking),
        disposed = cache.dispose();
      await assert.rejects(failed, (error) => error === failure);
      await disposed;
      assert.equal(dropped.length, 2, 'dirty gray and gradient are destroyed rather than pooled');
      assert.equal(cache.statistics().frames, 0);
      assert.equal(activeBytes(device), 68);
      assert.deepEqual(Array.from(await device.read(sentinel, 17)), Array(17).fill(0.25));
    } finally {
      device.writeF32 = write;
      device.drop = drop;
      await cache.dispose();
      device.recycle(sentinel);
    }
    assert.equal(activeBytes(device), 0);
  });
});

test('failed pyramid preparation invalidates reused storage before successful reconstruction', async () => {
  await withDevice(async (device) => {
    const cache = createKltPairCache(device, { capacity: 2 }),
      plan = device.planSteps.bind(device),
      failure = new Error('injected pyramid failure');
    try {
      const first = await cache.track(frames[0], frames[1], points, tracking);
      device.planSteps = async (steps) => {
        await plan(steps);
        throw failure;
      };
      await assert.rejects(
        cache.track(frames[2], frames[1], points, tracking),
        (error) => error === failure,
      );
      assert.equal(cache.statistics().frames, 1);
      device.planSteps = plan;
      const rebuilt = await cache.track(frames[0], frames[1], points, tracking);
      for (const key of ['records', 'forward', 'back']) assert.deepEqual(rebuilt[key], first[key]);
      assert.equal(cache.statistics().pyramidAllocations, 3);
    } finally {
      device.planSteps = plan;
      await cache.dispose();
    }
    assert.equal(activeBytes(device), 0);
  });
});

test('owned f32 write accounting validates type/length and disposed caches unregister reclamation', async () => {
  await withDevice(async (device) => {
    const values = Float32Array.of(1, 2, 3, 4),
      f32 = device.allocate(4, values),
      u32 = device.allocateU32(4, Uint32Array.of(1, 2, 3, 4)),
      before = device.statistics.applicationUploadBytes;
    device.writeF32(f32, Float32Array.of(4, 3, 2, 1));
    assert.equal(device.statistics.applicationUploadBytes - before, 16);
    assert.deepEqual(Array.from(await device.read(f32, 4)), [4, 3, 2, 1]);
    const after = device.statistics.applicationUploadBytes;
    for (const wrong of [new Float32Array(3), new Float64Array(4), [1, 2, 3, 4]])
      assert.throws(() => device.writeF32(f32, wrong), /active owned buffer/);
    assert.throws(() => device.writeF32(u32, new Float32Array(4)), /active owned buffer/);
    assert.equal(device.statistics.applicationUploadBytes, after);
    device.recycle(f32);
    assert.throws(() => device.writeF32(f32, values), /active owned buffer/);
    device.recycle(u32);
    const register = device.registerWorkingReclaimer.bind(device);
    let invoked = 0;
    device.registerWorkingReclaimer = (reclaim) =>
      register(() => {
        invoked++;
        return reclaim();
      });
    const cache = createKltPairCache(device);
    await cache.dispose();
    assert.throws(() => device.reserveWorkingBytes(device.workingBytes + 4), /working budget/);
    assert.equal(invoked, 0, 'closed cache callback no longer participates in admission');
    device.registerWorkingReclaimer = register;
    assert.equal(activeBytes(device), 0);
  });
});
