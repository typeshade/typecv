import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, verifyImagePairGpu } from '../../dist/gpu.js';
import { radialPairFixture } from './initialization-cases.mjs';

const options = {
  iterations: 32,
  minIterations: 32,
  homographyIterations: 32,
  refinementIterations: 0,
  threshold: 0.001,
  seed: 71,
};

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

async function allReadsStarted(signal) {
  let timer;
  try {
    await Promise.race([
      signal,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('Terminal reads did not overlap')), 10000);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function interceptSnapshots(device) {
  const originalRead = device.read.bind(device),
    originalRecycle = device.recycle.bind(device),
    originalDrop = device.drop.bind(device),
    started = deferred(),
    gates = Array.from({ length: 3 }, deferred),
    finished = Array.from({ length: 3 }, deferred),
    lengths = [];
  let recycled = 0,
    dropped = 0;
  device.read = async (buffer, length) => {
    // The chunked model generation reads one control word between its chunks: not a snapshot.
    if (length === 1) return originalRead(buffer, length);
    const index = lengths.length;
    lengths.push(length);
    assert.ok(index < 3, 'Verification must only take its existing three terminal snapshots');
    if (lengths.length === 3) started.resolve();
    try {
      await gates[index].promise;
      return await originalRead(buffer, length);
    } finally {
      finished[index].resolve();
    }
  };
  device.recycle = (buffer) => {
    recycled++;
    return originalRecycle(buffer);
  };
  device.drop = (buffer) => {
    dropped++;
    return originalDrop(buffer);
  };
  return {
    started: started.promise,
    gates,
    finished,
    lengths,
    get recycled() {
      return recycled;
    },
    get dropped() {
      return dropped;
    },
    restore() {
      device.read = originalRead;
      device.recycle = originalRecycle;
      device.drop = originalDrop;
    },
    release() {
      for (const gate of gates) gate.resolve();
    },
  };
}

test('verification snapshots overlap without changing results, lengths or queued ownership', async () => {
  const device = await ShadeDevice.open({
      backend: 'cpu',
      boundedSubmissions: false,
      solverCheckpoints: false,
    }),
    input = radialPairFixture({ n: 17 });
  let snapshots, pending, next;
  try {
    const expected = await verifyImagePairGpu(
      device,
      input.pixels,
      input.first,
      input.second,
      options,
    );
    snapshots = interceptSnapshots(device);
    let settled = false,
      nextStarted = false;
    pending = verifyImagePairGpu(device, input.pixels, input.first, input.second, options).then(
      (value) => {
        settled = true;
        return value;
      },
    );
    next = device.enqueue(async () => {
      nextStarted = true;
    });
    await allReadsStarted(snapshots.started);
    assert.deepEqual(snapshots.lengths, [72, 17, 12]);
    snapshots.gates[2].resolve();
    snapshots.gates[0].resolve();
    await Promise.all([snapshots.finished[2].promise, snapshots.finished[0].promise]);
    assert.equal(settled, false);
    assert.equal(nextStarted, false);
    assert.equal(snapshots.recycled, 0);
    assert.equal(snapshots.dropped, 0);
    snapshots.gates[1].resolve();
    const actual = await pending;
    await next;
    assert.equal(nextStarted, true);
    assert.ok(snapshots.recycled > 0);
    assert.equal(snapshots.dropped, 0);
    const { execution: actualExecution, ...actualResult } = actual,
      { execution: expectedExecution, ...expectedResult } = expected;
    assert.deepEqual(actualResult, expectedResult);
    // The second call reuses scratch allocations, so its upload accounting can differ, and the
    // chunk rates of its report are the device's learned times.
    const {
        applicationUploadBytes: actualUploads,
        report: { chunks: actualChunks, ...actualReport },
        ...actualWork
      } = actualExecution,
      {
        applicationUploadBytes: expectedUploads,
        report: { chunks: expectedChunks, ...expectedReport },
        ...expectedWork
      } = expectedExecution;
    assert.deepEqual(
      { ...actualWork, report: actualReport },
      { ...expectedWork, report: expectedReport },
    );
    assert.deepEqual(Object.keys(actualChunks), Object.keys(expectedChunks));
    assert.ok(actualUploads <= expectedUploads);
  } finally {
    snapshots?.release();
    await Promise.allSettled([pending, next]);
    snapshots?.restore();
    await device.dispose();
    assert.equal(device.statistics.liveBuffers, 0);
  }
});

for (const failure of [
  { index: 0, reason: undefined, label: 'undefined first read rejection' },
  { index: 1, reason: Symbol('mask read failure'), label: 'second read rejection' },
]) {
  test(`verification preserves ${failure.label} and awaits sibling reads before cleanup`, async () => {
    const device = await ShadeDevice.open({
        backend: 'cpu',
        boundedSubmissions: false,
        solverCheckpoints: false,
      }),
      input = radialPairFixture({ n: 17 }),
      snapshots = interceptSnapshots(device);
    let pending, next;
    try {
      let settled = false,
        nextStarted = false;
      pending = verifyImagePairGpu(device, input.pixels, input.first, input.second, options).then(
        (value) => {
          settled = true;
          return { rejected: false, value };
        },
        (reason) => {
          settled = true;
          return { rejected: true, reason };
        },
      );
      next = device.enqueue(async () => {
        nextStarted = true;
      });
      await allReadsStarted(snapshots.started);
      assert.deepEqual(snapshots.lengths, [72, 17, 12]);
      snapshots.gates[2].reject(new Error('Later read failed first'));
      await snapshots.finished[2].promise;
      snapshots.gates[failure.index].reject(failure.reason);
      await snapshots.finished[failure.index].promise;
      assert.equal(settled, false);
      assert.equal(nextStarted, false);
      assert.equal(snapshots.recycled, 0);
      assert.equal(snapshots.dropped, 0);
      snapshots.gates[1 - failure.index].resolve();
      const outcome = await pending;
      await next;
      assert.equal(outcome.rejected, true);
      assert.equal(outcome.reason, failure.reason);
      assert.equal(snapshots.recycled, 0);
      assert.ok(snapshots.dropped > 0);
      assert.equal(device.statistics.liveBuffers, 0);
      assert.equal(nextStarted, true);
      snapshots.restore();
      const retried = await verifyImagePairGpu(
        device,
        input.pixels,
        input.first,
        input.second,
        options,
      );
      assert.equal(retried.invertibleCount, 17, 'A failed read must not poison the device queue');
    } finally {
      snapshots.release();
      await Promise.allSettled([pending, next]);
      snapshots.restore();
      await device.dispose();
      assert.equal(device.statistics.liveBuffers, 0);
    }
  });
}
