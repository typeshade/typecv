import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice } from '../../dist/gpu.js';

const MiB = 1024 ** 2;

test('default pool retains legacy first-fit reuse and exact kind/length identity', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 4096, workingBytes: 16384 });
  try {
    const first = device.allocate(768),
      second = device.allocate(512);
    device.recycle(first);
    device.recycle(second);
    assert.equal(device.statistics.pooledBytes, 3072);
    assert.equal(device.statistics.liveBytes, 3072);
    assert.equal(device.allocate(768), first);
    const unsigned = device.allocateU32(768);
    assert.notEqual(unsigned, first);
    device.recycle(first);
    assert.notEqual(device.allocate(767), first);
    assert.equal(device.statistics.reuses, 1);
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBytes, 0);
  assert.equal(device.statistics.pooledBytes, 0);
});

test('protected small capacity evicts oldest large free buffer and retains active handles', async () => {
  const device = await ShadeDevice.open({
    backend: 'cpu',
    poolBytes: 4 * MiB,
    poolSmallBufferBytes: MiB,
    workingBytes: 16 * MiB,
  });
  try {
    const oldLarge = device.allocate((2 * MiB) / 4),
      newLarge = device.allocate((2 * MiB) / 4),
      activeLarge = device.allocate((2 * MiB) / 4),
      small = device.allocateU32(MiB / 8);
    device.recycle(oldLarge);
    device.recycle(small);
    device.recycle(newLarge);
    assert.equal(device.statistics.pooledBytes, 2.5 * MiB);
    assert.equal(device.statistics.liveBytes, 4.5 * MiB);
    assert.equal(device.allocateU32(MiB / 8), small);
    assert.equal(device.allocate((2 * MiB) / 4), newLarge);
    device.recycle(activeLarge);
    assert.equal(device.allocate((2 * MiB) / 4), activeLarge);
    assert.equal(device.statistics.reuses, 3);
  } finally {
    await device.dispose();
  }
});

test('protected pool rejects a single oversized cached entry without evicting smaller entries', async () => {
  const device = await ShadeDevice.open({
    backend: 'cpu',
    poolBytes: 4 * MiB,
    poolSmallBufferBytes: MiB,
    workingBytes: 16 * MiB,
  });
  try {
    const small = device.allocate(128),
      oversized = device.allocate(MiB);
    device.recycle(small);
    device.recycle(oversized);
    assert.equal(device.statistics.liveBytes, 512);
    assert.equal(device.statistics.pooledBytes, 512);
    assert.equal(device.allocate(128), small);
  } finally {
    await device.dispose();
  }
});

test('working reservation trims free entries but rejects active overcommit before mutation', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 8192, workingBytes: 8192 });
  try {
    const active = device.allocateU32(512, new Uint32Array(512).fill(17));
    const free = device.allocate(1024);
    device.recycle(free);
    const before = device.statistics;
    assert.throws(() => device.reserveWorkingBytes(8192), /working budget/);
    assert.deepEqual(device.statistics, before);
    device.reserveWorkingBytes(4096);
    assert.equal(device.statistics.liveBytes, 2048);
    assert.equal(device.statistics.pooledBytes, 0);
    assert.deepEqual(await active.read(), new Uint32Array(512).fill(17));
    const next = device.allocateF64(512);
    assert.equal(device.statistics.liveBytes, 6144);
    assert.throws(() => device.allocate(1024), /working budget/);
    assert.equal(device.statistics.liveBytes, 6144);
    device.drop(next);
  } finally {
    await device.dispose();
  }
});

test('new allocations trim free storage and exact reuse does not double-count live bytes', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 8192, workingBytes: 8192 });
  try {
    const active = device.allocateU32(512),
      free = device.allocate(1024);
    device.recycle(free);
    assert.equal(device.allocate(1024), free);
    assert.equal(device.statistics.liveBytes, 6144);
    device.recycle(free);
    const incompatible = device.allocateF64(512);
    assert.equal(device.statistics.liveBytes, 6144);
    assert.equal(device.statistics.pooledBytes, 0);
    device.drop(active);
    device.drop(incompatible);
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('reuse preserves explicit uploads, fresh zero values and failure cleanup', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 8192, workingBytes: 8192 });
  try {
    const fresh = device.allocateU32(4);
    assert.deepEqual(await fresh.read(), new Uint32Array(4));
    device.recycle(fresh);
    const values = new Uint32Array([9, 8, 7, 6]);
    assert.equal(device.allocateU32(4, values), fresh);
    assert.deepEqual(await fresh.read(), values);
    device.recycle(fresh);
    const failure = new Error('injected upload failure');
    fresh.write = () => {
      throw failure;
    };
    assert.throws(
      () => device.allocateU32(4, values),
      (error) => error === failure,
    );
    assert.equal(device.statistics.liveBytes, 0);
    assert.equal(device.statistics.pooledBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('invalid pool policies and working reservations fail explicitly', async () => {
  for (const poolSmallBufferBytes of [-1, 1.5, NaN, Infinity, 4097])
    await assert.rejects(
      ShadeDevice.open({ backend: 'cpu', poolBytes: 4096, poolSmallBufferBytes }),
      /options/,
    );
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const value of [-1, 1.5, NaN, Infinity])
      assert.throws(() => device.reserveWorkingBytes(value), /reservation/);
    assert.equal(device.statistics.allocations, 0);
  } finally {
    await device.dispose();
  }
  assert.throws(() => device.reserveWorkingBytes(0), /disposed/);
});
