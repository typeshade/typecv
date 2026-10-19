import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice } from '../../dist/gpu.js';

// The pool is the TypeShade device's (`createDevice({ poolBytes })`, change 0071): a released
// array comes back for the next array of the same element and length, and the oldest released
// array leaves first when the pool is full. Its rules are the compiler's tests to hold. These
// tests hold what TypeCV keeps over it: the working budget, the counters, and that `recycle` and
// `drop` take a handle back once.

test('a released array is reused for the same kind and length, and the counters say so', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 8192, workingBytes: 16384 });
  try {
    const first = device.allocate(768),
      second = device.allocate(512);
    device.recycle(first);
    device.recycle(second);
    // 768 and 512 f32 values: 3072 and 2048 bytes, both pooled.
    assert.equal(device.statistics.pooledBytes, 5120);
    assert.equal(device.statistics.liveBytes, 5120);
    assert.equal(device.statistics.liveBuffers, 2);
    device.allocate(768);
    assert.equal(device.statistics.reuses, 1);
    assert.equal(device.statistics.pooledBytes, 2048);
    device.allocateU32(768);
    device.allocate(767);
    assert.equal(device.statistics.reuses, 1);
    assert.equal(device.statistics.allocations, 4);
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBytes, 0);
  assert.equal(device.statistics.pooledBytes, 0);
});

test('working reservation admits by the active bytes and leaves the pool to poolBytes', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 8192, workingBytes: 8192 });
  try {
    const active = device.allocateU32(512, new Uint32Array(512).fill(17));
    const free = device.allocate(1024);
    device.recycle(free);
    // 2048 active bytes and 4096 pooled: the pooled ones are not the working budget's.
    const before = device.statistics;
    assert.throws(() => device.reserveWorkingBytes(8192), /working budget/);
    assert.deepEqual(device.statistics, before);
    device.reserveWorkingBytes(6144);
    assert.equal(device.statistics.liveBytes, 6144);
    assert.equal(device.statistics.pooledBytes, 4096);
    assert.deepEqual(await active.read(), new Uint32Array(512).fill(17));
    const next = device.allocateF64(512);
    assert.equal(device.statistics.liveBytes, 10240);
    assert.throws(() => device.allocate(1024), /working budget/);
    assert.equal(device.statistics.liveBytes, 10240);
    // A reused array becomes active again, and the admission counts it.
    assert.throws(() => device.allocate(1024), /working budget/);
    device.drop(next);
    device.allocate(1024);
    assert.equal(device.statistics.reuses, 1);
    assert.equal(device.statistics.pooledBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('reuse preserves explicit uploads and fresh zero values', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 8192, workingBytes: 8192 });
  try {
    const fresh = device.allocateU32(4);
    assert.deepEqual(await fresh.read(), new Uint32Array(4));
    device.recycle(fresh);
    const values = new Uint32Array([9, 8, 7, 6]);
    const uploaded = device.allocateU32(4, values);
    assert.equal(device.statistics.reuses, 1);
    assert.deepEqual(await uploaded.read(), values);
    device.recycle(uploaded);
    const zeroed = device.allocateU32(4);
    assert.equal(device.statistics.reuses, 2);
    assert.deepEqual(await zeroed.read(), new Uint32Array(4));
    assert.equal(device.statistics.applicationUploadBytes, 16);
  } finally {
    await device.dispose();
  }
});

test('recycle and drop take a handle back once, and a dropped array is not pooled', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 8192, workingBytes: 8192 });
  try {
    const kept = device.allocate(256),
      failed = device.allocate(256);
    device.recycle(kept);
    device.recycle(kept);
    assert.equal(device.statistics.pooledBytes, 1024);
    device.drop(failed);
    device.drop(failed);
    device.recycle(failed);
    assert.equal(device.statistics.liveBuffers, 1);
    assert.equal(device.statistics.pooledBytes, 1024);
    await assert.rejects(() => failed.read());
  } finally {
    await device.dispose();
  }
});

test('invalid options and working reservations fail explicitly', async () => {
  for (const poolBytes of [-1, 1.5, NaN, Infinity])
    await assert.rejects(ShadeDevice.open({ backend: 'cpu', poolBytes }), /options/);
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

test('two devices in one context are independent', async () => {
  const one = await ShadeDevice.open({ backend: 'cpu', poolBytes: 4096 });
  const two = await ShadeDevice.open({ backend: 'cpu', poolBytes: 4096 });
  try {
    const a = one.allocate(16, new Float32Array(16).fill(1));
    const b = two.allocate(16, new Float32Array(16).fill(2));
    assert.equal(one.statistics.liveBuffers, 1);
    assert.equal(two.statistics.liveBuffers, 1);
    await one.dispose();
    assert.deepEqual(await b.read(), new Float32Array(16).fill(2));
    await assert.rejects(() => a.read());
  } finally {
    await two.dispose();
  }
});

test('the flavor of the emulated double is an option of open(), float on the CPU tier by default', async () => {
  // TypeShade change 0073: the owner's flavor picks the WGSL a kernel call dispatches. The CPU
  // tier computes doubles as doubles, so `'auto'` is `'float'` there; a word is taken as given.
  const plain = await ShadeDevice.open({ backend: 'cpu' });
  try {
    assert.equal(plain.fp64Flavor, 'float');
  } finally {
    await plain.dispose();
  }
  const integer = await ShadeDevice.open({ backend: 'cpu', fp64Flavor: 'integer' });
  try {
    assert.equal(integer.fp64Flavor, 'integer');
  } finally {
    await integer.dispose();
  }
  await assert.rejects(ShadeDevice.open({ backend: 'cpu', fp64Flavor: 'fast' }), RangeError);
});
