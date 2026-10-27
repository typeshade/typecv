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

test('the TypeShade device chooses the flavor, with no adapter request of its own', async () => {
  // TypeShade change 0085: `'auto'` is resolved from the device the owner opens. TypeCV requests no
  // adapter to choose it, so a WebGPU open requests one adapter, the one the device would be made
  // from. With no adapter, WebGPU alone cannot open, and the open fails after that one request.
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  let requests = 0;
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      userAgent: 'test',
      gpu: {
        async requestAdapter() {
          requests++;
          return null;
        },
      },
    },
  });
  try {
    await assert.rejects(ShadeDevice.open({ backend: 'webgpu' }));
    await assert.rejects(ShadeDevice.open({ backend: 'webgpu', fp64Flavor: 'auto' }));
  } finally {
    if (descriptor) Object.defineProperty(globalThis, 'navigator', descriptor);
    else delete globalThis.navigator;
  }
  assert.equal(requests, 2);
});

test('the per-kernel table sums groups, dispatches and GPU time by fused call list, up to 160 keys', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  const fuse = device.shade.fuse;
  // Feed deterministic runtime reports through plan(), where work is collected once.
  // record() also stores composite reports and intentionally does not collect them.
  const plan = async (groups) => {
    device.shade.fuse = async () => ({ tier: 'cpu', dispatches: 0, submissions: 0, groups });
    await device.plan(() => {});
  };
  try {
    assert.deepEqual(device.statistics.kernels, []);
    await plan([
      { calls: ['blur', 'scan'], fused: true, workgroupShapes: [[64, 1, 1]], gpuTime: 1_500_000 },
      {
        calls: ['solve'],
        fused: false,
        workgroupShapes: [
          [8, 1, 1],
          [64, 1, 1],
        ],
      },
    ]);
    await plan([
      { calls: ['blur', 'scan'], fused: true, workgroupShapes: [[64, 1, 1]], gpuTime: 500_000 },
      { calls: ['solve'], fused: false, gpuTime: Number.NaN },
    ]);
    // A group that carries no time counts, and it does not add a 0: `timedGroups` says so.
    assert.deepEqual(device.statistics.kernels, [
      { name: 'blur+scan', groups: 2, dispatches: 2, timedGroups: 2, gpuNs: 2_000_000 },
      { name: 'solve', groups: 2, dispatches: 2, timedGroups: 0, gpuNs: 0 },
    ]);
    // The table is bounded: a call list beyond 160 keys is counted and dropped, and a key the
    // table holds keeps adding.
    device.resetKernelStatistics();
    for (let i = 0; i < 165; i++) await plan([{ calls: ['k' + i], fused: false }]);
    await plan([{ calls: ['k0'], fused: false, gpuTime: 7 }]);
    const { kernels, kernelsDropped } = device.statistics;
    assert.equal(kernels.length, 160);
    assert.equal(kernelsDropped, 5);
    assert.deepEqual(kernels[0], {
      name: 'k0',
      groups: 2,
      dispatches: 0,
      timedGroups: 1,
      gpuNs: 7,
    });
    device.resetKernelStatistics();
    assert.deepEqual(device.statistics.kernels, []);
    assert.equal(device.statistics.kernelsDropped, 0);
  } finally {
    device.shade.fuse = fuse;
    await device.dispose();
  }
});
