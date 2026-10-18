// CPU-only host admission/packing checks. Physical numerical parity is separate.
import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, bundleAdjustGpu } from '../../dist/gpu.js';

function scene(cameras, points = 4) {
  const views = Array.from({ length: cameras }, (_, id) => ({
    id,
    camera: { fx: 800, fy: 800, cx: 320, cy: 240 },
    pose: { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(-id * 0.01, 0, 0) },
  }));
  const landmarks = Array.from({ length: points }, (_, trackId) => {
    const xyz = Float64Array.of(
      ((trackId % 5) - 2) * 0.15,
      ((Math.floor(trackId / 5) % 5) - 2) * 0.11,
      4 + (trackId % 7) * 0.2,
    );
    // One complete long track connects every camera, including indices above64.
    const observed = trackId === 0 ? views : [views[0], views[1 + (trackId % (cameras - 1))]];
    return {
      trackId,
      xyz,
      observations: observed.map((view) => ({
        viewId: view.id,
        featureId: trackId,
        pixel: [(800 * (xyz[0] + view.pose.t[0])) / xyz[2] + 320, (800 * xyz[1]) / xyz[2] + 240],
      })),
    };
  });
  return { views, landmarks };
}
function storage(model, shared) {
  const C = model.views.length,
    P = model.landmarks.length,
    M = model.landmarks.reduce((n, p) => n + p.observations.length, 0),
    trial = 12 * C + 3 * P,
    index = 3 * M + P + 2 * C + 2,
    sizes = [
      18 * C + 3 * P + 2 * M,
      trial,
      (shared ? 45 : 42) * M + 13 * P + 78 * C,
      30 * C + 3 * P + (shared ? 5 : 0),
      26 + 2 * M,
      trial + (shared ? 27 : 26),
      ...(shared ? [6 * C + 6 + 2 * M] : []),
    ];
  // CPU backend has no WebGPU camera inversion scratch. Cache is disabled here.
  return {
    bytes: sizes.reduce((n, size) => n + size * 8, index * 4 + 8),
    binding: Math.max(index * 4, ...sizes.map((n) => n * 8)),
    elements: Math.max(M, P, C * 6 + Number(shared), trial, trial + (shared ? 27 : 26)),
    observations: M,
  };
}
const options = (shared) => ({
  iterations: 0,
  anchorViewId: 0,
  scaleViewId: 1,
  refineSharedFocal: shared,
});
const rejects = (device, model, opts, pattern) =>
  assert.rejects(async () => bundleAdjustGpu(device, model, opts), pattern);

test('CPU BA admits resource-fitting64/65/95/96 cameras without a replacement ceiling or observation truncation', async () => {
  for (const shared of [false, true])
    for (const cameras of [64, 65, 95, 96]) {
      const model = scene(cameras),
        device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
      try {
        const result = await bundleAdjustGpu(device, model, options(shared));
        assert.deepEqual(result.reconstruction, model);
        assert.equal(result.reconstruction.landmarks[0].observations.length, cameras);
        assert.equal(result.diagnostics.observations, storage(model, shared).observations);
        assert.equal(result.execution.backend, 'cpu');
        assert.equal(result.execution.report.dispatches, 0);
        assert.equal(device.statistics.liveBytes, 0);
      } finally {
        await device.dispose();
      }
    }
});

test('CPU BA exact owned working budget admits equality and rejects the next larger real scene', async () => {
  for (const shared of [false, true]) {
    const model = scene(95),
      next = scene(96),
      budget = storage(model, shared).bytes,
      device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, workingBytes: budget });
    try {
      const result = await bundleAdjustGpu(device, model, options(shared));
      assert.deepEqual(result.reconstruction, model);
      assert.ok(storage(next, shared).bytes > budget);
      const before = device.statistics;
      await rejects(device, next, options(shared), /working\/storage\/dispatch budget/);
      assert.deepEqual(device.statistics, before);
    } finally {
      await device.dispose();
    }
  }
});

test('CPU BA binding and whole-workgroup dispatch boundaries reject before allocation', async () => {
  for (const shared of [false, true]) {
    // These point counts make the largest dispatch domain an exact64 multiple.
    const model = scene(65, shared ? 51 : 30),
      required = storage(model, shared),
      device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
    const originalBinding = device.bindingBytes,
      originalElements = device.maxKernelElements;
    try {
      assert.equal(required.elements % 64, 0);
      device.bindingBytes = required.binding;
      device.maxKernelElements = required.elements;
      await bundleAdjustGpu(device, model, options(shared));
      const before = device.statistics;
      device.bindingBytes = required.binding - 1;
      await rejects(device, model, options(shared), /working\/storage\/dispatch budget/);
      assert.deepEqual(device.statistics, before);
      device.bindingBytes = required.binding;
      device.maxKernelElements = required.elements - 64;
      await rejects(device, model, options(shared), /working\/storage\/dispatch budget/);
      assert.deepEqual(device.statistics, before);
    } finally {
      device.bindingBytes = originalBinding;
      device.maxKernelElements = originalElements;
      await device.dispose();
    }
  }
});

test('CPU BA retains gauge, variable-window, ownership, connectivity and options rejection', async () => {
  for (const shared of [false, true]) {
    const model = scene(65),
      device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
    try {
      const before = device.statistics;
      await rejects(device, model, { ...options(shared), anchorViewId: 999 }, /gauge/);
      await rejects(device, model, { ...options(shared), scaleViewId: 0 }, /gauge/);
      await rejects(
        device,
        model,
        { ...options(shared), variableViewIds: [0, 2] },
        /variable camera window/,
      );
      await rejects(device, model, { ...options(shared), iterations: -1 }, /scene\/options/);
      const disconnected = { ...model, views: [...model.views, { ...model.views[0], id: 999 }] };
      await rejects(device, disconnected, options(shared), /connected observation graph/);
      const duplicate = {
        ...model,
        landmarks: [...model.landmarks, { ...model.landmarks[0], trackId: 999 }],
      };
      await rejects(device, duplicate, options(shared), /multiply owned/);
      assert.deepEqual(device.statistics, before);
    } finally {
      await device.dispose();
    }
  }
});

test('CPU host-validation mocks reject unsafe counts and derived index lengths before packing or enqueue', async () => {
  // Tiny iterables deliberately report inconsistent large counts. These isolate
  // integer admission guards; they are not geometry fixtures or GPU evidence.
  const original = scene(2),
    sentinel = new Error('packing/enqueue must not be reached');
  const fakeDevice = {
    backend: 'cpu',
    boundedSubmissions: false,
    bindingBytes: Number.MAX_SAFE_INTEGER,
    workingBytes: Number.MAX_SAFE_INTEGER,
    maxKernelElements: Number.MAX_SAFE_INTEGER,
    enqueue() {
      throw sentinel;
    },
  };
  for (const shared of [false, true]) {
    for (const count of [NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, 0x100000000]) {
      const views = {
        length: count,
        [Symbol.iterator]: () => original.views[Symbol.iterator](),
        map() {
          throw sentinel;
        },
      };
      await rejects(fakeDevice, { ...original, views }, options(shared), /integer index bounds/);
      const pointCount = {
        length: count,
        [Symbol.iterator]: () => original.landmarks[Symbol.iterator](),
        reduce: () => 8,
        map() {
          throw sentinel;
        },
      };
      await rejects(
        fakeDevice,
        { ...original, landmarks: pointCount },
        options(shared),
        /integer index bounds/,
      );
      const landmarks = {
        length: 1,
        [Symbol.iterator]: () => original.landmarks[Symbol.iterator](),
        reduce: () => count,
        map() {
          throw sentinel;
        },
      };
      await rejects(
        fakeDevice,
        { ...original, landmarks },
        options(shared),
        /integer index bounds/,
      );
    }
    const landmarks = {
      length: 0xffffffff,
      [Symbol.iterator]: () => original.landmarks[Symbol.iterator](),
      reduce: () => 2,
      map() {
        throw sentinel;
      },
    };
    await rejects(
      fakeDevice,
      { ...original, landmarks },
      options(shared),
      /working\/storage\/dispatch budget/,
    );
  }
});
