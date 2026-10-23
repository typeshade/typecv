import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('device information uses the supplied device without probing another adapter', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice } = await import('/dist/gpu.js');
    const adapter = await navigator.gpu.requestAdapter();
    const gpu = await adapter.requestDevice();
    const requestAdapter = navigator.gpu.requestAdapter;
    let probes = 0;
    navigator.gpu.requestAdapter = async () => {
      probes++;
      throw new Error('A supplied device must not probe another adapter');
    };
    const rows = [];
    try {
      const withoutIdentity = new Proxy(gpu, {
        get(target, key) {
          if (key === 'adapterInfo') return undefined;
          const value = Reflect.get(target, key, target);
          return typeof value === 'function' ? value.bind(target) : value;
        },
      });
      for (const native of [gpu, withoutIdentity]) {
        const device = await ShadeDevice.open({ gpuDevice: native });
        try {
          const info = device.info;
          const snapshot = JSON.stringify(info);
          const enabledFeatures = [...gpu.features].sort();
          const nativeInfo = native.adapterInfo;
          const software =
            info.webgpu.fallback === true ||
            /swiftshader|software|llvmpipe/i.test(
              [
                nativeInfo?.vendor,
                nativeInfo?.architecture,
                nativeInfo?.device,
                nativeInfo?.description,
              ].join(' '),
            );
          await device.dispose();
          rows.push({
            info,
            frozen: Object.isFrozen(info) && Object.isFrozen(info.webgpu),
            sameAfterDisposal: info === device.info && snapshot === JSON.stringify(device.info),
            enabledFeatures,
            bindingLimit: gpu.limits.maxStorageBufferBindingSize,
            occupancyFloors: device.occupancyFloors,
            expectedOccupancy: !software,
            expectedVendor: nativeInfo?.vendor || null,
          });
        } finally {
          await device.dispose();
        }
      }
      return { rows, probes };
    } finally {
      navigator.gpu.requestAdapter = requestAdapter;
      gpu.destroy();
    }
  });
  assert.deepEqual(errors, []);
  assert.equal(result.probes, 0);
  assert.equal(result.rows.length, 2);
  for (const row of result.rows) {
    assert.equal(row.info.tier, 'webgpu');
    assert.equal(typeof row.info.build.version, 'string');
    assert.equal(row.info.webgpu.vendor, row.expectedVendor);
    assert.deepEqual(row.info.webgpu.features, row.enabledFeatures);
    assert.equal(row.info.webgpu.limits.maxStorageBufferBindingSize, row.bindingLimit);
    assert.equal(row.occupancyFloors, row.expectedOccupancy);
    assert.equal(row.frozen, true);
    assert.equal(row.sameAfterDisposal, true);
  }
  const unknown = result.rows[1].info.webgpu;
  assert.equal(unknown.provenance, 'unavailable');
  assert.equal(unknown.vendor, null);
  assert.equal(unknown.fallback, null);
});

test('native errors and unexpected external destruction are captured before cleanup', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice } = await import('/dist/gpu.js');
    const adapter = await navigator.gpu.requestAdapter();
    const gpu = await adapter.requestDevice();
    const device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 4096 });
    const info = device.info;
    const nativeError = new Promise((resolve) =>
      gpu.addEventListener('uncapturederror', resolve, { once: true }),
    );
    const invalid = gpu.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.STORAGE,
    });
    await gpu.queue.onSubmittedWorkDone();
    await nativeError;
    invalid.destroy();
    const first = device.allocate(1024);
    device.recycle(first);
    const second = device.allocateU32(512);
    device.drop(second);
    gpu.destroy();
    const lost = await device.lost;
    await device.dispose();
    return { lost, after: device.diagnostics, sameInfoAfterLoss: info === device.info };
  });
  assert.deepEqual(errors, []);
  assert.equal(result.lost.reason, 'destroyed');
  assert.equal(result.sameInfoAfterLoss, true);
  const d = result.lost.diagnostics;
  assert.equal(d.disposalRequested, false);
  assert.equal(d.destroyRequested, false, 'TypeCV did not destroy the supplied device');
  assert.equal(d.errorEventsAvailable, true);
  assert.ok(d.recentErrors.some((e) => e.name === 'GPUValidationError' && e.message.length));
  assert.equal(d.memory.peakLiveBytes, 6144);
  assert.equal(d.memory.peakActiveBytes, 4096);
  assert.equal(result.after.memory.liveBytes, 0);
  assert.equal(result.after.disposalRequested, true);
  assert.equal(result.lost.diagnostics.disposalRequested, false, 'loss snapshot survives cleanup');
});

test('owned disposal is observed, while disposal leaves an external native device usable', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice } = await import('/dist/gpu.js');
    const owned = await ShadeDevice.open({ adapterLimits: true });
    await owned.dispose();
    const ownedLoss = await owned.lost;
    const adapter = await navigator.gpu.requestAdapter();
    const gpu = await adapter.requestDevice();
    const external = await ShadeDevice.open({ gpuDevice: gpu });
    await external.dispose();
    const buffer = gpu.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    gpu.queue.writeBuffer(buffer, 0, new Uint32Array([7]));
    await gpu.queue.onSubmittedWorkDone();
    buffer.destroy();
    gpu.destroy();
    return { ownedLoss, externalLoss: await external.lost };
  });
  assert.deepEqual(errors, []);
  assert.equal(result.ownedLoss.diagnostics.disposalRequested, true);
  assert.equal(result.ownedLoss.diagnostics.destroyRequested, true);
  assert.equal(result.externalLoss.diagnostics.disposalRequested, true);
  assert.equal(result.externalLoss.diagnostics.destroyRequested, false);
});
