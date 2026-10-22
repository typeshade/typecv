import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('native errors and unexpected external destruction are captured before cleanup', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice } = await import('/dist/gpu.js');
    const adapter = await navigator.gpu.requestAdapter();
    const gpu = await adapter.requestDevice();
    const device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 4096 });
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
    return { lost, after: device.diagnostics };
  });
  assert.deepEqual(errors, []);
  assert.equal(result.lost.reason, 'destroyed');
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
