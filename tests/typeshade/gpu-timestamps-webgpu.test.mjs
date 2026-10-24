import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('native timestamp options preserve defaults, borrowed features, output bits and ownership', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, fuseTsdfGpu } = await import('/dist/internal/gpu.js');
      const { tsdfStagingInput, TSDF_STAGING_OPTIONS, pointDigests } =
        await import('/tests/typeshade/tsdf-staging-cases.mjs');
      const probe = await navigator.gpu.requestAdapter();
      if (!probe?.features.has('timestamp-query')) throw new Error('Timestamp support is required');
      const input = tsdfStagingInput();
      const devices = [],
        rows = [];
      const wrapDestroy = (raw) => {
        let count = 0;
        const destroy = raw.destroy.bind(raw);
        Object.defineProperty(raw, 'destroy', {
          configurable: true,
          value() {
            count++;
            destroy();
          },
        });
        return () => count;
      };
      try {
        for (const mode of ['default', 'enabled', 'disabled', 'disabled-adapter-limits']) {
          const device = await ShadeDevice.open({
            backend: 'webgpu',
            poolBytes: 0,
            ...(mode === 'default' ? {} : { gpuTimestamps: mode === 'enabled' }),
            ...(mode === 'disabled-adapter-limits' ? { adapterLimits: true } : {}),
          });
          devices.push(device);
          const owned = device.owned;
          const destroyed = owned ? wrapDestroy(owned) : undefined;
          const points = await fuseTsdfGpu(device, input, {
            ...TSDF_STAGING_OPTIONS,
            maxBlocks: 12,
            maxQueuedViewBytes: 4096,
          });
          const row = {
            mode,
            enabled: device.info.webgpu.features.includes('timestamp-query'),
            adapterLimits: device.adapterLimits,
            count: points.count,
            channels: await pointDigests(points),
            queue: points.tsdf.queuedViewUploads,
            rates: Object.keys(device.chunkRates),
            liveBytes: device.statistics.liveBytes,
          };
          await device.dispose();
          await device.dispose();
          row.ownedDestroyCount = destroyed ? destroyed() : null;
          rows.push(row);
        }
        const adapter = await navigator.gpu.requestAdapter();
        const raw = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
        const destroyed = wrapDestroy(raw);
        const borrowed = await ShadeDevice.open({
          backend: 'webgpu',
          gpuDevice: raw,
          gpuTimestamps: false,
          poolBytes: 0,
        });
        devices.push(borrowed);
        const actual = borrowed.info.webgpu.features.includes('timestamp-query');
        await borrowed.dispose();
        const buffer = raw.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST });
        buffer.destroy();
        const borrowedDestroyCount = destroyed();
        raw.destroy();
        const invalid = [];
        for (const gpuTimestamps of [null, 0, 1, 'false', {}, []]) {
          try {
            await ShadeDevice.open({ backend: 'cpu', gpuTimestamps });
            invalid.push(null);
          } catch (error) {
            invalid.push(error.name);
          }
        }
        return { rows, borrowed: { actual, destroyCount: borrowedDestroyCount }, invalid };
      } finally {
        for (const device of devices) await device.dispose();
      }
    },
    undefined,
    { timeoutMs: 60000 },
  );
  assert.deepEqual(errors, []);
  const expected = result.rows[0];
  for (const row of result.rows) {
    assert.equal(row.enabled, !row.mode.startsWith('disabled'));
    assert.equal(row.adapterLimits, row.mode === 'disabled-adapter-limits');
    assert.equal(row.count, expected.count);
    assert.deepEqual(row.channels, expected.channels);
    assert.equal(Object.keys(row.channels).length, 7);
    assert.equal(row.liveBytes, 0);
    assert.ok(
      row.rates.includes('tsdfViewCarve'),
      'wall-clock chunk calibration remains available',
    );
    assert.ok(row.queue.peakSubmittedBytes <= row.queue.limitBytes);
    if (row.mode.startsWith('disabled')) {
      assert.equal(row.ownedDestroyCount, 1);
      assert.ok(row.queue.drainCount > 0);
    }
  }
  assert.deepEqual(result.borrowed, { actual: true, destroyCount: 0 });
  assert.ok(result.invalid.every((name) => name === 'RangeError'));
});

test('an owned native device is destroyed if initialization fails after creation', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice } = await import('/dist/internal/gpu.js');
    const gpu = navigator.gpu;
    const original = gpu.requestAdapter.bind(gpu);
    const descriptor = Object.getOwnPropertyDescriptor(gpu, 'requestAdapter');
    let destroyed = 0,
      created = 0;
    Object.defineProperty(gpu, 'requestAdapter', {
      configurable: true,
      value: async (...args) => {
        const adapter = await original(...args);
        if (!adapter) return null;
        return {
          limits: adapter.limits,
          features: adapter.features,
          info: adapter.info,
          requestDevice: async (options) => {
            const raw = await adapter.requestDevice(options);
            created++;
            const destroy = raw.destroy.bind(raw);
            Object.defineProperty(raw, 'destroy', {
              configurable: true,
              value() {
                destroyed++;
                destroy();
              },
            });
            return raw;
          },
        };
      },
    });
    let reads = 0,
      error;
    const options = { backend: 'webgpu', gpuTimestamps: false };
    Object.defineProperty(options, 'onProgress', {
      get() {
        if (++reads > 2) throw new Error('initialization sentinel');
        return () => {};
      },
    });
    try {
      await ShadeDevice.open(options);
    } catch (caught) {
      error = caught.message;
    } finally {
      if (descriptor) Object.defineProperty(gpu, 'requestAdapter', descriptor);
      else delete gpu.requestAdapter;
    }
    return { created, destroyed, error };
  });
  assert.deepEqual(errors, []);
  assert.deepEqual(result, { created: 1, destroyed: 1, error: 'initialization sentinel' });
});
