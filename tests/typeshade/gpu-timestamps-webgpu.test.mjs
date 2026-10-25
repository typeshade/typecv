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

test('native timestamp modes preserve the complete tiny MVS depth stage and fused points', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, reconstructDense } = await import('/dist/internal/gpu.js');
      const { mvsScene } = await import('/tests/typeshade/mvs-cases.mjs');
      const { digestArrays, pointDigests } =
        await import('/tests/typeshade/tsdf-staging-cases.mjs');
      const scene = mvsScene({ width: 32, height: 24, seed: 17 });
      const rgba = Array.from(scene.images.values(), (image) => image.rgba);
      const before = await digestArrays(rgba);
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter?.features.has('timestamp-query'))
        throw new Error('Timestamp support is required');
      const native = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
      const devices = [],
        runs = [];
      try {
        for (const query of [true, false]) {
          const device = await ShadeDevice.open({
            backend: 'webgpu',
            poolBytes: 0,
            ...(query ? { gpuDevice: native } : { gpuTimestamps: false }),
          });
          devices.push(device);
          const diagnostics = [],
            fullMaps = [];
          let tsdfChannels;
          const dense = await reconstructDense(device, scene, scene.images, {
            seed: 19,
            photometricIterations: 1,
            geometricIterations: 1,
            windowRadius: 2,
            windowStep: 2,
            fusion: 'tsdf',
            planarDepth: true,
            tsdf: { visibility: 'view-weighted', levels: 2, voxelScale: 1, truncation: 3 },
            onDepthDiagnostic: async (event) => {
              diagnostics.push({
                stage: event.stage,
                viewId: event.viewId,
                width: event.width,
                height: event.height,
                depth: await digestArrays([event.depth]),
                positive: event.depth.reduce((n, depth) => n + (depth > 0 ? 1 : 0), 0),
                confidence: event.confidence ? await digestArrays([event.confidence]) : null,
              });
            },
            onDepthMaps: async (maps) => {
              fullMaps.push({
                depth: await digestArrays(maps.depth),
                normal: await digestArrays(maps.normal),
                confidence: maps.confidence ? await digestArrays(maps.confidence) : null,
                lengths: maps.depth.map((depth) => depth.length),
              });
            },
            onTsdfPoints: async (points) => {
              tsdfChannels = await pointDigests(points);
            },
          });
          runs.push({
            query: device.info.webgpu.features.includes('timestamp-query'),
            diagnostics,
            fullMaps,
            tsdfChannels,
            finalChannels: await pointDigests(dense.points),
            points: dense.points.count,
            views: dense.views,
            sourceUnchanged: before === (await digestArrays(rgba)),
            liveBytes: device.statistics.liveBytes,
          });
        }
        return { runs };
      } finally {
        for (const device of devices) await device.dispose();
        native.destroy();
      }
    },
    undefined,
    { timeoutMs: 60000 },
  );
  assert.deepEqual(errors, []);
  const [profiled, fast] = result.runs;
  assert.equal(profiled.query, true);
  assert.equal(fast.query, false);
  assert.deepEqual(fast.diagnostics, profiled.diagnostics);
  assert.deepEqual(fast.fullMaps, profiled.fullMaps);
  assert.deepEqual(fast.tsdfChannels, profiled.tsdfChannels);
  assert.equal(Object.keys(fast.tsdfChannels).length, 7);
  assert.deepEqual(fast.finalChannels, profiled.finalChannels);
  assert.deepEqual(fast.views, profiled.views);
  assert.equal(fast.points, profiled.points);
  assert.ok(fast.points > 0);
  for (const run of result.runs) {
    assert.equal(run.sourceUnchanged, true);
    assert.equal(run.liveBytes, 0);
    assert.equal(run.diagnostics.length, 20);
    assert.equal(run.fullMaps.length, 1, 'TSDF exposes final completed maps once');
    assert.ok(run.diagnostics.every((map) => map.width === 32 && map.height === 24));
    assert.ok(
      run.diagnostics
        .filter((map) => ['geometric-filtered', 'planar'].includes(map.stage))
        .every((map) => map.confidence !== null),
    );
    assert.ok(
      run.fullMaps.every((maps) => maps.confidence && maps.lengths.every((n) => n === 768)),
    );
    for (const stage of ['photometric', 'geometric-unfiltered', 'geometric-filtered', 'planar'])
      assert.ok(
        run.diagnostics.some((map) => map.stage === stage && map.positive > 0),
        stage,
      );
  }
});
