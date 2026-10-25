import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('TSDF view upload cap preserves both visibility modes without timestamp-query', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, fuseTsdfGpu } = await import('/dist/internal/gpu.js');
    const { tsdfStagingInput, TSDF_STAGING_OPTIONS, pointDigests, digestArrays, inputChannels } =
      await import('/tests/typeshade/tsdf-staging-cases.mjs');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('A WebGPU adapter is required');
    const raw = await adapter.requestDevice({ requiredFeatures: [] });
    const boundedAdapter = await navigator.gpu.requestAdapter();
    if (!boundedAdapter) throw new Error('A WebGPU adapter is required');
    const boundedRaw = await boundedAdapter.requestDevice({ requiredFeatures: [] });
    const devices = [];
    const open = async (options) => {
      const device = await ShadeDevice.open({
        backend: 'webgpu',
        poolBytes: 0,
        chunkTimingSamples: 1,
        occupancyFloors: false,
        ...options,
      });
      devices.push(device);
      return device;
    };
    const input = tsdfStagingInput();
    const before = await digestArrays(inputChannels(input));
    const oneViewBytes = 16 * input.width * input.height;
    const base = { ...TSDF_STAGING_OPTIONS, maxBlocks: 12 };
    const plain = async (points, device) => ({
      count: points.count,
      channels: await pointDigests(points),
      queue: points.tsdf.queuedViewUploads,
      liveBytes: device.statistics.liveBytes,
    });
    try {
      const noQuery = await open({ gpuDevice: raw });
      const variants = [];
      for (const visibility of ['ray-count', 'view-weighted']) {
        const baseline = await fuseTsdfGpu(noQuery, input, {
          ...base,
          visibility,
          maxQueuedViewBytes: Number.MAX_SAFE_INTEGER,
        });
        const capped = await fuseTsdfGpu(noQuery, input, {
          ...base,
          visibility,
          maxQueuedViewBytes: oneViewBytes,
        });
        variants.push({
          visibility,
          baseline: await plain(baseline, noQuery),
          capped: await plain(capped, noQuery),
        });
      }
      const timestampAvailable = adapter.features.has('timestamp-query');
      const normal = await open();
      const normalResult = await fuseTsdfGpu(normal, input, {
        ...base,
        maxQueuedViewBytes: oneViewBytes,
      });
      const bounded = await open({ gpuDevice: boundedRaw, boundedSubmissions: true });
      const boundedResult = await fuseTsdfGpu(bounded, input, {
        ...base,
        maxQueuedViewBytes: oneViewBytes,
      });
      return {
        noQueryHasTimestamps: raw.features.has('timestamp-query'),
        boundedHasTimestamps: boundedRaw.features.has('timestamp-query'),
        timestampAvailable,
        oneViewBytes,
        before,
        after: await digestArrays(inputChannels(input)),
        variants,
        normal: await plain(normalResult, normal),
        bounded: await plain(boundedResult, bounded),
      };
    } finally {
      for (const device of devices) await device.dispose();
      raw.destroy();
      boundedRaw.destroy();
    }
  });
  assert.deepEqual(errors, []);
  assert.equal(result.noQueryHasTimestamps, false);
  assert.equal(result.boundedHasTimestamps, false);
  assert.equal(result.before, result.after);
  for (const { visibility, baseline, capped } of result.variants) {
    assert.ok(baseline.count > 0, visibility);
    assert.equal(capped.count, baseline.count, visibility);
    assert.deepEqual(capped.channels, baseline.channels, visibility);
    assert.equal(Object.keys(capped.channels).length, visibility === 'view-weighted' ? 7 : 6);
    assert.equal(capped.queue.limitBytes, result.oneViewBytes);
    assert.ok(capped.queue.peakSubmittedBytes <= capped.queue.limitBytes, visibility);
    assert.ok(capped.queue.drainCount > 0, `${visibility} exercises a real non-query queue drain`);
    assert.equal(baseline.liveBytes, 0);
    assert.equal(capped.liveBytes, 0);
  }
  const weighted = result.variants.find((variant) => variant.visibility === 'view-weighted').capped;
  for (const run of [result.normal, result.bounded]) {
    assert.equal(run.count, weighted.count);
    assert.deepEqual(run.channels, weighted.channels);
    assert.equal(run.liveBytes, 0);
    assert.ok(run.queue.peakSubmittedBytes <= run.queue.limitBytes);
  }
  assert.equal(result.bounded.queue.drainCount, 0, 'bounded chunks already confirm completion');
  if (result.timestampAvailable)
    assert.equal(
      result.normal.queue.drainCount,
      0,
      'timestamp readbacks already confirm completion',
    );
  console.log(
    JSON.stringify({
      viewUploadQueues: result.variants.map(({ visibility, baseline, capped }) => ({
        visibility,
        baseline: baseline.queue,
        capped: capped.queue,
      })),
      timestamp: result.normal.queue,
      bounded: result.bounded.queue,
    }),
  );
});

test('TSDF view upload admission cleans up when its queue wait fails or is aborted', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, fuseTsdfGpu } = await import('/dist/internal/gpu.js');
    const { tsdfStagingInput, TSDF_STAGING_OPTIONS, digestArrays, inputChannels } =
      await import('/tests/typeshade/tsdf-staging-cases.mjs');
    const input = tsdfStagingInput();
    const before = await digestArrays(inputChannels(input));
    const outcomes = [];
    for (const kind of ['failure', 'abort']) {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('A WebGPU adapter is required');
      const raw = await adapter.requestDevice({ requiredFeatures: [] });
      const device = await ShadeDevice.open({
        backend: 'webgpu',
        gpuDevice: raw,
        poolBytes: 0,
        chunkTimingSamples: 1,
        occupancyFloors: false,
      });
      const controller = new AbortController();
      let waits = 0;
      let writes = 0;
      let arrays = 0;
      const allocate = device.allocateU32.bind(device);
      device.allocateU32 = (length, values) => {
        const handle = allocate(length, values);
        // The first four allocations are the only per-view input channels.
        if (arrays++ < 4) {
          const write = handle.write.bind(handle);
          handle.write = (source) => {
            writes++;
            write(source);
          };
        }
        return handle;
      };
      device.waitForSubmitted = () => {
        waits++;
        if (kind === 'failure') return Promise.reject(new Error('synthetic queue wait failure'));
        controller.abort();
        return new Promise(() => {});
      };
      try {
        await fuseTsdfGpu(device, input, {
          ...TSDF_STAGING_OPTIONS,
          maxBlocks: 12,
          maxQueuedViewBytes: 16 * input.width * input.height,
          signal: controller.signal,
        });
        outcomes.push({ kind, error: null, waits, writes, liveBytes: device.statistics.liveBytes });
      } catch (error) {
        outcomes.push({
          kind,
          error: kind === 'abort' ? error.name : error.message,
          waits,
          writes,
          liveBytes: device.statistics.liveBytes,
        });
      } finally {
        await device.dispose();
        raw.destroy();
      }
    }
    return { before, after: await digestArrays(inputChannels(input)), outcomes };
  });
  assert.deepEqual(errors, []);
  assert.equal(result.before, result.after);
  for (const outcome of result.outcomes) {
    assert.equal(outcome.waits, 1, outcome.kind);
    assert.ok(
      outcome.writes >= 4 && outcome.writes <= 8,
      'only initial complete views were written',
    );
    assert.equal(outcome.writes % 4, 0, 'a blocked upload admitted none of its channel writes');
    assert.equal(outcome.liveBytes, 0, outcome.kind);
    assert.equal(
      outcome.error,
      outcome.kind === 'abort' ? 'AbortError' : 'synthetic queue wait failure',
    );
  }
  console.log(JSON.stringify({ uploadWaitFailures: result.outcomes }));
});
