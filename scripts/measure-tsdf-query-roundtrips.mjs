/** Compare native query/non-query devices; no feature masking and no production-default changes. */
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const { values } = parseArgs({ options: { bundle: { type: 'string' }, out: { type: 'string' } } });
const { result, errors } = await inBrowser(
  async () => {
    const { ShadeDevice, fuseTsdfGpu } = await import('/tsdf-query-candidate.mjs');
    const { tsdfStagingInput, TSDF_STAGING_OPTIONS, inputChannels, digestArrays, pointDigests } =
      await import('/tests/typeshade/tsdf-staging-cases.mjs');
    const input = tsdfStagingInput({ width: 128, height: 96, views: 24 });
    const before = await digestArrays(inputChannels(input));
    const specs = [
      { name: 'query-standalone', query: true, standaloneVoteReset: true },
      { name: 'query-inline', query: true, standaloneVoteReset: false },
      { name: 'no-query-inline', query: false, standaloneVoteReset: false },
    ];
    const devices = [],
      rawDevices = [],
      adapters = [],
      runs = [],
      warmups = [];
    let expected;
    try {
      for (const spec of specs) {
        // Dawn consumes the JS adapter after requestDevice; fresh handles must report identical adapter metadata.
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('A WebGPU adapter is required');
        if (spec.query && !adapter.features.has('timestamp-query'))
          throw new Error('This comparison requires timestamp-query support');
        const raw = await adapter.requestDevice({
          requiredFeatures: spec.query ? ['timestamp-query'] : [],
        });
        rawDevices.push(raw);
        adapters.push({
          vendor: adapter.info.vendor,
          architecture: adapter.info.architecture,
          description: adapter.info.description,
          fallback: adapter.isFallbackAdapter ?? adapter.info.isFallbackAdapter ?? null,
          maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
        });
        devices.push(
          await ShadeDevice.open({
            backend: 'webgpu',
            gpuDevice: raw,
            poolBytes: 0,
            occupancyFloors: false,
            chunkTimingSamples: 8,
          }),
        );
      }
      const run = async (index, round, measured) => {
        const device = devices[index],
          spec = specs[index];
        device.resetKernelStatistics();
        const started = performance.now();
        const points = await fuseTsdfGpu(device, input, {
          ...TSDF_STAGING_OPTIONS,
          maxBlocks: 64,
          standaloneVoteReset: spec.standaloneVoteReset,
          maxQueuedViewBytes: 64 * 1024 ** 2,
        });
        const elapsedMs = performance.now() - started;
        const channels = await pointDigests(points);
        expected ??= channels;
        if (JSON.stringify(channels) !== JSON.stringify(expected))
          throw new Error('Output bits changed');
        const row = {
          variant: spec.name,
          round,
          elapsedMs,
          points: points.count,
          channels,
          nativeFeatures: Array.from(rawDevices[index].features).sort(),
          passes: points.tsdf.passes,
          blocks: points.tsdf.blocks,
          inlineVoteResetGroups: points.tsdf.inlineVoteResetGroups,
          viewUploadBytes: points.tsdf.uploadBytes,
          queue: points.tsdf.queuedViewUploads,
          dispatches: points.tsdf.report.dispatches,
          submissions: points.tsdf.report.submissions,
          kernels: device.kernelStatistics.rows,
          liveBytes: device.statistics.liveBytes,
        };
        (measured ? runs : warmups).push(row);
      };
      for (let index = 0; index < specs.length; index++) await run(index, -1, false);
      for (let round = 0; round < 3; round++)
        for (let offset = 0; offset < specs.length; offset++)
          await run((round + offset) % specs.length, round, true);
      return {
        schema: 'typecv-tsdf-query-roundtrips-v1',
        fixture: { width: 128, height: 96, views: 24, maxBlocks: 64 },
        adapters,
        warmups,
        runs,
        inputBefore: before,
        inputAfter: await digestArrays(inputChannels(input)),
        caveat:
          'Synthetic software/hardware adapter result only; not an uploaded run or full-process memory measurement.',
      };
    } finally {
      for (const device of devices) await device.dispose();
      for (const raw of rawDevices) raw.destroy();
    }
  },
  undefined,
  {
    timeoutMs: 45000,
    serve: { '/tsdf-query-candidate.mjs': resolve(values.bundle ?? 'dist/internal/gpu.js') },
  },
);
assert.deepEqual(errors, []);
assert.equal(result.inputBefore, result.inputAfter);
assert.ok(
  result.adapters.every(
    (adapter) => JSON.stringify(adapter) === JSON.stringify(result.adapters[0]),
  ),
);
for (const row of [...result.warmups, ...result.runs]) {
  assert.equal(row.liveBytes, 0);
  assert.equal(Object.keys(row.channels).length, 7);
  assert.equal(row.nativeFeatures.includes('timestamp-query'), row.variant !== 'no-query-inline');
  assert.ok(row.queue.peakSubmittedBytes <= row.queue.limitBytes);
}
const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
result.medianMs = Object.fromEntries(
  ['query-standalone', 'query-inline', 'no-query-inline'].map((name) => [
    name,
    median(result.runs.filter((row) => row.variant === name).map((row) => row.elapsedMs)),
  ]),
);
const text = JSON.stringify(result, null, 2) + '\n';
if (values.out) await writeFile(resolve(values.out), text);
process.stdout.write(text);
