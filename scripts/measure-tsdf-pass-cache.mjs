/**
 * Bounded synthetic TSDF comparison. Build the baseline revision separately, then run:
 * TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/measure-tsdf-pass-cache.mjs \
 *   --baseline /absolute/path/to/baseline/dist/internal/gpu.js --pairs 3 --out /tmp/tsdf.json
 *
 * This measures a synthetic multi-box fusion, not an uploaded run or hardware-GPU throughput.
 */
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    current: { type: 'string' },
    backend: { type: 'string', default: 'webgpu' },
    width: { type: 'string', default: '128' },
    height: { type: 'string', default: '96' },
    views: { type: 'string', default: '24' },
    'max-blocks': { type: 'string', default: '64' },
    pairs: { type: 'string', default: '3' },
    warmup: { type: 'string', default: '1' },
    out: { type: 'string' },
  },
});
if (!values.baseline) throw new Error('--baseline must name a separately built baseline bundle');
if (!['cpu', 'webgpu'].includes(values.backend)) throw new Error('--backend must be cpu or webgpu');
const dimensions = Object.fromEntries(
  ['width', 'height', 'views', 'max-blocks', 'pairs'].map((name) => [name, Number(values[name])]),
);
for (const [name, value] of Object.entries(dimensions))
  if (!Number.isSafeInteger(value) || value < 1)
    throw new Error(`${name} must be a positive integer`);
if (dimensions.width * dimensions.height * dimensions.views > 2_000_000 || dimensions.pairs > 5)
  throw new Error('Keep this bounded: at most 2,000,000 input pixels and five pairs');
const warmup = Number(values.warmup);
if (!Number.isSafeInteger(warmup) || warmup < 0 || warmup > 2)
  throw new Error('--warmup must be zero, one or two unmeasured runs per variant');

const { result, errors } = await inBrowser(
  async ({ backend, dimensions, warmup }) => {
    const variants = {
      baseline: await import('/tsdf-benchmark-baseline.mjs'),
      current: await import('/tsdf-benchmark-current.mjs'),
    };
    const { tsdfStagingInput, TSDF_STAGING_OPTIONS, digestArrays, inputChannels, pointDigests } =
      await import('/tests/typeshade/tsdf-staging-cases.mjs');
    const { width, height, views, pairs } = dimensions;
    const input = tsdfStagingInput({ width, height, views });
    const before = await digestArrays(inputChannels(input));
    const devices = {};
    const runs = [];
    try {
      for (const [variant, api] of Object.entries(variants))
        devices[variant] = await api.ShadeDevice.open({ backend, poolBytes: 0 });
      for (let iteration = 0; iteration < warmup; iteration++)
        for (const [variant, api] of Object.entries(variants))
          await api.fuseTsdfGpu(devices[variant], input, {
            ...TSDF_STAGING_OPTIONS,
            maxBlocks: dimensions['max-blocks'],
          });
      for (let pair = 0; pair < pairs; pair++) {
        for (const variant of pair % 2 ? ['current', 'baseline'] : ['baseline', 'current']) {
          const device = devices[variant];
          device.resetKernelStatistics();
          const work = {};
          const previous = new Map();
          const started = performance.now();
          const points = await variants[variant].fuseTsdfGpu(device, input, {
            ...TSDF_STAGING_OPTIONS,
            maxBlocks: dimensions['max-blocks'],
            onProgress(event) {
              const key = [event.phase, event.pass, event.view, event.phaseTotal].join(':');
              const done = previous.get(key) ?? 0;
              if (event.done > done)
                work[event.phase] = (work[event.phase] ?? 0) + event.done - done;
              previous.set(key, event.done);
            },
          });
          const elapsedMs = performance.now() - started;
          runs.push({
            variant,
            pair,
            elapsedMs,
            count: points.count,
            channels: await pointDigests(points),
            work,
            tsdf: { ...points.tsdf, report: undefined },
            kernels: device.kernelStatistics,
            statistics: device.statistics,
            liveBytes: device.statistics.liveBytes,
          });
        }
      }
      return {
        backend,
        dimensions,
        unmeasuredWarmupsPerVariant: warmup,
        inputBefore: before,
        inputAfter: await digestArrays(inputChannels(input)),
        // Structural model of the old TypeCV packView cache; excludes input/runtime/GPU storage.
        baselinePackedHostBytes: 16 * width * height * views,
        devices: Object.fromEntries(
          Object.entries(devices).map(([name, device]) => [name, device.info]),
        ),
        runs,
      };
    } finally {
      for (const device of Object.values(devices)) await device.dispose();
    }
  },
  { backend: values.backend, dimensions, warmup },
  {
    timeoutMs: 60000,
    serve: {
      '/tsdf-benchmark-baseline.mjs': resolve(values.baseline),
      '/tsdf-benchmark-current.mjs': resolve(values.current ?? 'dist/internal/gpu.js'),
    },
  },
);
assert.deepEqual(errors, []);
assert.equal(result.inputAfter, result.inputBefore, 'caller-owned input bytes changed');
for (const run of result.runs) {
  assert.equal(run.liveBytes, 0, `${run.variant} leaked active resident bytes`);
  assert.equal(run.count, result.runs[0].count, `${run.variant} point count changed`);
  assert.deepEqual(run.channels, result.runs[0].channels, `${run.variant} output bits changed`);
}
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
};
const medianMs = Object.fromEntries(
  ['baseline', 'current'].map((variant) => [
    variant,
    median(result.runs.filter((run) => run.variant === variant).map((run) => run.elapsedMs)),
  ]),
);
result.summary = {
  sameOutputBits: true,
  medianMs,
  elapsedChangePercent: 100 * (medianMs.current / medianMs.baseline - 1),
  ...(dimensions.pairs > 1
    ? {
        afterFirstPairMedianMs: Object.fromEntries(
          ['baseline', 'current'].map((variant) => [
            variant,
            median(
              result.runs
                .filter((run) => run.variant === variant && run.pair > 0)
                .map((run) => run.elapsedMs),
            ),
          ]),
        ),
      }
    : {}),
};
const json = JSON.stringify(result, null, 2) + '\n';
if (values.out) await writeFile(resolve(values.out), json);
process.stdout.write(json);
