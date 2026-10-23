import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const { values } = parseArgs({
  options: {
    hardware: { type: 'boolean', default: false },
    samples: { type: 'string', default: '5' },
    stage: { type: 'string', default: 'all' },
    warmup: { type: 'string', default: '2' },
  },
});
if (!['all', 'bundle', 'dense'].includes(values.stage))
  throw new RangeError('stage must be all, bundle or dense');
const samples = Number(values.samples),
  warmup = Number(values.warmup);
if (![samples, warmup].every((n) => Number.isSafeInteger(n) && n >= 1 && n <= 20))
  throw new RangeError('samples and warmup must be integers in [1, 20]');
if (values.hardware) process.env.TYPECV_HARDWARE_GPU = '1';
const files = [
  'src/gpu/bundle.shade.ts',
  'src/gpu/bundle.ts',
  'src/gpu/joint-focal.ts',
  'src/gpu/device.ts',
  'scripts/benchmark-gpu-invariants.mjs',
  'src/gpu/joint-focal.shade.ts',
  'src/gpu/mvs.shade.ts',
  'tests/typeshade/geometry-cases.mjs',
  'tests/typeshade/mvs-cases.mjs',
];
const hashes = Object.fromEntries(
  await Promise.all(
    files.map(async (name) => [
      name,
      createHash('sha256')
        .update(await readFile(name))
        .digest('hex'),
    ]),
  ),
);
const pkg = JSON.parse(await readFile('package.json', 'utf8'));
const { result, errors } = await inBrowser(
  async ({ samples, warmup, hardware, stage }) => {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const description = [
        adapter.info?.vendor,
        adapter.info?.architecture,
        adapter.info?.device,
        adapter.info?.description,
      ]
        .filter(Boolean)
        .join(' '),
      software =
        /swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter === true;
    if (hardware && software)
      throw new Error(`Hardware requested, software adapter selected: ${description}`);
    const timestamps = adapter.features.has('timestamp-query'),
      gpu = await adapter.requestDevice({
        requiredFeatures: timestamps ? ['timestamp-query'] : [],
      }),
      validation = [],
      rows = [];
    gpu.addEventListener('uncapturederror', (event) => validation.push(event.error.message));
    const publicApi = await import('/dist/gpu.js'),
      denseApi = await import('/dist/internal/gpu.js'),
      { bundleFixture } = await import('/tests/typeshade/geometry-cases.mjs'),
      { mvsScene } = await import('/tests/typeshade/mvs-cases.mjs');
    const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
    const hash = async (input) =>
      [
        ...new Uint8Array(
          await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(input))),
        ),
      ]
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
    try {
      for (const config of [
        { name: 'BA fixed 4x256', views: 4, count: 256, shared: false },
        { name: 'BA shared focal 17x256', views: 17, count: 256, shared: true },
        { name: 'Dense photometric 128x96', dense: true },
      ]) {
        if ((stage === 'bundle' && config.dense) || (stage === 'dense' && !config.dense)) continue;
        const variants = config.dense ? [0, 1] : [0, 1, 2];
        const api = config.dense ? denseApi : publicApi,
          input = config.dense ? mvsScene() : bundleFixture(config),
          plan = config.dense ? denseApi.planMvsViews(input) : undefined,
          devices = await Promise.all(
            variants.map(() =>
              api.ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0, chunkTimingSamples: Infinity }),
            ),
          ),
          runs = variants.map(() => []);
        const inputSha256 = await hash(
          config.dense
            ? {
                views: input.views,
                landmarks: input.landmarks,
                images: [...input.images].map(([id, image]) => [id, [...image.rgba]]),
              }
            : input,
        );
        const run = async (variant) => {
          const device = devices[variant],
            before = device.statistics,
            started = performance.now();
          const out = config.dense
            ? await api.mvsPhotometricGpu(device, input, plan, input.images, plan.views[2].index, {
                photometricIterations: 2,
                relativePoseCache: !!variant,
              })
            : await api.bundleAdjustGpu(device, input, {
                iterations: 4,
                maxCgIterations: 64,
                refineSharedFocal: config.shared,
                schurReduction: variant === 1 ? 'cooperative' : 'serial',
                cameraProduct: variant === 2 ? 'workgroups' : 'rows',
              });
          const wallMs = performance.now() - started,
            report = config.dense ? out.report : out.execution.report;
          if (report.tier !== 'webgpu') throw new Error('Benchmark fell back from WebGPU');
          if (
            !config.dense &&
            variant === 2 &&
            !report.groups.some((g) => g.calls.includes('bundleSchurCameraWorkgroup'))
          )
            throw new Error(
              'Per-camera workgroups require a float-fp64 owner; integer owners retain rows until native validation.',
            );
          if (device.statistics.liveBuffers) throw new Error('Benchmark leaked resident buffers');
          return {
            wallMs,
            gpuMs: report.groups.every((g) => Number.isFinite(g.gpuTime))
              ? report.groups.reduce((n, g) => n + g.gpuTime, 0) / 1e6
              : null,
            trackedUploadBytes:
              device.statistics.applicationUploadBytes - before.applicationUploadBytes,
            trackedReadbackBytes:
              device.statistics.applicationReadbackBytes - before.applicationReadbackBytes,
            dispatches: report.dispatches,
            submissions: report.submissions,
            ...(config.dense
              ? { kept: out.cost.filter((c) => c <= 0.9).length }
              : {
                  cost: out.diagnostics.finalCost,
                  cgIterations: out.cgIterations,
                  failures: out.linearSolverFailures,
                }),
          };
        };
        try {
          for (let round = 0; round < warmup + samples; round++) {
            const offset = round % variants.length;
            const order = [...variants.slice(offset), ...variants.slice(0, offset)];
            if (round % 2) order.reverse();
            for (const variant of order) {
              const value = await run(variant);
              if (round >= warmup) runs[variant].push(value);
            }
          }
          rows.push({
            ...config,
            inputSha256,
            variants: runs.map((raw, i) => ({
              name: config.dense
                ? i
                  ? 'cached poses'
                  : 'scalar poses'
                : ['serial', 'cooperative rows', 'camera workgroups'][i],
              wallMedianMs: median(raw.map((r) => r.wallMs)),
              gpuMedianMs: raw.every((r) => r.gpuMs !== null)
                ? median(raw.map((r) => r.gpuMs))
                : null,
              raw,
            })),
          });
        } finally {
          for (const device of devices) await device.dispose();
        }
      }
      if (validation.length) throw new Error(`GPU validation: ${validation.join('; ')}`);
      return { description, software, timestamps, rows };
    } finally {
      gpu.destroy();
    }
  },
  { samples, warmup, hardware: process.env.TYPECV_HARDWARE_GPU === '1', stage: values.stage },
  { timeoutMs: 600000 },
);
assert.deepEqual(errors, []);
console.log(
  JSON.stringify(
    {
      methodology:
        'Complete BA / photometric stage after pipeline warmup, including resident allocation, uploads, kernel dispatches and terminal readback; variant order rotates and reverses. GPU time is null unless every group has a timestamp. Transfer counters cover ShadeDevice hooks; direct Resident.write uploads are not counted. Synthetic fixtures, not full SfM or full Dense reconstruction.',
      samples,
      warmup,
      typeshade: pkg.devDependencies.typeshade,
      hashes,
      ...result,
    },
    null,
    2,
  ),
);
