// Compare the complete BA stage; pipeline preparation is warmed before timed samples.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const { values } = parseArgs({
  options: {
    hardware: { type: 'boolean', default: false },
    samples: { type: 'string', default: '5' },
    warmup: { type: 'string', default: '2' },
  },
});
const samples = Number(values.samples),
  warmup = Number(values.warmup);
if (![samples, warmup].every((n) => Number.isSafeInteger(n) && n >= 1 && n <= 20))
  throw new RangeError('samples and warmup must be integers in [1,20]');
if (values.hardware) process.env.TYPECV_HARDWARE_GPU = '1';
const sources = [
  'src/gpu/bundle.shade.ts',
  'src/gpu/joint-focal.shade.ts',
  'src/gpu/bundle.ts',
  'src/gpu/joint-focal.ts',
  'src/gpu/bundle-camera-product.ts',
  'src/gpu/device.ts',
  'tests/typeshade/geometry-cases.mjs',
  'scripts/benchmark-ba-camera-products.mjs',
];
const hashes = Object.fromEntries(
  await Promise.all(
    sources.map(async (p) => [
      p,
      createHash('sha256')
        .update(await readFile(p))
        .digest('hex'),
    ]),
  ),
);
const pkg = JSON.parse(await readFile('package.json', 'utf8'));
const { result, errors } = await inBrowser(
  async ({ samples, warmup, hardware }) => {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const description = [
      adapter.info?.vendor,
      adapter.info?.architecture,
      adapter.info?.description,
    ]
      .filter(Boolean)
      .join(' ');
    const software =
      /swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter === true;
    if (hardware && (software || !description))
      throw new Error(`Native adapter required: ${description}`);
    const timestamps = adapter.features.has('timestamp-query');
    const gpu = await adapter.requestDevice({
      requiredFeatures: timestamps ? ['timestamp-query'] : [],
    });
    const validation = [],
      rows = [];
    gpu.addEventListener('uncapturederror', (e) => validation.push(e.error.message));
    const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
    const { bundleFixture } = await import('/tests/typeshade/geometry-cases.mjs');
    const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
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
        { name: 'few cameras, long lists', views: 4, count: 1025, shared: false },
        { name: 'shared focal', views: 17, count: 256, shared: true },
        { name: 'skewed camera lists', views: 17, count: 512, shared: false, skew: true },
      ]) {
        const input = bundleFixture(config);
        if (config.skew)
          input.landmarks.forEach((p, i) => {
            const extra = 2 + (i % (config.views - 2));
            p.observations = p.observations.filter((_, c) => c < 2 || c === extra);
          });
        const inputSha256 = await hash(input);
        const counts = input.views.map((v) =>
          input.landmarks.reduce(
            (n, p) => n + p.observations.filter((o) => o.viewId === v.id).length,
            0,
          ),
        );
        const variants = [
          { name: 'serial rows', cameraProduct: 'rows' },
          ...[32, 64, 128, 256, 'auto'].map((width) => ({
            name: `camera ${width}`,
            cameraProduct: 'workgroups',
            cameraProductWorkgroupSize: width,
          })),
          ...[128, 256, 512].map((tile) => ({
            name: `camera 64, tile ${tile}`,
            cameraProduct: 'workgroups',
            cameraProductWorkgroupSize: 64,
            cameraProductTileSize: tile,
          })),
        ];
        const devices = [];
        try {
          for (const v of variants)
            devices.push(
              await ShadeDevice.open({
                gpuDevice: gpu,
                poolBytes: 0,
                chunkTimingSamples: Infinity,
              }),
            );
          const raw = variants.map(() => []);
          let referenceGeometry;
          const run = async (i) => {
            const device = devices[i],
              variant = variants[i],
              before = device.statistics,
              start = performance.now();
            const out = await bundleAdjustGpu(device, input, {
              iterations: 3,
              maxCgIterations: 96,
              refineSharedFocal: config.shared,
              schurReduction: 'serial',
              ...variant,
            });
            const wallMs = performance.now() - start,
              report = out.execution.report;
            if (
              report.tier !== 'webgpu' ||
              device.statistics.liveBuffers ||
              out.linearSolverFailures
            )
              throw new Error(`Invalid BA result: ${config.name}, ${variant.name}`);
            const kernel = variant.cameraProductTileSize
              ? 'bundleSchurCameraSegments'
              : 'bundleSchurCameraWorkgroup';
            const shapes = report.groups.flatMap((g) =>
              g.calls.flatMap((call, j) => (call === kernel ? [g.workgroupShapes?.[j]?.[0]] : [])),
            );
            if (
              variant.cameraProduct === 'workgroups' &&
              (!shapes.length || shapes.some((n) => ![32, 64, 128, 256].includes(n)))
            )
              throw new Error(`Camera product fallback or missing shape: ${variant.name}`);
            const geometry = [
              ...out.reconstruction.views.flatMap((v) => [
                ...v.pose.R,
                ...v.pose.t,
                v.camera.fx,
                v.camera.fy,
              ]),
              ...out.reconstruction.landmarks.flatMap((p) => [...p.xyz]),
            ];
            if (i === 0 && !referenceGeometry) referenceGeometry = geometry;
            if (!referenceGeometry) throw new Error('Reference must warm first');
            const maxGeometryRelativeError = geometry.reduce(
              (maximum, value, j) =>
                Math.max(
                  maximum,
                  Math.abs(value - referenceGeometry[j]) /
                    Math.max(1, Math.abs(referenceGeometry[j])),
                ),
              0,
            );
            if (!Number.isFinite(maxGeometryRelativeError) || maxGeometryRelativeError > 1e-6)
              throw new Error(
                `BA geometry differs: ${config.name}, ${variant.name}, ${maxGeometryRelativeError}`,
              );
            return {
              maxGeometryRelativeError,
              wallMs,
              gpuMs: report.groups.every((g) => Number.isFinite(g.gpuTime))
                ? report.groups.reduce((n, g) => n + g.gpuTime, 0) / 1e6
                : null,
              workgroupSizes: [...new Set(shapes)],
              cost: out.diagnostics.finalCost,
              cgIterations: out.cgIterations,
              accepted: out.diagnostics.acceptedSteps,
              rejected: out.diagnostics.rejectedSteps,
              failures: out.linearSolverFailures,
              dispatches: report.dispatches,
              submissions: report.submissions,
              trackedUploadBytes:
                device.statistics.applicationUploadBytes - before.applicationUploadBytes,
              trackedReadbackBytes:
                device.statistics.applicationReadbackBytes - before.applicationReadbackBytes,
            };
          };
          for (let round = 0; round < warmup + samples; round++) {
            const order = variants.map((_, i) => (i + round) % variants.length);
            if (round % 2) order.reverse();
            for (const i of order) {
              const value = await run(i);
              if (round >= warmup) raw[i].push(value);
            }
          }
          const reference = raw[0][0];
          for (let i = 0; i < raw.length; i++)
            for (const r of raw[i]) {
              if (
                Math.abs(r.cost - reference.cost) > 1e-6 * Math.max(1, Math.abs(reference.cost)) ||
                r.accepted !== reference.accepted ||
                r.rejected !== reference.rejected
              )
                throw new Error(`BA convergence differs: ${config.name}, ${variants[i].name}`);
            }
          rows.push({
            ...config,
            inputSha256,
            observationsPerCamera: counts,
            variants: variants.map((v, i) => ({
              ...v,
              wallMedianMs: median(raw[i].map((r) => r.wallMs)),
              gpuMedianMs: raw[i].every((r) => r.gpuMs !== null)
                ? median(raw[i].map((r) => r.gpuMs))
                : null,
              raw: raw[i],
            })),
          });
        } finally {
          for (const d of devices) await d.dispose();
        }
      }
      if (validation.length) throw new Error(validation.join('; '));
      return { description, software, timestamps, rows };
    } finally {
      gpu.destroy();
    }
  },
  { samples, warmup, hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
  { timeoutMs: 1200000 },
);
assert.deepEqual(errors, []);
console.log(
  JSON.stringify(
    {
      methodology:
        'Complete resident BA after pipeline warmup; scalar assembly for all variants. Rotate/reverse variant order. GPU time is null without full timestamp coverage. Tracked uploads exclude direct Resident.write. Synthetic fixtures, not full SfM. Native auto is an unvalidated structural heuristic; software auto uses 32 lanes.',
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
