// Complete synthetic Dense MVS, including uploads, depth passes, pixel fusion and readbacks.
// The output contains aggregates/hashes only; no frame, point or depth-map data is published.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { execFileSync } from 'node:child_process';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    out: { type: 'string' },
    repetitions: { type: 'string', default: '3' },
    width: { type: 'string', default: '160' },
    height: { type: 'string', default: '96' },
    photometric: { type: 'string', default: '8' },
    geometric: { type: 'string', default: '2' },
    'timeout-ms': { type: 'string', default: '1200000' },
    hardware: { type: 'boolean', default: false },
  },
});
if (!values.baseline || !values.out)
  throw new Error(
    'Usage: node scripts/measure-mvs-pipeline-performance.mjs --baseline <frozen-internal.mjs> --out <aggregate.json> [--repetitions 3] [--hardware]',
  );
const baseline = resolve(values.baseline),
  output = resolve(values.out),
  repetitions = Number(values.repetitions),
  width = Number(values.width),
  height = Number(values.height),
  photometricIterations = Number(values.photometric),
  geometricIterations = Number(values.geometric),
  timeoutMs = Number(values['timeout-ms']);
if (!Number.isSafeInteger(repetitions) || repetitions < 1 || repetitions > 20)
  throw new RangeError('repetitions must be an integer in [1,20]');
if (![width, height].every((n) => Number.isSafeInteger(n) && n >= 32 && n <= 4096))
  throw new RangeError('width and height must be integers in [32,4096]');
if (
  ![photometricIterations, geometricIterations].every(
    (n) => Number.isSafeInteger(n) && n >= 0 && n <= 64,
  )
)
  throw new RangeError('iteration counts must be integers in [0,64]');
if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 1200000)
  throw new RangeError('timeout-ms must be in [1000,1200000]');
if (values.hardware) process.env.TYPECV_HARDWARE_GPU = '1';
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex'),
  files = [
    'src/mvs/gpu/mvs.shade.ts',
    'src/mvs/gpu/mvs.ts',
    'src/mvs/gpu/mvs-residency.ts',
    'src/device/gpu/device.ts',
    'tests/typeshade/mvs-cases.mjs',
    'scripts/measure-mvs-pipeline-performance.mjs',
  ],
  hashes = Object.fromEntries(
    await Promise.all(
      files.map(async (name) => [
        name,
        hash(await readFile(new URL('../' + name, import.meta.url))),
      ]),
    ),
  ),
  baselineHash = hash(await readFile(baseline)),
  optimizedBundleHash = hash(await readFile(new URL('../dist/internal/gpu.js', import.meta.url))),
  revision = execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: new URL('..', import.meta.url),
    encoding: 'utf8',
  }).trim(),
  pkg = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));

const { result, errors } = await inBrowser(
  async ({ repetitions, width, height, photometricIterations, geometricIterations, hardware }) => {
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
        adapter.isFallbackAdapter === true || /swiftshader|software|llvmpipe/i.test(description),
      timestamps = adapter.features.has('timestamp-query');
    if (hardware && software)
      throw new Error(`Hardware requested, software adapter selected: ${description}`);
    const gpu = await adapter.requestDevice({
        requiredFeatures: timestamps ? ['timestamp-query'] : [],
      }),
      validation = [],
      apis = {
        baseline: await import('/baseline-internal.mjs'),
        optimized: await import('/dist/internal/gpu.js'),
      },
      { mvsScene, nearestSurface } = await import('/tests/typeshade/mvs-cases.mjs'),
      scene = mvsScene({ width, height }),
      plan = apis.optimized.planMvsViews(scene),
      scale = plan.normalization.scale,
      rows = [],
      cold = [];
    gpu.addEventListener('uncapturederror', (event) => validation.push(event.error.message));
    const digest = async (bytes) =>
        [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
          .map((b) => b.toString(16).padStart(2, '0'))
          .join(''),
      jsonHash = async (x) => digest(new TextEncoder().encode(JSON.stringify(x))),
      arrayHash = async (a) => digest(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)),
      median = (a) => {
        const s = [...a].sort((x, y) => x - y),
          m = Math.floor(s.length / 2);
        return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
      },
      summarize = (a) => ({
        samples: a,
        median: median(a),
        min: Math.min(...a),
        max: Math.max(...a),
      });
    const inputHash = await jsonHash({
      views: scene.views,
      landmarks: scene.landmarks,
      images: [...scene.images].map(([id, image]) => [
        id,
        { width: image.width, height: image.height, rgba: [...image.rgba] },
      ]),
    });
    const bytesOf = (a) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    const compareArray = (a, b, label) => {
      if (
        !ArrayBuffer.isView(a) ||
        !ArrayBuffer.isView(b) ||
        a.constructor.name !== b.constructor.name ||
        a.length !== b.length
      )
        throw new Error(`Dense array layout changed: ${label}`);
      const x = bytesOf(a),
        y = bytesOf(b);
      for (let i = 0; i < x.length; i++)
        if (x[i] !== y[i]) throw new Error(`Dense array differs: ${label} byte ${i}`);
      return x.length;
    };
    const surfaceMetrics = (points) => {
      const distances = [];
      for (let i = 0; i < points.count; i++)
        distances.push(
          nearestSurface(Array.from(points.positions.subarray(3 * i, 3 * i + 3))).distance / scale,
        );
      distances.sort((a, b) => a - b);
      const quantile = (p) =>
        distances.length ? distances[Math.floor((distances.length - 1) * p)] : null;
      return {
        points: points.count,
        within1pct: distances.length
          ? distances.filter((d) => d <= 0.01).length / distances.length
          : null,
        within5pct: distances.length
          ? distances.filter((d) => d <= 0.05).length / distances.length
          : null,
        medianDistance: quantile(0.5),
        p95Distance: quantile(0.95),
      };
    };
    async function run(version) {
      const api = apis[version],
        device = await api.ShadeDevice.open({
          gpuDevice: gpu,
          poolBytes: 0,
          boundedSubmissions: false,
        }),
        started = performance.now(),
        stageEnds = {},
        events = [];
      let maps;
      try {
        const out = await api.reconstructDense(device, scene, scene.images, {
            photometricIterations,
            geometricIterations,
            onDepthMaps: (m) => {
              maps = {
                depth: m.depth.map((a) => a.slice()),
                normal: m.normal.map((a) => a.slice()),
                ...(m.confidence ? { confidence: m.confidence.map((a) => a.slice()) } : {}),
              };
            },
            onProgress: (event) => {
              events.push({ ...event, elapsedMs: performance.now() - started });
              if (event.done === event.total) stageEnds[event.stage] = performance.now() - started;
            },
          }),
          wallMs = performance.now() - started;
        if (out.report.tier !== 'webgpu') throw new Error('Dense benchmark fell back from WebGPU');
        if (!maps || !out.points.count)
          throw new Error('Dense fixture produced no complete maps/points');
        const statistics = device.statistics,
          runtimeStatistics = { ...device.shade.statistics },
          diagnostics = device.diagnostics;
        if (statistics.liveBytes !== 0 || statistics.liveBuffers !== 0)
          throw new Error('Dense pipeline leaked resident arrays');
        const hashes = {
          positions: await arrayHash(out.points.positions),
          colors: await arrayHash(out.points.colors),
          normals: await arrayHash(out.points.normals),
          depth: await Promise.all(maps.depth.map(arrayHash)),
          depthNormals: await Promise.all(maps.normal.map(arrayHash)),
        };
        const measured = out.report.groups.filter((g) => Number.isFinite(g.gpuTime)),
          gpuCoveredMs = measured.reduce((n, g) => n + g.gpuTime, 0) / 1e6;
        return {
          raw: { points: out.points, maps, views: out.views, residency: out.residency },
          aggregate: {
            wallMs,
            stageWallMs: {
              photometric: stageEnds.photometric ?? null,
              geometric:
                stageEnds.geometric === undefined || stageEnds.photometric === undefined
                  ? null
                  : stageEnds.geometric - stageEnds.photometric,
              fusionAndFinalize:
                stageEnds.geometric === undefined ? null : wallMs - stageEnds.geometric,
            },
            progress: events,
            points: out.points.count,
            depthPixels: out.views.map((v) => ({ id: v.id, count: v.depthPixels })),
            surfaceMetrics: surfaceMetrics(out.points),
            hashes,
            residency: out.residency,
            gpu: {
              totalMs: measured.length === out.report.groups.length ? gpuCoveredMs : null,
              timestampCoveredMs: gpuCoveredMs,
              timedGroups: measured.length,
              totalGroups: out.report.groups.length,
              dispatches: out.report.dispatches,
              submissions: out.report.submissions,
            },
            transfers: {
              runtimeUploadBytes: runtimeStatistics.uploadBytes ?? null,
              runtimeReadBytes: runtimeStatistics.readBytes ?? null,
              applicationTrackedUploadBytes: statistics.applicationUploadBytes,
              applicationReadbackBytes: statistics.applicationReadbackBytes,
              reads: statistics.reads,
              readWaves: statistics.readWaves,
              residentSnapshotBytes: statistics.residentSnapshotBytes,
            },
            managedMemory: diagnostics.memory,
            allocations: statistics.allocations,
            reuses: statistics.reuses,
            kernels: statistics.kernels,
          },
        };
      } finally {
        await device.dispose();
      }
    }
    function compare(a, b) {
      if (JSON.stringify(a.views) !== JSON.stringify(b.views))
        throw new Error('Dense view selection or output metadata changed');
      if (JSON.stringify(a.residency) !== JSON.stringify(b.residency))
        throw new Error('Dense residency changed');
      let comparedBytes = 0;
      for (const key of ['positions', 'colors', 'normals'])
        comparedBytes += compareArray(a.points[key], b.points[key], 'points.' + key);
      for (const key of ['depth', 'normal', 'confidence']) {
        if (Boolean(a.maps[key]) !== Boolean(b.maps[key]))
          throw new Error('Dense map availability changed');
        if (!a.maps[key]) continue;
        if (a.maps[key].length !== b.maps[key].length) throw new Error('Dense map count changed');
        for (let i = 0; i < a.maps[key].length; i++)
          comparedBytes += compareArray(a.maps[key][i], b.maps[key][i], `maps.${key}[${i}]`);
      }
      return { bitwiseIdentical: true, comparedBytes };
    }
    try {
      // Separate the first complete run of each module as cold/warm-up, with fresh ShadeDevices.
      const warm = {};
      for (const version of ['baseline', 'optimized']) {
        warm[version] = await run(version);
        cold.push({ version, ...warm[version].aggregate });
      }
      const coldComparison = compare(warm.baseline.raw, warm.optimized.raw);
      for (let round = 0; round < repetitions; round++) {
        const order = round % 2 ? ['optimized', 'baseline'] : ['baseline', 'optimized'],
          pair = {};
        for (const version of order) pair[version] = await run(version);
        const comparison = compare(pair.baseline.raw, pair.optimized.raw);
        if (
          JSON.stringify(pair.baseline.aggregate.surfaceMetrics) !==
          JSON.stringify(pair.optimized.aggregate.surfaceMetrics)
        )
          throw new Error('Dense surface metrics changed');
        rows.push({
          round: round + 1,
          order,
          comparison,
          baseline: pair.baseline.aggregate,
          optimized: pair.optimized.aggregate,
        });
      }
      if (validation.length) throw new Error('GPU validation: ' + validation.join('; '));
      return {
        adapter: { description, software, timestamps, fallback: adapter.isFallbackAdapter ?? null },
        input: {
          width,
          height,
          views: scene.views.length,
          landmarks: scene.landmarks.length,
          sha256: inputHash,
        },
        options: {
          photometricIterations,
          geometricIterations,
          fusion: 'pixel',
          boundedSubmissions: false,
          poolBytes: 0,
        },
        cold,
        coldComparison,
        pairs: rows,
        timingMs: {
          baseline: summarize(rows.map((r) => r.baseline.wallMs)),
          optimized: summarize(rows.map((r) => r.optimized.wallMs)),
        },
        medianTimeReductionFraction:
          1 -
          median(rows.map((r) => r.optimized.wallMs)) / median(rows.map((r) => r.baseline.wallMs)),
      };
    } finally {
      gpu.destroy();
    }
  },
  {
    repetitions,
    width,
    height,
    photometricIterations,
    geometricIterations,
    hardware: process.env.TYPECV_HARDWARE_GPU === '1',
  },
  { timeoutMs, serve: { '/baseline-internal.mjs': baseline } },
);
assert.deepEqual(errors, []);
const summary = {
  createdAt: new Date().toISOString(),
  revision,
  baselineHash,
  optimizedBundleHash,
  sourceHashes: hashes,
  typeshade: pkg.devDependencies.typeshade,
  protocol:
    'Complete five-view synthetic Dense pipeline, explicit equal 8/2 iteration defaults unless CLI-overridden, including view planning, uploads, photometric/geometric/filter/fusion and terminal readbacks. One complete cold/warm-up per variant is separate; measured pairs alternate order. Fresh ShadeDevice/pool=0 each run, one externally owned native GPUDevice shared for driver/pipeline warm-up.',
  timingMeaning:
    'wallMs ends when the complete Dense result returns; hash/surface comparison and device disposal are excluded. Phase boundaries use completed-reference progress; fusionAndFinalize also includes geometric-stage cleanup, map snapshot callback and final result assembly. onDepthMaps clones existing host maps equally and causes no extra GPU readbacks.',
  transferMeaning:
    'Runtime upload/read bytes include Resident.write/read for managed arrays. Application-tracked uploads alone omit direct Resident.write. Resident snapshot bytes may exceed requested prefixes. These counters exclude driver staging, textures and physical VRAM.',
  scope:
    'Synthetic supplied geometry/RGB, not a source-video SfM/Dense reconstruction or native-hardware performance claim. Software adapter identity is reported and --hardware rejects software/fallback.',
  ...result,
};
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(summary, null, 2) + '\n');
process.stdout.write(
  JSON.stringify({
    out: output,
    medianTimeReductionFraction: summary.medianTimeReductionFraction,
    timingMs: summary.timingMs,
    bitwiseIdentical: summary.pairs.every((p) => p.comparison.bitwiseIdentical),
  }) + '\n',
);
