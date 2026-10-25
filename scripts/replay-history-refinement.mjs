// Run the production archive finalizer on an exported Lab map. Private input/output stay local.
// node scripts/replay-history-refinement.mjs --input=/private/map.json --out=/private/replay
import { readFile, writeFile, mkdir, mkdtemp } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { build } from 'esbuild';
import { projectPoint } from '../src/geometry/pose.ts';
import { focalGrid, focalSearchControls } from '../src/reconstruction/focal-search.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const arg = (name, fallback) =>
  process.argv.find((v) => v.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
if (!arg('input') || !arg('out')) throw new Error('--input and --out required');
const inputPath = resolve(arg('input')),
  out = resolve(arg('out')),
  rawBytes = await readFile(inputPath),
  raw = JSON.parse(rawBytes),
  inputSha256 = createHash('sha256').update(rawBytes).digest('hex'),
  grid = focalGrid(focalSearchControls(true)),
  latest = raw.views.find((v) => v.id === raw.diagnostics.registrationOrder[0]),
  currentRatio = latest.camera.fx / (2 * latest.camera.cx + 1),
  ratios = arg('ratios')?.split(',').map(Number) ?? [
    currentRatio,
    grid[Math.floor((grid.length - 1) / 2)],
    grid.at(-1),
  ],
  iterations = Number(arg('iterations', '32'));
if (
  !Number.isSafeInteger(iterations) ||
  iterations < 0 ||
  iterations > 100 ||
  ratios.some((r) => !Number.isFinite(r) || r <= 0)
)
  throw new RangeError('Invalid replay controls');
const byView = new Map(raw.views.map((v) => [v.id, v])),
  conflicts = [];
const residual = (point, o) => {
  const view = byView.get(o.viewId),
    pixel = projectPoint(view.camera, view.pose, point.xyz);
  return pixel ? Math.hypot(pixel[0] - o.pixel[0], pixel[1] - o.pixel[1]) : Infinity;
};
// A historical export may contain duplicate-camera tracks. Select once using original
// geometry, retain the audit, and evaluate the result on *all original observations* too.
const model = {
  views: raw.views,
  landmarks: raw.landmarks.map((point) => {
    const observations = new Map();
    for (const o of point.observations) {
      const old = observations.get(o.viewId);
      if (!old) observations.set(o.viewId, o);
      else {
        const keepNew = residual(point, o) < residual(point, old);
        conflicts.push({
          trackId: point.trackId,
          viewId: o.viewId,
          keptFeatureId: keepNew ? o.featureId : old.featureId,
          excludedFeatureId: keepNew ? old.featureId : o.featureId,
        });
        if (keepNew) observations.set(o.viewId, o);
      }
    }
    return { ...point, observations: [...observations.values()] };
  }),
};
await mkdir(out, { recursive: true });
const controlDirectory = await mkdtemp(join(tmpdir(), 'typecv-history-replay-')),
  control = join(controlDirectory, 'history.mjs');
await build({
  entryPoints: [resolve('src/reconstruction/history-refinement.ts')],
  outfile: control,
  bundle: true,
  format: 'esm',
  platform: 'browser',
});
const { result, errors } = await inBrowser(
  async ({ model: raw, iterations, ratios }) => {
    const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js'),
      { refineHistory } = await import('/history.mjs'),
      model = {
        views: raw.views.map((v) => ({
          ...v,
          pose: { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) },
        })),
        landmarks: raw.landmarks.map((p) => ({ ...p, xyz: Float64Array.from(p.xyz) })),
      },
      adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter required');
    const gpu = await adapter.requestDevice(),
      gpuErrors = [],
      info = {
        vendor: adapter.info.vendor,
        architecture: adapter.info.architecture,
        description: adapter.info.description,
        fallback: adapter.info.isFallbackAdapter,
      };
    gpu.addEventListener('uncapturederror', (e) => gpuErrors.push(e.error.message));
    const device = await ShadeDevice.open({
        gpuDevice: gpu,
        fp64Flavor: 'float',
        boundedSubmissions: false,
        poolBytes: 0,
      }),
      solves = [],
      started = performance.now();
    let completed;
    try {
      completed = await refineHistory(
        model,
        { iterations, refineSharedFocal: true, commonFocalRatios: ratios, huberPx: 1.5 },
        async (input, options) => {
          const before = performance.now(),
            solved = await bundleAdjustGpu(device, input, options);
          solves.push({
            views: input.views.length,
            points: input.landmarks.length,
            observations: solved.diagnostics.observations,
            diagnostics: solved.diagnostics,
            elapsedMs: performance.now() - before,
            execution: solved.execution,
          });
          return solved;
        },
      );
      return {
        ...JSON.parse(
          JSON.stringify(completed, (_k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v)),
        ),
        info,
        solves,
        elapsedMs: performance.now() - started,
        workingBytes: device.workingBytes,
        statistics: device.statistics,
        gpuErrors,
      };
    } finally {
      await device.dispose();
    }
  },
  { model, iterations, ratios },
  { timeoutMs: 1800000, serve: { '/history.mjs': control } },
);
const finalByView = new Map(result.reconstruction.views.map((v) => [v.id, v])),
  finalByPoint = new Map(result.reconstruction.landmarks.map((p) => [p.trackId, p]));
let total = 0,
  count = 0,
  above3 = 0,
  behind = 0;
for (const point of raw.landmarks)
  for (const o of point.observations) {
    const view = finalByView.get(o.viewId),
      current = finalByPoint.get(point.trackId),
      pixel = projectPoint(view.camera, view.pose, current.xyz);
    if (!pixel) {
      behind++;
      continue;
    }
    const error = Math.hypot(pixel[0] - o.pixel[0], pixel[1] - o.pixel[1]);
    total += error;
    count++;
    above3 += +(error > 3);
  }
await writeFile(
  join(out, 'model.json'),
  JSON.stringify({
    ...raw,
    views: result.reconstruction.views,
    landmarks: result.reconstruction.landmarks,
  }),
);
const summary = {
  inputPath,
  inputSha256,
  iterations,
  ratios,
  conflicts,
  ...result,
  reconstruction: undefined,
  allOriginalObservationCheck: { observations: count, behind, meanErrorPx: total / count, above3 },
  pageErrors: errors,
  unchangedInput:
    createHash('sha256')
      .update(await readFile(inputPath))
      .digest('hex') === inputSha256,
  limitation: 'Software WebGPU replay; no native speed or ground-truth shape claim.',
};
await writeFile(join(out, 'result.json'), JSON.stringify(summary, null, 2));
console.log(
  JSON.stringify({
    output: out,
    inputSha256,
    before: result.before,
    after: result.after,
    allOriginal: summary.allOriginalObservationCheck,
    info: result.info,
    solves: result.solves.length,
    errors,
    gpuErrors: result.gpuErrors,
    unchangedInput: summary.unchangedInput,
  }),
);
