// Replay BA admission on an exported partial map, without rerunning RGB features or poses.
// node scripts/replay-window-bundle.mjs --input=/private/reconstruction.json \
//   --next-id=124 --output=/tmp/window-replay.json
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { build } from 'esbuild';
import { cameraCenter } from '../src/geometry/pose.ts';
import { planConnectedBundles } from '../src/reconstruction/local-bundle.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const arg = (name, fallback) =>
  process.argv.find((v) => v.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
if (!arg('input')) throw new Error('--input=<typecv-reconstruction.json> required');
const bytes = await readFile(resolve(arg('input'))),
  exported = JSON.parse(bytes);
if (exported.format !== 'typecv-reconstruction-v1') throw new Error('Unsupported input format');
const model = {
    views: exported.views.map(({ id, camera, pose }) => ({
      id,
      camera,
      pose: { R: Float64Array.from(pose.R), t: Float64Array.from(pose.t) },
    })),
    landmarks: exported.landmarks.map(({ trackId, xyz, observations }) => ({
      trackId,
      xyz: Float64Array.from(xyz),
      observations,
    })),
  },
  nextId = Number(arg('next-id', String(Math.max(...model.views.map((v) => v.id)) + 1))),
  size = Number(arg('window', '32')),
  iterations = Number(arg('iterations', '4'));
if (
  !Number.isSafeInteger(nextId) ||
  model.views.some((v) => v.id === nextId) ||
  !Number.isSafeInteger(size) ||
  size < 8 ||
  size > 64 ||
  !Number.isSafeInteger(iterations) ||
  iterations < 1 ||
  iterations > 100
)
  throw new RangeError('Invalid next id, window size or iterations');

// The session's chronological input order differs from the registration order in views.
const chronological = exported.analysisFrames
  .filter((f) => f.metrics?.features > 0)
  .map((f) => f.id)
  .concat(nextId);
if (chronological.length <= size) throw new Error('Input does not roll the requested window');
const recent = new Set(chronological.slice(-(size - 2)));
let anchors = [model.views[0].id, model.views[1].id],
  baseline = 0;
const shared = new Map();
for (const p of model.landmarks) {
  const ids = p.observations.map((o) => o.viewId).sort((a, b) => a - b);
  for (let i = 0; i < ids.length; i++)
    for (let j = i + 1; j < ids.length; j++) {
      const key = `${ids[i]}:${ids[j]}`;
      shared.set(key, (shared.get(key) ?? 0) + 1);
    }
}
const candidates = model.views.filter((v) => recent.has(v.id));
for (let i = 0; i < candidates.length; i++)
  for (let j = i + 1; j < candidates.length; j++) {
    const a = candidates[i],
      b = candidates[j],
      common = shared.get(a.id < b.id ? `${a.id}:${b.id}` : `${b.id}:${a.id}`) ?? 0,
      ac = cameraCenter(a.pose),
      bc = cameraCenter(b.pose),
      distance = Math.hypot(...ac.map((x, k) => x - bc[k]));
    if (distance > baseline && distance > 1e-8 && common >= 12) {
      anchors = [a.id, b.id];
      baseline = distance;
    }
  }
const keep = new Set([...recent, ...anchors]),
  reduced = {
    views: model.views.filter((v) => keep.has(v.id)),
    landmarks: model.landmarks
      .map((p) => ({
        ...p,
        observations: p.observations.filter((o) => keep.has(o.viewId)),
      }))
      .filter((p) => p.observations.length >= 2),
  };
const directory = await mkdtemp(join(tmpdir(), 'typecv-window-replay-'));
try {
  const control = join(directory, 'control.mjs');
  await build({
    entryPoints: [resolve('src/reconstruction/local-bundle.ts')],
    outfile: control,
    bundle: true,
    platform: 'browser',
    format: 'esm',
  });
  const { result, errors } = await inBrowser(
    async ({ raw, anchors, iterations }) => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js'),
        { planConnectedBundles, mergeBundleRefinements } = await import('/control.mjs');
      const model = {
        views: raw.views.map((v) => ({
          ...v,
          pose: { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) },
        })),
        landmarks: raw.landmarks.map((p) => ({ ...p, xyz: Float64Array.from(p.xyz) })),
      };
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('WebGPU adapter unavailable');
      const gpu = await adapter.requestDevice(),
        gpuErrors = [];
      gpu.addEventListener('uncapturederror', (e) => gpuErrors.push(e.error.message));
      const device = await ShadeDevice.open({
          gpuDevice: gpu,
          fp64Flavor: 'float',
          boundedSubmissions: false,
          poolBytes: 0,
        }),
        options = {
          iterations,
          anchorViewId: anchors[0],
          scaleViewId: anchors[1],
          refineSharedFocal: true,
          huberPx: 1.5,
        };
      try {
        const before = device.statistics;
        let baselineError;
        try {
          await bundleAdjustGpu(device, model, options);
        } catch (error) {
          baselineError = error.message;
        }
        const plan = planConnectedBundles(model, options),
          costs = [],
          refined = [],
          start = performance.now();
        for (const part of plan.problems) {
          const solved = await bundleAdjustGpu(device, part.reconstruction, part.options);
          costs.push(solved.diagnostics);
          refined.push(solved.reconstruction);
        }
        const elapsedMs = performance.now() - start,
          merged = mergeBundleRefinements(model, refined),
          after = device.statistics;
        return {
          adapter: {
            vendor: adapter.info.vendor,
            architecture: adapter.info.architecture,
            description: adapter.info.description,
            device: adapter.info.device,
          },
          backend: device.backend,
          fp64Flavor: device.fp64Flavor,
          baselineError,
          partition: plan.partition,
          costs,
          views: merged.views.length,
          landmarks: merged.landmarks.length,
          sameObservationMembership:
            JSON.stringify(model.landmarks.map((p) => p.observations)) ===
            JSON.stringify(merged.landmarks.map((p) => p.observations)),
          fixedPosesPreserved: plan.partition.fixedViewIds.every(
            (id) =>
              JSON.stringify(model.views.find((v) => v.id === id).pose) ===
              JSON.stringify(merged.views.find((v) => v.id === id).pose),
          ),
          gpuErrors,
          elapsedMs,
          uploads: after.applicationUploadBytes - before.applicationUploadBytes,
          readbacks: after.applicationReadbackBytes - before.applicationReadbackBytes,
          liveBytesAfter: after.liveBytes,
        };
      } finally {
        await device.dispose();
      }
    },
    {
      raw: JSON.parse(JSON.stringify(reduced, (_k, v) => (ArrayBuffer.isView(v) ? [...v] : v))),
      anchors,
      iterations,
    },
    { serve: { '/control.mjs': control }, timeoutMs: 600000 },
  );
  const all = model.landmarks.flatMap((p) => p.observations),
    heights = new Map(exported.views.map((v) => [v.id, v.height])),
    resultFile = {
      inputSha256: createHash('sha256').update(bytes).digest('hex'),
      sourceSiteCommit: exported.runRecord?.site?.commit,
      scope:
        'Exported-map BA admission replay; no RGB frontend, re-verification or 33rd-frame registration',
      truth: 'No ground-truth geometry; residual cost is an optimizer check only',
      original: {
        views: model.views.length,
        landmarks: model.landmarks.length,
        observations: all.length,
        ...(model.views.every((v) => Number.isFinite(heights.get(v.id)))
          ? {
              topQuarterObservations: all.filter((o) => o.pixel[1] < heights.get(o.viewId) / 4)
                .length,
            }
          : {}),
        partition: planConnectedBundles(model, {}).partition,
      },
      window: {
        size,
        nextId,
        anchors,
        views: reduced.views.length,
        landmarks: reduced.landmarks.length,
      },
      result,
      browserErrors: errors,
    };
  if (
    errors.length ||
    result.gpuErrors.length ||
    !result.baselineError ||
    !result.sameObservationMembership ||
    !result.fixedPosesPreserved ||
    result.liveBytesAfter
  )
    throw new Error('Replay validation failed');
  await writeFile(
    resolve(arg('output', '/tmp/typecv-window-replay.json')),
    JSON.stringify(resultFile, null, 2) + '\n',
  );
  console.log(JSON.stringify(resultFile, null, 2));
} finally {
  await rm(directory, { recursive: true, force: true });
}
