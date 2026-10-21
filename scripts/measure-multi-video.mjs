// Measures the multi-video pipeline on a device (docs/multi-video.md, Amendment 8): the stage
// times, the calls of the mapper, and a summary of the result that two builds can compare.
//
//   npm run build
//   node scripts/measure-multi-video.mjs [--paths 3] [--views 12] [--points 300] [--out file.json]
//     [--bundle other/internal/gpu.js]   measure another build of the internal entry
//     [--shuffle 5]                      shuffle the input order with this seed
//
// The scene is synthetic (`tests/view-graph-scene.ts`). The run is headless Chromium. It uses
// SwiftShader unless TYPECV_HARDWARE_GPU=1.
import { writeFile } from 'node:fs/promises';
import { pathScene } from '../tests/view-graph-scene.ts';
import { quantizeDescriptors } from '../src/reconstruction/view-graph.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const paths = Number(flag('paths', 3)),
  views = Number(flag('views', 12)),
  points = Number(flag('points', 300)),
  out = flag('out', ''),
  shuffle = Number(flag('shuffle', 0)),
  bundle = flag('bundle', '');

const specs = [];
for (let p = 0; p < paths; p++)
  specs.push({
    label: `path${p}`,
    cloud: p < 2 ? 0 : p - 1,
    views,
    from: p % 2 ? 5 : -35,
    to: p % 2 ? 40 : 30,
    radius: 9,
    height: -1 - (p % 2) * 2,
  });
const scene = pathScene(specs, { points });
const b64 = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('base64');
let input = scene.keyframes.map(({ viewId: _id, descriptors, ...k }) => ({
  videoId: k.videoId,
  frameKey: k.frameKey,
  time: k.time,
  camera: k.camera,
  keypoints: b64(k.keypoints),
  descriptors: b64(quantizeDescriptors(descriptors)),
  global: b64(k.global),
}));
if (shuffle) {
  let s = shuffle;
  for (let i = input.length - 1; i > 0; i--) {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    const j = s % (i + 1);
    [input[i], input[j]] = [input[j], input[i]];
  }
}
console.error(`scene: ${input.length} keyframes, ${points} points per cloud`);

const { result, errors } = await inBrowser(
  async ({ input }) => {
    const api = await import('/dist/internal/gpu.js');
    const dec = (s, T) => {
      const bin = atob(s),
        u8 = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      return new T(u8.buffer);
    };
    const keyframes = input.map((k) => ({
      ...k,
      keypoints: dec(k.keypoints, Float64Array),
      descriptors: dec(k.descriptors, Uint8Array),
      global: dec(k.global, Float32Array),
    }));
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const gpu = await adapter.requestDevice();
    const device = await api.ShadeDevice.open({ gpuDevice: gpu, solverCheckpoints: true });
    // The options of the lab's worker (`website/src/workers/multi-video.ts`).
    const MAPPER = {
      seed: 7,
      ransacIterations: 500,
      maxErrorPx: 3,
      minInitialAngleDeg: 1,
      minTriangulationAngleDeg: 0.5,
      bundleIterations: 4,
      finalBundleIterations: 8,
      refineSharedFocal: true,
    };
    let lastReads = 0;
    // Round trips: `plan` calls are submissions, `read` calls are readbacks.
    const counters = { plans: 0 };
    const plan = device.plan.bind(device);
    device.plan = (...args) => {
      counters.plans++;
      return plan(...args);
    };
    const calls = {},
      stages = [];
    const started = performance.now();
    const events = [];
    try {
      const r = await api.reconstructMultiVideoGpu(device, keyframes, {
        graph: { verification: MAPPER },
        mapper: {
          ...MAPPER,
          onStage: (s) => {
            const reads = device.statistics.reads ?? 0;
            const readsDelta = reads - lastReads;
            lastReads = reads;
            const c = (calls[s.operation] ??= {
              n: 0,
              ms: 0,
              dispatches: 0,
              submissions: 0,
              reads: 0,
            });
            c.reads += readsDelta;
            c.n++;
            stages.push([
              s.operation,
              Math.round(s.elapsedMs),
              s.execution.report?.submissions ?? 0,
              s.problem?.views ?? 0,
              s.problem?.landmarks ?? 0,
              s.problem?.observations ?? 0,
            ]);
            c.ms += s.elapsedMs;
            c.dispatches += s.execution.report?.dispatches ?? 0;
            c.submissions += s.execution.report?.submissions ?? 0;
          },
        },
        onProgress: (s) =>
          events.push([
            s,
            performance.now() - started,
            counters.plans,
            device.statistics.reads ?? 0,
          ]),
      });
      const keyOf = new Map(r.keyframes.map((k) => [k.viewId, k.frameKey]));
      return {
        totalMs: performance.now() - started,
        graphTiming: r.graph.timing,
        reconstructionMs: r.timing.reconstructionMs,
        stats: r.graph.stats,
        edges: r.graph.edges.map((e) => [
          keyOf.get(e.first),
          keyOf.get(e.second),
          e.inliers,
          e.homographyInliers,
        ]),
        components: r.components.map((c) => ({
          size: c.viewIds.length,
          registered: c.reconstruction?.views.length ?? 0,
          error: c.meanReprojectionErrorPx,
          landmarks: c.reconstruction?.landmarks.length ?? 0,
          order: (c.registrationOrder ?? []).map((v) => keyOf.get(v)),
          ms: c.ms,
          failure: c.error,
        })),
        calls,
        stages,
        events,
      };
    } finally {
      await device.dispose();
    }
  },
  { input },
  { timeoutMs: 3600000, serve: bundle ? { '/dist/internal/gpu.js': bundle } : {} },
);
if (errors.length) console.error(errors);
const s = (v) => (v / 1000).toFixed(2);
console.log(
  `graph: matching ${s(result.graphTiming.pairMatchingMs)} s, verification ${s(result.graphTiming.pairVerificationMs)} s, ` +
    `${result.stats.verifiedPairs} pairs, ${result.stats.edges} edges`,
);
{
  const [, , plans0, reads0] = result.events.find((e) => e[0] === 'viewGraph') ?? [0, 0, 0, 0];
  const next = result.events.find((e) => e[0].startsWith('component'));
  if (next)
    console.log(
      `graph round trips: ${next[2] - plans0} plans, ${next[3] - reads0} reads, ` +
        `${(result.stats.verifiedPairs ? (next[2] - plans0) / result.stats.candidatePairs : 0).toFixed(1)} plans per candidate pair`,
    );
}
console.log(`reconstruction: ${s(result.reconstructionMs)} s; total ${s(result.totalMs)} s`);
for (const c of result.components)
  console.log(
    `component: ${c.registered}/${c.size} registered, ${c.landmarks} landmarks, ${c.error?.toFixed(3)} px, ${s(c.ms)} s ${c.failure ?? ''}`,
  );
for (const [k, v] of Object.entries(result.calls))
  console.log(
    `  ${k}: ${v.n} calls, ${s(v.ms)} s, ${v.dispatches} dispatches, ${v.submissions} submissions, ${v.reads} reads`,
  );
if (out) await writeFile(out, JSON.stringify(result));
