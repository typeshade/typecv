// Measures what one pair verification costs on a device, alone and in shared plans (docs/
// multi-video.md, Amendment 8): wall time, time inside `plan()`, dispatches, plans, read round trips.
//
//   npm run build
//   node scripts/measure-pair-cost.mjs [--pairs 24] [--matches 600] [--bundle other/internal/gpu.js]
//     [--batch 64]                 the model batch size of the search (default: the device's)
//     [--profile out.cpuprofile]   write a CPU profile of the page (see scripts/top-self-time.mjs)
import { inBrowser } from '../tests/typeshade/browser.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const pairs = Number(flag('pairs', 24)),
  matches = Number(flag('matches', 600)),
  bundle = flag('bundle', ''),
  profileOut = flag('profile', ''),
  batch = Number(flag('batch', 0));

const { result, errors, profile } = await inBrowser(
  async ({ pairs, matches, batch }) => {
    const api = await import('/dist/internal/gpu.js');
    const cases = await import('/tests/typeshade/initialization-cases.mjs');
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const gpu = await adapter.requestDevice();
    const device = await api.ShadeDevice.open({ gpuDevice: gpu, solverCheckpoints: true });
    const counters = { plans: 0, planMs: 0, dispatches: 0 };
    const plan = device.plan.bind(device);
    device.plan = async (...a) => {
      counters.plans++;
      const t = performance.now();
      const report = await plan(...a);
      counters.planMs += performance.now() - t;
      counters.dispatches += report.dispatches ?? 0;
      return report;
    };
    const requests = Array.from({ length: pairs }, (_, i) => {
      const f = cases.radialPairFixture({ n: matches + (i % 5) * 7 });
      return {
        matches: f.pixels,
        first: f.first,
        second: f.second,
        options: {
          iterations: 500,
          seed: 7 + i,
          threshold: 3 / 850,
          refinementIterations: 0,
          minParallaxDeg: 0.1,
          maxReprojectionErrorPx: 3,
          homographyThresholdPx: 3,
          homographyIterations: 500,
          ...(batch ? { batchSize: batch } : {}),
        },
      };
    });
    const snapshot = () => ({
      ms: performance.now(),
      plans: counters.plans,
      planMs: counters.planMs,
      dispatches: counters.dispatches,
      waves: device.statistics.readWaves ?? 0,
      reads: device.statistics.reads ?? 0,
    });
    const delta = (a, b, n) => ({
      wallMsPerPair: (b.ms - a.ms) / n,
      planMsPerPair: (b.planMs - a.planMs) / n,
      plansPerPair: (b.plans - a.plans) / n,
      dispatchesPerPair: (b.dispatches - a.dispatches) / n,
      readWavesPerPair: (b.waves - a.waves) / n,
      readsPerPair: (b.reads - a.reads) / n,
    });
    const out = {};
    try {
      // Warm the pipelines.
      await api.verifyImagePairGpu(
        device,
        requests[0].matches,
        requests[0].first,
        requests[0].second,
        requests[0].options,
      );
      let a = snapshot();
      const alone = [];
      for (const r of requests)
        alone.push(await api.verifyImagePairGpu(device, r.matches, r.first, r.second, r.options));
      let b = snapshot();
      out.alone = delta(a, b, pairs);
      out.inliers = alone.map((r) => r.inlierRows.length);
      if (api.verifyImagePairsGpu) {
        for (const k of [1, 4, 16]) {
          a = snapshot();
          const many = await api.verifyImagePairsGpu(device, requests, k);
          b = snapshot();
          out['shared' + k] = delta(a, b, pairs);
          out['same' + k] = many.every(
            (r, i) => r.inlierRows.length === alone[i].inlierRows.length,
          );
        }
      }
    } finally {
      await device.dispose();
    }
    return out;
  },
  { pairs, matches, batch },
  {
    timeoutMs: 3600000,
    serve: bundle ? { '/dist/internal/gpu.js': bundle } : {},
    profile: Boolean(profileOut),
  },
);
if (errors.length) console.error(errors);
if (profile) {
  const { writeFile } = await import('node:fs/promises');
  await writeFile(profileOut, JSON.stringify(profile));
}
for (const [k, v] of Object.entries(result))
  console.log(
    k,
    typeof v === 'object' && !Array.isArray(v)
      ? Object.entries(v)
          .map(([a, b]) => `${a}=${Number(b).toFixed(2)}`)
          .join('  ')
      : Array.isArray(v)
        ? v.slice(0, 8).join(',') + '…'
        : v,
  );
