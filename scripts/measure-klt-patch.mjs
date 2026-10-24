import { writeFile } from 'node:fs/promises';
import { patchFixture, assertPatches } from '../tests/typeshade/klt-patch-fixture.mjs';
import { inBrowser } from '../tests/typeshade/browser.mjs';
const input = patchFixture();
const { result, errors } = await inBrowser(
  async (input) => {
    const api = await import('/dist/internal/gpu.js'),
      { runPatches } = await import('/tests/typeshade/klt-patch-run.mjs'),
      adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice(),
      device = await api.ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0 });
    try {
      return {
        ...(await runPatches(api, device, input)),
        adapter: {
          vendor: adapter.info.vendor,
          architecture: adapter.info.architecture,
          fallback: adapter.info.isFallbackAdapter,
        },
      };
    } finally {
      await device.dispose();
    }
  },
  input,
  { timeoutMs: 120000 },
);
if (errors.length) throw new Error(JSON.stringify(errors));
assertPatches(result, input);
const receipt = {
  mode: 'synthetic patch tracking with supplied affine matrices and deliberately inexact initial centres; not RGB-only SfM',
  options: {
    width: input.width,
    height: input.height,
    levels: 2,
    halfWindow: 5,
    epsilonPx: 0.005,
    maxIterations: 30,
  },
  adapter: result.adapter,
  cases: result.cases.map((r, i) => {
    const e = Array.from({ length: input.points.length / 2 }, (_, k) =>
      Math.hypot(
        r.records[4 * k] - input.cases[i].truth[2 * k],
        r.records[4 * k + 1] - input.cases[i].truth[2 * k + 1],
      ),
    ).sort((a, b) => a - b);
    return {
      warp: input.cases[i].warp,
      seeds: input.points.length / 2,
      plainAlive: r.plainAlive,
      plainMaxErrorPx: r.plainMax,
      affineAlive: e.length,
      affineMaxErrorPx: e.at(-1),
      affineMedianErrorPx: e[e.length >> 1],
    };
  }),
  liveBytes: result.liveBytes,
};
const output =
  process.argv.find((v) => v.startsWith('--output='))?.slice(9) ??
  'docs/experiments/general-video/klt-affine-patches.json';
await writeFile(output, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt));
