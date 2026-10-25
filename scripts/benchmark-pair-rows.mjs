import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    candidate: { type: 'string', default: 'dist/internal/gpu.js' },
    output: { type: 'string', default: '.typecv-build/pair-row-benchmark.json' },
    samples: { type: 'string', default: '5' },
    sizes: { type: 'string', default: '513,2049' },
  },
});
if (!values.baseline)
  throw new Error(
    'Usage: node scripts/benchmark-pair-rows.mjs --baseline OLD_BUNDLE [--candidate NEW_BUNDLE] [--output FILE]',
  );
const paths = { baseline: resolve(values.baseline), candidate: resolve(values.candidate) };
const sha = (data) => createHash('sha256').update(data).digest('hex');
const samples = Number(values.samples);
const sizes = values.sizes.split(',').map(Number);
if (
  !Number.isInteger(samples) ||
  samples < 1 ||
  samples > 9 ||
  sizes.some((n) => !Number.isInteger(n) || n < 8 || n > 4096)
)
  throw new Error('Expected 1..9 samples and fixture sizes 8..4096');

const output = await inBrowser(
  async ({ samples, sizes }) => {
    const old = await import('/probe/baseline-internal.js');
    const next = await import('/probe/in-place-internal.js');
    const { radialPairFixture } = await import('/tests/typeshade/initialization-cases.mjs');
    const variants = [old, next];
    const devices = await Promise.all(
      variants.map((v) =>
        v.ShadeDevice.open({
          poolBytes: 0,
          occupancyFloors: true,
          maxModelsPerBatch: 256,
        }),
      ),
    );
    const fixtures = sizes.map((n) => [`radial${n}`, radialPairFixture({ n })]);
    function plain(value) {
      if (ArrayBuffer.isView(value)) return Array.from(value);
      if (Array.isArray(value)) return value.map(plain);
      if (value && typeof value === 'object')
        return Object.fromEntries(
          Object.entries(value)
            .filter(([key]) => key !== 'execution')
            .map(([key, v]) => [key, plain(v)]),
        );
      return value;
    }
    const rows = [];
    try {
      for (const [name, f] of fixtures) {
        const options = {
          iterations: 256,
          minIterations: 64,
          homographyIterations: 128,
          threshold: 0.001,
          seed: 71,
          refinementIterations: 10,
          batchSize: 256,
        };
        const results = [[], []],
          wallMs = [[], []],
          executions = [[], []];
        for (let v = 0; v < variants.length; v++)
          await variants[v].verifyImagePairGpu(devices[v], f.pixels, f.first, f.second, options);
        for (let sample = 0; sample < samples; sample++) {
          for (const v of sample % 2 ? [1, 0] : [0, 1]) {
            const start = performance.now();
            const r = await variants[v].verifyImagePairGpu(
              devices[v],
              f.pixels,
              f.first,
              f.second,
              options,
            );
            wallMs[v].push(performance.now() - start);
            results[v].push(plain(r));
            executions[v].push({
              dispatches: r.execution.report.dispatches,
              submissions: r.execution.report.submissions,
              hypotheses: r.execution.hypothesesScored,
              gpuNs: r.execution.report.groups.reduce((s, g) => s + (g.gpuTime ?? 0), 0),
            });
          }
        }
        rows.push({ name, wallMs, results, executions });
      }
      return { rows, statistics: devices.map((d) => d.statistics) };
    } finally {
      for (const d of devices) await d.dispose();
    }
  },
  { samples, sizes },
  {
    timeoutMs: 180000,
    serve: {
      '/probe/baseline-internal.js': paths.baseline,
      '/probe/in-place-internal.js': paths.candidate,
    },
  },
);
assert.deepEqual(output.errors, []);
for (const row of output.result.rows) {
  for (const v of row.results) for (const r of v) assert.deepEqual(r, row.results[0][0], row.name);
}
const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
const receipt = {
  scope:
    "Chromium WebGPU; software SwiftShader unless TYPECV_HARDWARE_GPU=1; not the user's Turing GPU",
  deviceOptions: { occupancyFloors: true, maxModelsPerBatch: 256, poolBytes: 0 },
  warmup: 1,
  samples,
  order: 'alternate old/new and new/old',
  baselineSha256: sha(await readFile(paths.baseline)),
  candidateSha256: sha(await readFile(paths.candidate)),
  fixtureSourceSha256: sha(
    await readFile(new URL('../tests/typeshade/initialization-cases.mjs', import.meta.url)),
  ),
  rows: output.result.rows.map(({ name, wallMs, executions, results }) => ({
    name,
    exact: true,
    terminalSha256: sha(JSON.stringify(results[0][0])),
    wallMs,
    medianWallMs: wallMs.map(median),
    executions,
  })),
  statistics: output.result.statistics,
};
await writeFile(values.output, JSON.stringify(receipt, null, 2) + '\n');
console.log(
  JSON.stringify({ rows: receipt.rows, liveBuffers: receipt.statistics.map((s) => s.liveBuffers) }),
);
