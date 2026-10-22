import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { buildGuidedProbe } from '../tests/typeshade/guided-probe-build.mjs';
import { guidedWorkgroupFixture } from '../tests/typeshade/workgroup-cases.mjs';

const { values } = parseArgs({
  options: {
    hardware: { type: 'boolean', default: false },
    queries: { type: 'string', default: '257,1025' },
    candidates: { type: 'string', default: '1025' },
    samples: { type: 'string', default: '7' },
    warmup: { type: 'string', default: '2' },
    fixture: { type: 'string' },
  },
});
const queries = values.queries.split(',').map(Number),
  candidates = Number(values.candidates),
  samples = Number(values.samples),
  warmup = Number(values.warmup);
if (
  [...queries, candidates, samples, warmup].some((n) => !Number.isSafeInteger(n) || n < 1) ||
  queries.some((n) => n > 16384) ||
  candidates > 16384 ||
  samples > 100 ||
  warmup > 20
)
  throw new RangeError(
    'Positive counts required: queries/candidates <= 16384, samples <= 100, warmup <= 20',
  );
const fixtures = values.fixture
  ? [JSON.parse(await readFile(values.fixture, 'utf8'))]
  : queries.flatMap((n) =>
      [2, 1e4].map((band) => guidedWorkgroupFixture(n, candidates, 128, band)),
    );
if (values.hardware) process.env.TYPECV_HARDWARE_GPU = '1';
const hardware = process.env.TYPECV_HARDWARE_GPU === '1';
const source = await readFile(
  new URL('../src/gpu/kernels/float-matching.shade.ts', import.meta.url),
);
const packageInfo = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
const file = await buildGuidedProbe();
const { result, errors } = await inBrowser(
  async (input) => {
    const { guidedWorkgroupBenchmark } = await import('/probe/guided-probe.js');
    return guidedWorkgroupBenchmark(input);
  },
  { fixtures, samples, warmup, hardware },
  { timeoutMs: 1800000, serve: { '/probe/guided-probe.js': file } },
);
assert.deepEqual(errors, []);
console.log(
  JSON.stringify(
    {
      methodology:
        'Resident SfM guided kernel; pipeline compilation and transfers excluded. GPU timestamps cover the compute pass; wall time covers encoding, submission and queue completion. Variant order rotates/reverses per round.',
      productionDefault: 64,
      kernelSha256: createHash('sha256').update(source).digest('hex'),
      typeshade: packageInfo.devDependencies.typeshade,
      hardwareRequested: hardware,
      ...result,
    },
    null,
    2,
  ),
);
