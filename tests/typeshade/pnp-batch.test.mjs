import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { ShadeDevice, estimatePosePnPGpu } from '../../dist/gpu.js';
import { poseFixture } from './geometry-cases.mjs';

/** The searched fields of a result, with typed arrays as arrays, for a digest of their bits. */
function plain(value) {
  if (ArrayBuffer.isView(value)) return Array.from(value, plain);
  if (Array.isArray(value)) return value.map(plain);
  if (value && typeof value === 'object')
    return Object.fromEntries(Object.entries(value).map(([key, v]) => [key, plain(v)]));
  return typeof value === 'number' && !Number.isFinite(value) ? String(value) : value;
}

const fixtures = [
  ['noisy', {}],
  ['planar', { count: 41, planar: true, noise: 0, outliers: 0 }],
  ['four', { count: 4, noise: 0, outliers: 0 }],
  // 60 % outliers: the budget the best consensus sets, 106 and 119 trials, is not a multiple of
  // a batch, so the selection stops inside a batch.
  ['outliers', { count: 200, outliers: 0.6 }],
];
// The mapper's settings (750 trials, at least 64) and the CPU test's (128 trials, 12 steps).
const searches = [
  ['mapper', { iterations: 750, seed: 17 }],
  ['short', { iterations: 128, seed: 91, refinementIterations: 12 }],
];
// At least one trial: the budget falls inside the first batch of 32 trials.
const early = [['early', { iterations: 500, minIterations: 1, seed: 5, refinementIterations: 4 }]];
// 4, the batch of 32 trials the search used before, and the most trials of a batch.
const BATCH_SIZES = [4, 32, 512];
// The batches of a search between two reads of the scheduling word (`PROGRESS_INTERVAL` of
// `src/gpu/pnp.ts`).
const PROGRESS_INTERVAL = 4;
// The digests of the results the last test reads (the base 909aa54, TypeShade d70b539, Node
// 22.22.0, the CPU tier). `PREVIOUS_RESULTS` is the search of the mapper's and the short settings
// in batches of 32 trials, the default then, which selected the best candidate of a whole batch
// and set the budget once per batch: on these fixtures the budget changed only at a batch's end,
// so that search is the sequential one. `SEQUENTIAL_RESULTS` is the early search of that build in
// batches of one trial, the sequential search: its batches of 32 trials ran past the budget the
// first trials set, so they counted 32 trials and could select a later candidate. A change of a
// digest is a change of the search's results: find its cause before you record a new value.
const PREVIOUS_RESULTS = '33d6fc826ee9e7cb3c113577bbf48813332312a072dc8994d7ee05d7a593ac76';
const SEQUENTIAL_RESULTS = '219212ea6095d32b28b6c4f1688fa81d194e71b60f098417f5ecb78744044470';

async function search(device, settings, options, batchSize) {
  const f = poseFixture(settings);
  return estimatePosePnPGpu(device, f.world, f.pixels, f.camera, { ...options, batchSize });
}

// The selection stops where a sequential RANSAC would (`selectPnP`), so the result of a search is
// the same for every batch size: the batch is a performance choice, and a device may size it by
// its time budget without changing what the search returns.
test('a PnP search returns the same result for every batch size', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu', maxModelsPerBatch: 4096 });
  try {
    for (const [label, settings] of fixtures)
      for (const [name, options] of [...searches, ...early]) {
        const results = [];
        for (const batchSize of BATCH_SIZES) {
          const { execution, hypothesisBatches, ...rest } = await search(
            d,
            settings,
            options,
            batchSize,
          );
          results.push({ batchSize, rest, hypothesisBatches, execution });
        }
        for (const r of results.slice(1)) {
          assert.deepEqual(r.rest, results[0].rest, `${label} ${name}: batch ${r.batchSize}`);
          assert.ok(
            r.hypothesisBatches <= results[0].hypothesisBatches,
            `${label} ${name}: a larger batch takes no more batches`,
          );
        }
        // The trials and the candidates counted are the sequential search's.
        assert.ok(results[0].rest.iterations >= 1 && results[0].rest.candidatesEvaluated > 0);
        assert.ok(results[0].hypothesisBatches >= Math.ceil(results[0].rest.iterations / 4));
      }
  } finally {
    await d.dispose();
  }
});

// The GPU writes each batch's workgroup counts (TypeShade change 0075): the host reads the
// adaptive budget after every fourth batch of a search, and nothing between the kernels of a
// batch. The results are those of the base build.
test('a PnP search reads its scheduling word once per four batches, with the same results', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu', maxModelsPerBatch: 4096 }),
    read = d.read.bind(d);
  let words = 0;
  d.read = (buffer, length) => {
    if (length === 1) words++;
    return read(buffer, length);
  };
  const previous = [],
    sequential = [];
  try {
    for (const [label, settings] of fixtures)
      for (const [name, options] of [...searches, ...early])
        for (const batchSize of BATCH_SIZES) {
          words = 0;
          const r = await search(d, settings, options, batchSize);
          const calls = r.execution.report.groups.flatMap((group) => group.calls);
          // Each read follows exactly four batches: a search starts at `initializePnP`, and each
          // batch of it at `planPnPBatch`.
          let batches = 0,
            snapshots = 0;
          for (const call of calls) {
            if (call === 'initializePnP') batches = 0;
            else if (call === 'planPnPBatch') batches++;
            else if (call === 'snapshotPnPBudget') {
              assert.equal(batches, PROGRESS_INTERVAL, `${label} ${name} ${batchSize}: batches`);
              batches = 0;
              snapshots++;
            }
          }
          const submitted = calls.filter((c) => c === 'planPnPBatch').length;
          assert.equal(words, snapshots, `${label} ${name} ${batchSize}: one word per snapshot`);
          assert.ok(snapshots * PROGRESS_INTERVAL <= submitted);
          // The GPU selects from the batches below the budget; at most three past it are
          // submitted before the host reads the word that stops the loop.
          assert.ok(r.hypothesisBatches <= submitted);
          assert.ok(submitted - r.hypothesisBatches < PROGRESS_INTERVAL);
          // The final pose and summary, the nine counters, the flags and the words read.
          assert.equal(
            r.execution.applicationReadbackBytes,
            164 + 4 * r.inliers.length + 4 * snapshots,
          );
          const { execution, hypothesisBatches, ...rest } = r;
          if (batchSize === 32)
            (name === 'early' ? sequential : previous).push({ label, name, rest });
        }
    // The batch of 512 trials, of which the CPU tier scores 64 at a time, read one word for each
    // search of 750 trials (the budget, 64 to 119 trials, is met in the first four batches) and
    // none for a search of 128 trials (two batches); a batch of 4 trials reads one word per 16
    // trials until the budget is met.
    const digest = (results) =>
      createHash('sha256')
        .update(JSON.stringify(plain(results)))
        .digest('hex');
    assert.equal(digest(previous), PREVIOUS_RESULTS);
    assert.equal(digest(sequential), SEQUENTIAL_RESULTS);
  } finally {
    d.read = read;
    await d.dispose();
  }
});

// A fresh WebGPU device would make the pipeline of `scorePnP` inside its first timed scoring (581
// ms on the lab's Blackwell device, against the work's few ms). The step carries a warm-up that
// dispatches `scorePnP` over zero workgroups in an untimed plan of its own before that scoring.
// The CPU tier cannot run `scorePnP` (it has a barrier). So this test sets the backend name to
// `webgpu`, swaps the body of each `scorePnP` step and of its warm-up for a record, and reads the
// order of the records. The search that follows has no scores, and the test reads only the order.
test('a PnP search warms scorePnP in an untimed plan before its first timed scoring, once', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu', maxModelsPerBatch: 4096 }),
    log = [],
    planSteps = d.planSteps.bind(d);
  d.backend = 'webgpu';
  const plan = d.plan.bind(d);
  d.plan = (body, chunk) => {
    log.push(chunk ? `plan ${chunk.key}` : 'plan');
    // `plan` checks that the report names the backend; the CPU tier ran the body.
    d.backend = 'cpu';
    return plan(body, chunk).finally(() => {
      d.backend = 'webgpu';
    });
  };
  d.planSteps = (steps) =>
    planSteps(
      (function* () {
        for (const [name, run, extra, chunk] of steps) {
          if (name !== 'scorePnP') yield [name, run, extra, chunk];
          else
            yield [
              name,
              () => log.push('score'),
              extra,
              {
                ...chunk,
                warm: { ...chunk.warm, run: () => log.push(`warm ${chunk.warm.id}`) },
              },
            ];
        }
      })(),
    );
  try {
    const f = poseFixture({});
    const options = { iterations: 128, seed: 91, batchSize: 32 };
    for (const run of [0, 1]) {
      log.length = 0;
      await estimatePosePnPGpu(d, f.world, f.pixels, f.camera, options).catch(() => undefined);
      const marks = log.filter((entry) => entry === 'score' || entry.startsWith('warm'));
      assert.ok(marks.filter((entry) => entry === 'score').length >= 1);
      // The first search warms once, before the first scoring. The second search warms nothing.
      assert.deepEqual(
        marks.filter((entry) => entry.startsWith('warm')),
        run === 0 ? ['warm pnp'] : [],
      );
      if (run === 0) assert.equal(marks[0], 'warm pnp');
      // The warm-up is a plan with no chunk, and the first timed scoring follows it.
      if (run === 0) {
        const at = log.indexOf('warm pnp');
        assert.equal(log[at - 1], 'plan');
        assert.equal(log[at + 1], 'plan scorePnP');
      }
    }
  } finally {
    await d.dispose();
  }
});
