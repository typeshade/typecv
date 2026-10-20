import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { ShadeDevice, verifyImagePairGpu, estimateHomographyGpu } from '../../dist/gpu.js';
import {
  radialPairFixture,
  compactedPairFixture,
  homographyFixture,
} from './initialization-cases.mjs';

// The selection stops where a sequential RANSAC would (`selectPairModel`), so the result of a
// search is the same for every batch size: the batch is a performance choice, and a device may
// size it by its time budget without changing what the search returns.
test('a pair search returns the same result for every batch size', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu', maxModelsPerBatch: 4096 });
  try {
    for (const [label, f] of [
      ['radial', radialPairFixture({ n: 321 })],
      ['compacted', compactedPairFixture()],
    ]) {
      const results = [];
      for (const batchSize of [3, 32, 512]) {
        const r = await verifyImagePairGpu(d, f.pixels, f.first, f.second, {
          iterations: 256,
          threshold: 0.001,
          seed: 71,
          homographyIterations: 128,
          refinementIterations: 10,
          batchSize,
        });
        const { execution, ...rest } = r;
        results.push({ batchSize, rest, execution });
      }
      for (const r of results.slice(1)) {
        assert.deepEqual(r.rest, results[0].rest, `${label}: batch ${r.batchSize}`);
        assert.equal(
          r.execution.hypothesisBatches <= results[0].execution.hypothesisBatches,
          true,
          `${label}: a larger batch takes no more batches`,
        );
      }
      // The hypotheses counted are the sequential search's, the same for every batch size.
      for (const r of results.slice(1))
        assert.equal(
          r.execution.hypothesesScored,
          results[0].execution.hypothesesScored,
          `${label}: hypotheses at batch ${r.batchSize}`,
        );
      assert.ok(results[0].execution.hypothesesScored > 0);
    }
    const h = homographyFixture({});
    const homographies = [];
    for (const batchSize of [2, 16, 256])
      homographies.push(
        await estimateHomographyGpu(d, h.pixels, { iterations: 256, thresholdPx: 0.8, batchSize }),
      );
    for (const r of homographies.slice(1)) {
      assert.deepEqual([...r.inliers], [...homographies[0].inliers]);
      assert.deepEqual([...r.H], [...homographies[0].H]);
      assert.equal(r.inlierCount, homographies[0].inlierCount);
    }
  } finally {
    await d.dispose();
  }
});

/** The searched fields of a result, with typed arrays as arrays, for a digest of their bits. */
function plain(value) {
  if (ArrayBuffer.isView(value)) return Array.from(value, plain);
  if (Array.isArray(value)) return value.map(plain);
  if (value && typeof value === 'object')
    return Object.fromEntries(Object.entries(value).map(([key, v]) => [key, plain(v)]));
  return typeof value === 'number' && !Number.isFinite(value) ? String(value) : value;
}

// The batches of a search between two reads of the scheduling word (`PROGRESS_INTERVAL` of
// `src/gpu/pair-estimation.ts`).
const PROGRESS_INTERVAL = 4;
// The digest of the results the last test reads, from the loop that read the scheduling word
// after each batch (main at 15f895f, TypeShade 41a140a, Node 22.22.0). A change of it is a change
// of the search's results: find its cause before you record a new value.
const PREVIOUS_RESULTS = 'b9806b43724fb38754ddb3dd024d831d46d02ae2a8c41b0c5095d346f2f2fd35';

// The GPU writes each batch's workgroup counts (TypeShade change 0075): the host reads the
// adaptive budget after every fourth batch of a search, and nothing between the kernels of a
// batch. The results are those of the loop that read the scheduling word after each batch.
test('a pair search reads its scheduling word once per four batches, with the same results', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu', maxModelsPerBatch: 4096 }),
    read = d.read.bind(d);
  let words = 0;
  d.read = (buffer, length) => {
    if (length === 1) words++;
    return read(buffer, length);
  };
  const results = [];
  try {
    for (const [label, f] of [
      ['radial', radialPairFixture({ n: 321 })],
      ['compacted', compactedPairFixture()],
      ['planar', radialPairFixture({ n: 65, planar: true, noise: 0, outliers: false })],
    ])
      for (const batchSize of [3, 32, 512]) {
        words = 0;
        const r = await verifyImagePairGpu(d, f.pixels, f.first, f.second, {
          iterations: 256,
          threshold: 0.001,
          seed: 71,
          homographyIterations: 128,
          refinementIterations: 10,
          batchSize,
        });
        const calls = r.execution.report.groups.flatMap((group) => group.calls);
        // Each read follows exactly four batches of its search: a search starts at
        // `initializePair`, and each batch of it at `planPairModelBatch`.
        let batches = 0,
          snapshots = 0;
        for (const call of calls) {
          if (call === 'initializePair') batches = 0;
          else if (call === 'planPairModelBatch') batches++;
          else if (call === 'snapshotPairModelBudget') {
            assert.equal(
              batches,
              PROGRESS_INTERVAL,
              `${label} ${batchSize}: batches before a read`,
            );
            batches = 0;
            snapshots++;
          }
        }
        assert.equal(words, snapshots, `${label} ${batchSize}: one word read per snapshot`);
        assert.ok(
          snapshots * PROGRESS_INTERVAL <= calls.filter((c) => c === 'planPairModelBatch').length,
        );
        const { execution, ...rest } = r;
        results.push({
          label,
          batchSize,
          rest,
          scored: execution.hypothesesScored,
          batches: execution.hypothesisBatches,
          homography: execution.homographyIterations,
        });
      }
    // Before the change, the batch loop of 32 trials read 20 words for the radial pair, 17 for
    // the compacted pair and 24 for the planar pair (main at 15f895f); a batch of 512 trials, of
    // which the CPU tier scores 64 at a time, read 14, 12 and 16.
    const digest = createHash('sha256')
      .update(JSON.stringify(plain(results)))
      .digest('hex');
    // The results are the bits of those of the loop before the change.
    assert.equal(digest, PREVIOUS_RESULTS);
  } finally {
    d.read = read;
    await d.dispose();
  }
});
