import test from 'node:test';
import assert from 'node:assert/strict';
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
