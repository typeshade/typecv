import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { buildGuidedProbe } from './guided-probe-build.mjs';
import { guidedWorkgroupFixture } from './workgroup-cases.mjs';

test('SfM guided workgroup overrides preserve raw results, tail queries and cached pipelines', async () => {
  const file = await buildGuidedProbe();
  const fixtures = [
    guidedWorkgroupFixture(257, 67, 33),
    guidedWorkgroupFixture(65, 67, 128, 1e4),
    // All candidates tie, including the ratio-test boundary; one candidate is nonfinite.
    {
      ...guidedWorkgroupFixture(9, 5, 1),
      queries: Array(9).fill(0),
      candidates: [0, 0, 1, 2, Infinity],
    },
    // A sole candidate, an empty band, and a degenerate epipolar line.
    guidedWorkgroupFixture(3, 1, 17),
    { ...guidedWorkgroupFixture(3, 2, 1), candidatePoints: [0, 100, 0, 200], maxLinePx: 0.1 },
    { ...guidedWorkgroupFixture(3, 2, 1), matrix: Array(9).fill(0) },
  ];
  const { result, errors } = await inBrowser(
    async (input) => {
      const { guidedWorkgroupBenchmark } = await import('/probe/guided-probe.js');
      return guidedWorkgroupBenchmark(input);
    },
    { fixtures, samples: 2, warmup: 1, hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
    { timeoutMs: 120000, serve: { '/probe/guided-probe.js': file } },
  );
  assert.deepEqual(errors, []);
  for (const row of result.results) {
    assert.equal(row.bitExact, true);
    assert.deepEqual(row.skipped, []);
    assert.deepEqual(
      row.variants.map((v) => v.size),
      [64, 128, 256],
    );
    for (const variant of row.variants) {
      assert.equal(variant.cached, true);
      assert.deepEqual(variant.shape, [variant.size, 1, 1]);
      assert.equal(variant.groups, Math.ceil(row.queries / variant.size));
      assert.equal(variant.wallMs.length, 2);
      assert.ok(variant.wallMs.every((ms) => Number.isFinite(ms) && ms >= 0));
      assert.equal(variant.gpuMs.length, row.timestampQuery ? 2 : 0);
      assert.ok(variant.gpuMs.every((ms) => Number.isFinite(ms) && ms >= 0));
    }
  }
});
