import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { buildComputeProbe } from './compute-probe-build.mjs';

test('WebGPU in-place pair QR preserves factors, masked row windows and tail models', async () => {
  const file = await buildComputeProbe('pair-row-probe');
  const { result, errors } = await inBrowser(
    async () => {
      const { checkPairRows, checkPairRowWindows } = await import('/probe/pair-row-probe.js');
      const results = [];
      for (const fp64Flavor of ['float'])
        results.push({
          factors: await checkPairRows({ backend: 'webgpu', fp64Flavor }),
          windows: await checkPairRowWindows({ backend: 'webgpu', fp64Flavor }),
        });
      return results;
    },
    null,
    { timeoutMs: 120000, serve: { '/probe/pair-row-probe.js': file } },
  );
  assert.deepEqual(errors, []);
  for (const flavor of result) {
    assert.equal(flavor.factors.backend, 'webgpu');
    for (const row of flavor.factors.rows) {
      assert.equal(row.exact, true, JSON.stringify(row.input));
      assert.deepEqual(row.second.bits, row.first.bits);
      const limit = row.input.refit === false ? (row.input.kind === 0 ? 8 : 4) : row.input.rows;
      assert.equal(row.first.dispatches, Math.ceil(limit / 128));
      assert.equal(row.second.dispatches, row.first.dispatches);
    }
    assert.equal(flavor.factors.statistics.liveBuffers, 0);
    assert.equal(flavor.factors.statistics.liveBytes, 0);
    for (const row of flavor.windows.rows) {
      assert.deepEqual(row.second.bits, row.first.bits);
      assert.equal(row.first.dispatches, Math.ceil(row.input.rows / 128));
      assert.equal(row.second.dispatches, Math.ceil(row.input.rows / 512));
    }
    assert.equal(flavor.windows.statistics.liveBuffers, 0);
  }
});

test('pair refits honor the requested latency policy with identical homography and mask bits', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, estimateHomographyGpu } = await import('/dist/gpu.js');
      const { homographyFixture } = await import('/tests/typeshade/initialization-cases.mjs');
      const rows = [];
      const adapter = await navigator.gpu.requestAdapter();
      const gpu = await adapter.requestDevice();
      try {
        for (const policy of [
          { points: 1024, timeBudgetMs: 250, boundedSubmissions: false, expected: 6, old: 1 },
          { points: 1024, timeBudgetMs: 249, boundedSubmissions: false, expected: 0, old: 25 },
          { points: 1024, timeBudgetMs: 250, boundedSubmissions: true, expected: 0, old: 25 },
          { points: 1023, timeBudgetMs: 250, boundedSubmissions: false, expected: 0, old: 25 },
        ]) {
          const f = homographyFixture({ n: policy.points });
          const d = await ShadeDevice.open({
            ...policy,
            gpuDevice: gpu,
            fp64Flavor: 'float',
            maxDispatchesPerSubmission: 64,
            poolBytes: 0,
          });
          try {
            const r = await estimateHomographyGpu(d, f.pixels, {
              iterations: 16,
              minIterations: 16,
              batchSize: 16,
            });
            if (!r) throw new Error('Homography fixture failed');
            const calls = r.execution.report.groups.flatMap((g) => g.calls);
            rows.push({
              expected: policy.expected,
              old: policy.old,
              points: policy.points,
              appends: calls.filter((c) => c === 'appendPairModelRowsInPlace').length,
              searches: calls.filter((c) => c === 'appendPairModelRows').length,
              H: Array.from(new Uint32Array(r.H.buffer)),
              inliers: Array.from(r.inliers),
              liveBuffers: d.statistics.liveBuffers,
            });
          } finally {
            await d.dispose();
          }
        }
      } finally {
        gpu.destroy();
      }
      return rows;
    },
    null,
    { timeoutMs: 90000 },
  );
  assert.deepEqual(errors, []);
  for (const row of result) {
    assert.equal(row.appends, row.expected);
    assert.equal(row.searches, row.old);
    assert.equal(row.liveBuffers, 0);
    if (row.points === 1024) {
      assert.deepEqual(row.H, result[0].H);
      assert.deepEqual(row.inliers, result[0].inliers);
    }
  }
});
