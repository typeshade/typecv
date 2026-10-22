import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('Dense independent readbacks overlap on WebGPU and keep the same points and depth maps', async () => {
  const { result: r, errors } = await inBrowser(async () => {
    const api = await import('/dist/internal/gpu.js');
    const { readbackCase } = await import('/tests/typeshade/readback-cases.mjs');
    return readbackCase(api, 'webgpu');
  });
  assert.deepEqual(errors, []);
  assert.equal(r.liveBytes, 0);
  for (const { name, baseline, concurrent } of r.rows) {
    assert.deepEqual(concurrent.output, baseline.output, name);
    for (const key of ['reads', 'bytes', 'snapshots'])
      assert.equal(concurrent.counts[key], baseline.counts[key], name + ' ' + key);
    assert.ok(concurrent.counts.waves < baseline.counts.waves, name);
  }
  assert.ok(r.rows[2].concurrent.output.count > 0);
  console.log(
    JSON.stringify(
      r.rows.map(({ name, baseline, concurrent }) => ({
        name,
        baseline: baseline.counts,
        concurrent: concurrent.counts,
      })),
    ),
  );
});
