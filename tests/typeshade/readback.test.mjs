import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { readbackCase } from './readback-cases.mjs';

export function checkReadbacks(r) {
  assert.equal(r.liveBytes, 0);
  for (const { name, baseline, concurrent } of r.rows) {
    assert.deepEqual(concurrent.output, baseline.output, name);
    for (const key of ['reads', 'bytes', 'snapshots'])
      assert.equal(concurrent.counts[key], baseline.counts[key], name + ' ' + key);
    assert.ok(concurrent.counts.waves < baseline.counts.waves, name);
  }
  assert.ok(r.rows[2].concurrent.output.count > 0);
}

test('Dense independent readbacks overlap without changing arrays or transfer bytes on CPU', async () => {
  const r = await readbackCase(api, 'cpu');
  checkReadbacks(r);
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
