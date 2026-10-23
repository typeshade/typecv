import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { boundaryMapCases } from './mvs-boundary-cases.mjs';
import { inBrowser } from './browser.mjs';

test('WebGPU boundary and observability kernels preserve finite geometry across grouped tail chunks', async () => {
  const cpu = await boundaryMapCases(api, { backend: 'cpu' }, ['combined']);
  const { result, errors } = await inBrowser(
    async () => {
      const api = await import('/dist/internal/gpu.js');
      const { boundaryMapCases } = await import('/tests/typeshade/mvs-boundary-cases.mjs');
      return boundaryMapCases(api, { backend: 'webgpu' }, ['combined']);
    },
    null,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.backend, 'webgpu');
  assert.equal(result.liveBytes, 0);
  const rows = [];
  for (let j = 0; j < result.rows.length; j++) {
    const a = result.rows[j],
      b = cpu.rows[j],
      relative = [];
    for (let i = 0; i < a.depth.length; i++)
      if (a.depth[i] > 0 && b.depth[i] > 0)
        relative.push(Math.abs(a.depth[i] - b.depth[i]) / b.depth[i]);
    relative.sort((x, y) => x - y);
    assert.ok(relative.length > 300, `${a.name}: finite common support ${relative.length}`);
    const median = relative[Math.floor(relative.length / 2)];
    assert.ok(median < 0.001, `${a.name}: CPU/WebGPU median relative difference ${median}`);
    assert.ok(
      Math.abs(a.metrics.kept - b.metrics.kept) < 0.03,
      `${a.name}: CPU/WebGPU coverage mismatch`,
    );
    assert.ok(a.metrics.median < 0.04, `${a.name}: finite geometry error`);
    rows.push({
      name: a.name,
      metrics: a.metrics,
      common: relative.length,
      medianRelativeDifference: median,
    });
  }
  console.log(JSON.stringify({ boundaryWebGPU: rows }));
});
