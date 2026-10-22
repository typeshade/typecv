import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('workgroup relative-pose cache preserves every Dense stage and final geometry bit for bit', async () => {
  const { result, errors } = await inBrowser(
    async (hardware) => {
      const api = await import('/dist/internal/gpu.js'),
        { mvsPoseCacheCases } = await import('/tests/typeshade/mvs-pose-cache-cases.mjs'),
        adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter');
      const description = [
        adapter.info?.vendor,
        adapter.info?.architecture,
        adapter.info?.device,
        adapter.info?.description,
      ]
        .filter(Boolean)
        .join(' ');
      if (
        hardware &&
        (/swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter)
      )
        throw new Error(`Hardware requested, software adapter selected: ${description}`);
      const gpu = await adapter.requestDevice(),
        validation = [];
      gpu.addEventListener('uncapturederror', (event) => validation.push(event.error.message));
      try {
        return { description, rows: await mvsPoseCacheCases(api, { gpuDevice: gpu }), validation };
      } finally {
        gpu.destroy();
      }
    },
    process.env.TYPECV_HARDWARE_GPU === '1',
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.validation, []);
  console.log(JSON.stringify(result));
  assert.equal(result.rows[0].sources, 6);
  for (const row of result.rows) {
    assert.ok(row.kept > 0, row.name);
    if ('points' in row) assert.ok(row.points > 0, row.name);
  }
  const budget = result.rows.at(-1).residency;
  assert.equal(budget.geometric.mode, 'streamed');
  assert.equal(budget.geometric.slots, 2);
  assert.ok(budget.geometric.elements < 25 * 37);
  assert.ok(budget.peakBytes <= 75_000);
});
