import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('wide descriptor slabs preserve tiled nearest-neighbor bits and boundary behavior', async () => {
  const { result, errors } = await inBrowser(async () => {
    const api = await import('/dist/internal/gpu.js');
    const { floatMatchingWideCases } =
      await import('/tests/typeshade/float-matching-wide-cases.mjs');
    return floatMatchingWideCases(api);
  });
  assert.deepEqual(errors, []);
  console.log(JSON.stringify(result));
});
