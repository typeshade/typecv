import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { assertInlineVotes } from './tsdf-inline-votes-assertions.mjs';

test('inline visibility resets match the standalone oracle across early returns on WebGPU', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const api = await import('/dist/internal/gpu.js');
      const { inlineVotesCase } = await import('/tests/typeshade/tsdf-inline-votes-cases.mjs');
      return inlineVotesCase(api, 'webgpu');
    },
    undefined,
    { timeoutMs: 60000 },
  );
  assert.deepEqual(errors, []);
  assertInlineVotes(result);
});
