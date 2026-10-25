import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import {
  assertDepthPublication,
  assertDepthSessionPublication,
} from './depth-publication-cases.mjs';

for (const mode of ['direct', 'refine', 'float'])
  test(`WebGPU ${mode} mapper publishes admitted depth once and nothing on failure`, async () => {
    const { result, errors } = await inBrowser(
      async (mode) => {
        const api = await import('/dist/gpu.js'),
          { runDepthPublicationCase } =
            await import('/tests/typeshade/depth-publication-cases.mjs');
        return runDepthPublicationCase(api, 'webgpu', mode);
      },
      mode,
      { timeoutMs: 600000 },
    );
    assert.deepEqual(errors, []);
    assertDepthPublication(result);
    console.log(JSON.stringify(result));
  });

test('WebGPU session preserves persistent ids and archived observation revocation', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const api = await import('/dist/gpu.js'),
        { runDepthSessionPublicationCase } =
          await import('/tests/typeshade/depth-publication-cases.mjs');
      return runDepthSessionPublicationCase(api, 'webgpu');
    },
    null,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assertDepthSessionPublication(result);
  console.log(JSON.stringify(result));
});
