import test from 'node:test';
import * as api from '../../dist/gpu.js';
import {
  runDepthPublicationCase,
  assertDepthPublication,
  runDepthSessionPublicationCase,
  assertDepthSessionPublication,
} from './depth-publication-cases.mjs';

for (const mode of ['direct', 'refine', 'float'])
  test(`TypeShade CPU ${mode} mapper publishes admitted depth once and nothing on failure`, async () => {
    const result = await runDepthPublicationCase(api, 'cpu', mode);
    assertDepthPublication(result);
    console.log(JSON.stringify(result));
  });

test('TypeShade CPU session preserves persistent ids and archived observation revocation', async () => {
  const result = await runDepthSessionPublicationCase(api, 'cpu');
  assertDepthSessionPublication(result);
  console.log(JSON.stringify(result));
});
