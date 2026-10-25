import test from 'node:test';
import * as api from '../../dist/internal/gpu.js';
import { stagingCase } from './tsdf-staging-cases.mjs';
import { assertStagingCase } from './tsdf-staging-assertions.mjs';

test('TSDF bounded staging and reused discovery keys preserve all output channels on CPU', async () => {
  assertStagingCase(await stagingCase(api, 'cpu'));
});
