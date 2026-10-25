import test from 'node:test';
import * as api from '../../dist/internal/gpu.js';
import { inlineVotesCase } from './tsdf-inline-votes-cases.mjs';
import { assertInlineVotes } from './tsdf-inline-votes-assertions.mjs';

test('inline visibility resets match the standalone oracle across early returns on CPU', async () => {
  assertInlineVotes(await inlineVotesCase(api, 'cpu'));
});
