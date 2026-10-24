import test from 'node:test';
import * as api from '../../dist/internal/gpu.js';
import { residencyFixture, assertResidency } from './observation-residency-fixture.mjs';
import { runResidencyCases } from './observation-residency-cases.mjs';

test('default and opt-in observation residency preserve exact evidence with zero and production pooling', async () => {
  const input = residencyFixture();
  for (const poolBytes of [0, 32 * 1024 ** 2])
    assertResidency(await runResidencyCases(api, input, { poolBytes }), input);
});

test('observation residency respects a two-pyramid budget and preserves caller-owned buffers', async () => {
  const input = residencyFixture();
  for (const poolBytes of [0, 32 * 1024 ** 2])
    assertResidency(await runResidencyCases(api, input, { poolBytes, tightBudget: true }), input, {
      tightBudget: true,
    });
});
