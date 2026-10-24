import test from 'node:test';
import * as api from '../../dist/internal/gpu.js';
import { workFixture, workReferences, assertWork } from './klt-work-fixture.mjs';
import { runWork } from './klt-work-cases.mjs';

test('KLT work elision preserves exhaustive f32 positions, NCC, convergence and ended records', async () => {
  const input = workFixture(),
    references = workReferences(api, input),
    device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    assertWork(await runWork(api, device, input), input, references);
  } finally {
    await device.dispose();
  }
});
