import test from 'node:test';
import * as api from '../../dist/internal/gpu.js';
import {
  roundtripFixture,
  roundtripReferences,
  assertRoundtrips,
} from './klt-roundtrip-fixture.mjs';
import { runRoundtrips } from './klt-roundtrip-cases.mjs';

test('opt-in grouped KLT roundtrips preserve f32 tracking and the separate-pass default', async () => {
  const input = roundtripFixture(),
    reference = roundtripReferences(api, input),
    device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    assertRoundtrips(await runRoundtrips(api, device, input), input, reference);
  } finally {
    await device.dispose();
  }
});
