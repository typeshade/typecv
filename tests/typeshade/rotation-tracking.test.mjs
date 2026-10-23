import test from 'node:test';
import * as api from '../../dist/gpu.js';
import { runRotationTrackingCases } from './rotation-tracking-cases.mjs';
import { assertTracking, rotationFixtures } from './rotation-tracking-fixture.mjs';

test('TypeShade CPU session preserves rotation-only evidence, resumes geometry and isolates snapshots', async () => {
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    assertTracking(await runRotationTrackingCases(api, device, rotationFixtures()));
  } finally {
    await device.dispose();
  }
});
