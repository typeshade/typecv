import test from 'node:test';
import * as api from '../../dist/gpu.js';
import {
  surfaceFixture,
  consensusFixture,
  assertSurface,
  assertConsensus,
} from './observation-surface-fixture.mjs';
import { runSurfaceCases, runConsensusCases } from './observation-surface-cases.mjs';

test('TypeShade CPU measures translated tilted-surface patches from slightly inexact map proposals', async () => {
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const input = surfaceFixture();
    console.log(
      JSON.stringify(assertSurface(await runSurfaceCases(api, device, input.surface), input)),
    );
  } finally {
    await device.dispose();
  }
});

test('TypeShade CPU corroborates additions from another original image and preserves two-view candidates', async () => {
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const input = consensusFixture();
    assertConsensus(await runConsensusCases(api, device, input), input);
  } finally {
    await device.dispose();
  }
});
