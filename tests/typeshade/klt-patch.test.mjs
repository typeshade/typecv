import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { patchFixture, assertPatches } from './klt-patch-fixture.mjs';
import { runPatches } from './klt-patch-run.mjs';
test('affine KLT measures rotation/zoom patches and agrees with scalar f32 reference', async () => {
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const input = patchFixture(),
      r = await runPatches(api, device, input);
    assertPatches(r, input);
    for (let j = 0; j < input.cases.length; j++) {
      const c = input.cases[j],
        opts = api.kltOptionsFor(input.width, {
          levels: 2,
          halfWindow: 5,
          epsilonPx: 0.005,
          maxIterations: 30,
        }),
        ref = api.trackKlt(
          api.buildKltPyramid(input.first, input.width, input.height, 2),
          api.buildKltPyramid(c.second, input.width, input.height, 2),
          input.points,
          opts,
          c.guesses,
          c.affine,
        );
      assert.deepEqual(r.cases[j].records, Array.from(ref.records));
    }
    assert.throws(
      () =>
        api.trackKltGpu(
          device,
          input.first,
          input.cases[0].second,
          input.width,
          input.height,
          input.points,
          api.kltOptionsFor(input.width),
          { affineWarps: new Float32Array(input.points.length * 2) },
        ),
      /affineWarps/,
    );
  } finally {
    await device.dispose();
  }
});
