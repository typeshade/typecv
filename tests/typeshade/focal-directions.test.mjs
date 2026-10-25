import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/gpu.js';
import { directionObservationFixture } from '../fixtures/direction-observations.ts';
import { compareSeedGauge } from '../../scripts/geometry-comparison.mjs';
import { runFocalDirectionCase } from './focal-directions-cases.mjs';

test('TypeShade CPU focal calibration preserves unresolved directions and geometric accuracy', async () => {
  const f = directionObservationFixture({ noisePx: 0.2 });
  for (const v of f.input.images) v.camera = { ...v.camera, fx: 512, fy: 512 };
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const r = await runFocalDirectionCase(api, device, f.input),
      q = compareSeedGauge(
        { views: f.truth.views, points: f.truth.finitePoints },
        r.model,
        r.initialPair,
      );
    assert.equal(r.model.views.length, 5);
    assert.equal(r.model.landmarks.length, 60);
    assert.ok(r.model.landmarks.every((p) => p.observations.every((o) => o.featureId < 60)));
    assert.equal(r.pending, 60);
    assert.equal(r.focalSearch.scoring, 'directional-bic');
    assert.ok(r.model.views.every((v) => Math.abs(v.camera.fx / 480 - 1) < 0.02));
    assert.ok(q.maxRotationErrorDeg < 0.2);
    assert.ok(q.pointRmse < 0.1);
    assert.ok(r.cpuStages.some((s) => s.includes('bearing seed/selection')));
    assert.equal(r.liveBuffers, 0);
    assert.equal(r.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});
