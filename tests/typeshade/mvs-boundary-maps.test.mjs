import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { mvsScene } from './mvs-cases.mjs';
import { boundaryMapCases } from './mvs-boundary-cases.mjs';

test('boundary/observability maps agree with grouped, tail-chunked CPU processing without leaking arrays', async () => {
  const r = await boundaryMapCases(api, { backend: 'cpu' });
  assert.equal(r.liveBytes, 0);
  for (const row of r.rows) {
    assert.ok(
      row.metrics.kept > 0.2,
      `${row.name}: insufficient finite coverage ${row.metrics.kept}`,
    );
    assert.ok(row.metrics.median < 0.04, `${row.name}: finite depth error ${row.metrics.median}`);
  }
  console.log(JSON.stringify({ boundaryMaps: r.rows.map(({ depth, normal, cost, ...r }) => r) }));
});

test('MVS boundary and finite-gain options reject invalid values before allocating GPU arrays', () => {
  const scene = mvsScene({ width: 25, height: 19 }),
    plan = api.planMvsViews(scene);
  for (const options of [
    { boundaryAware: 'on' },
    { minFiniteGain: -0.1 },
    { minFiniteGain: NaN },
    { minFiniteGain: Infinity },
    { minFiniteGain: 2.1 },
  ])
    assert.throws(() => api.mvsDepthMapsGpu({}, scene, plan, scene.images, options), RangeError);
});
