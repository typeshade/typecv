import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { observationPathCases } from './observation-path-cases.mjs';
import { mvsScene } from './mvs-cases.mjs';

test('explicit optical paths constrain Dense sources, maps and planar completion on CPU', async () => {
  const result = await observationPathCases(api, 'cpu');
  assert.equal(result.liveBytes, 0);
  assert.equal(result.filled, 18);
  console.log(JSON.stringify(result));
});

test('invalid optical-path masks fail before allocating device arrays', () => {
  const scene = mvsScene({ width: 25, height: 19 }),
    plan = api.planMvsViews(scene),
    [id, image] = [...scene.images][0];
  for (const mask of [
    { width: 25, height: 19, source: 'manual', values: new Uint8Array(1) },
    { width: 25, height: 19, source: 'manual', values: new Uint8Array(475).fill(4) },
    { width: 19, height: 25, source: 'manual', values: new Uint8Array(475) },
  ]) {
    const images = new Map(scene.images);
    images.set(id, { ...image, observationMask: mask });
    assert.throws(() => api.mvsDepthMapsGpu({}, scene, plan, images), RangeError);
  }
});
