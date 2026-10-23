import assert from 'node:assert/strict';
import { generalVideoFixture } from '../fixtures/general-video.ts';
import { foregroundFixture } from '../fixtures/foreground.ts';
import { compareSeedGauge } from '../../scripts/geometry-comparison.mjs';

const pack = (f) => ({
  images: f.images.map((v) => ({ ...v, keypoints: [...v.keypoints] })),
  pairs: f.pairs.map((p) => ({ ...p, matches: [...p.matches] })),
});
export function structureQualityFixtures() {
  const f = generalVideoFixture('translation'),
    fg = foregroundFixture(),
    noisy = generalVideoFixture('rotation-start');
  const n = {
    images: noisy.input.images.map((v) => ({
      ...v,
      keypoints: v.keypoints.map((x, i) => x + 0.7 * Math.sin((i + 1) * (v.id + 3) * 1.37)),
    })),
    pairs: noisy.input.pairs.filter((p) => p.second - p.first === 1),
  };
  return {
    translation: pack(f.input),
    foreground: { ...pack(fg), masks: fg.masks.map((m) => ({ ...m, values: [...m.values] })) },
    noisy: pack(n),
  };
}
export function assertStructureQuality(r) {
  assert.deepEqual(r.status, [5, 0]);
  assert.equal(r.rejectedHasGeometry, false);
  assert.deepEqual(r.postMasks, [
    [0, 0],
    [1, 1],
  ]);
  assert.deepEqual(r.depthRejected, [1, 0]);
  assert.equal(r.extraReadbackBytes, 8);
  assert.equal(r.readCalls[0], r.readCalls[1]);
  assert.equal(r.sessionPending.depth, 60);
  assert.equal(r.sessionPending.tracks, 60);
  assert.equal(r.sessionFinal.registered, 5);
  assert.equal(r.sessionFinal.points, 60);
  assert.deepEqual(r.sessionFinal.depth, []);
  assert.equal(r.foreground.registered, 5);
  assert.equal(r.foreground.points, 60);
  assert.equal(r.foreground.sessionPoints, 60);
  assert.equal(r.foreground.selectionIsolated, true);
  assert.equal(r.foreground.backgroundOnly, true);
  assert.equal(r.foreground.separated, 2400);
  assert.equal(r.foreground.inputRows, 3000);
  assert.equal(r.foreground.source, 'ground-truth');
  assert.equal(r.noisy.registered, 5);
  assert.equal(r.noisy.model.landmarks.length, 60);
  const quality = compareSeedGauge(
    generalVideoFixture('rotation-start').truth,
    r.noisy.model,
    r.noisy.initialPair,
  );
  assert.ok(quality.maxRotationErrorDeg < 0.7, JSON.stringify(quality));
  assert.ok(quality.pointRmse < 0.25, JSON.stringify(quality));
  assert.equal(r.liveBuffers, 0);
}
