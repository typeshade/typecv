import assert from 'node:assert/strict';
import {
  directionObservationFixture,
  directionPairPixels,
} from '../fixtures/direction-observations.ts';
import { reconstructTwoView } from '../../src/sfm/two-view.ts';

export function directionFixtures() {
  const pack = (options) => {
    const f = directionObservationFixture(options);
    return {
      images: f.input.images.map((v) => ({ ...v, keypoints: [...v.keypoints] })),
      pairs: f.input.pairs.map((p) => ({ ...p, matches: [...p.matches] })),
    };
  };
  const pairCases = [];
  for (const directionalCount of [30, 60, 180])
    for (const noisePx of [0, 0.2]) {
      const f = directionObservationFixture({ directionalCount, noisePx }),
        pixels = directionPairPixels(f),
        first = f.input.images[0].camera,
        second = f.input.images[4].camera,
        options = {
          iterations: 128,
          refinementIterations: noisePx ? 5 : 0,
          separateDirectionalObservations: true,
        },
        oracle = reconstructTwoView(pixels, first, second, options);
      pairCases.push({
        directionalCount,
        noisePx,
        pixels: [...pixels],
        first,
        second,
        options,
        oracle: { R: [...oracle.R], t: [...oracle.t] },
      });
    }
  return {
    pairCases,
    mixed: pack({ noisePx: 0.2 }),
    pure: pack({ finiteCount: 0 }),
    far: pack({ distantDepth: 100 }),
  };
}

export function assertDirectionCases(result, fixtures) {
  assert.equal(result.liveBuffers, 0);
  for (const [index, pair] of result.pairs.entries()) {
    const f = fixtures.pairCases[index];
    assert.equal(pair.points, 60);
    assert.equal(pair.directional, f.directionalCount);
    assert.equal(pair.directionConsensus, f.directionalCount);
    assert.equal(pair.falseFinite, 0);
    assert.equal(pair.unresolvedAreNaN, true);
    assert.ok(pair.medianAngleDeg > 1);
    const cosine = pair.R.reduce((s, x, i) => s + x * f.oracle.R[i], 0);
    assert.ok(Math.acos(Math.max(-1, Math.min(1, (cosine - 1) / 2))) < 0.001);
    assert.ok(pair.t.reduce((s, x, i) => s + x * f.oracle.t[i], 0) > 0.999);
    assert.deepEqual(pair.cpuStages, []);
  }
  assert.equal(result.mixed.views, 5);
  assert.equal(result.mixed.points, 60);
  assert.equal(result.mixed.pending, 60);
  assert.equal(result.mixed.falseFinite, 0);
  assert.equal(result.pure.name, 'ReconstructionInitializationError');
  assert.equal(result.pure.pending, 60);
  assert.equal(result.pure.fabricatedMap, false);
  assert.equal(result.far.prefixPoints, 60);
  assert.equal(result.far.prefixPending, 60);
  assert.equal(result.far.points, 120);
  assert.equal(result.far.pending, 0);
}
