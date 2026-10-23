import assert from 'node:assert/strict';
import { generalVideoFixture } from '../fixtures/general-video.ts';
import { reconstructIncremental, ReconstructionInitializationError } from '../../src/index.ts';

export function rotationFixtures() {
  const pack = (scenario) => {
    const f = generalVideoFixture(scenario);
    return {
      images: f.input.images.map((v) => ({ ...v, keypoints: [...v.keypoints] })),
      pairs: f.input.pairs
        .filter((p) => scenario !== 'rotation-start' || p.second - p.first === 1)
        .map((p) => ({ ...p, matches: [...p.matches] })),
    };
  };
  return {
    prefix: pack('rotation-start'),
    unseen: pack('mapped-rotation'),
    reseen: pack('reobserved'),
  };
}

export function assertTracking(result) {
  assert.equal(result.isolated, true);
  assert.equal(result.windowSafe, true);
  assert.deepEqual(result.transfer[0], result.transfer[1]);
  assert.equal(result.transfer[0].failure, 'ReconstructionInitializationError');
  assert.equal(result.resultIsolated, true);
  assert.equal(result.waitingFinish.name, 'ReconstructionInitializationError');
  assert.equal(result.waitingFinish.fabricatedMap, false);
  assert.equal(result.waitingFinish.tracking.tracks, 60);
  assert.equal(result.waitingFinish.tracking.pending, 60);
  assert.equal(result.checkpoints[2].registered, 0);
  assert.equal(result.checkpoints[2].tracking.rotations.length, 2);
  const f = generalVideoFixture('rotation-start');
  let oracle;
  try {
    reconstructIncremental(
      f.input.images.slice(0, 3),
      f.input.pairs.filter((p) => p.second < 3 && p.second - p.first === 1),
      { ransacIterations: 128, rotationTracking: true },
    );
  } catch (error) {
    assert.ok(error instanceof ReconstructionInitializationError);
    oracle = error.tracking;
  }
  for (const [i, pair] of result.checkpoints[2].tracking.rotations.entries()) {
    assert.deepEqual(pair.R, [...oracle.rotations[i].rotation.R]);
    assert.equal(pair.translation, 'unobservable');
    assert.equal(pair.depth, 'not-estimated');
  }
  assert.equal(result.final.registered, 5);
  assert.equal(result.final.points, 60);
  assert.equal(result.final.tracking.pending, 0);
  assert.equal(result.finishPairVerifications, 0);
  assert.ok(result.final.cpuStages.some((s) => s.includes('rotation hypotheses (f64 CPU')));
  assert.equal(result.cases[0].points, 40);
  assert.equal(result.cases[0].tracking.pending, 20);
  assert.ok(
    result.cases[0].pendingObservations.every((obs) =>
      obs.every((o) => o.featureId >= 40 && (o.viewId === 2 || o.viewId === 3)),
    ),
  );
  assert.equal(result.cases[1].points, 60);
  assert.equal(result.cases[1].tracking.pending, 0);
  assert.equal(result.liveBuffers, 0);
}
