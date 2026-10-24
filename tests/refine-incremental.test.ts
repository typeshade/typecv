import test from 'node:test';
import assert from 'node:assert/strict';
import { reconstructIncremental, refineIncremental, InverseDepthTracker } from '../src/index.ts';
import { cameraCenter } from '../src/geometry/pose.ts';

import { refinementFixture } from './fixtures/refinement.ts';

test('batch refinement retains the map gauge, adds observations to registered views and triangulates new tracks', () => {
  const f = refinementFixture(),
    initial = reconstructIncremental(f.limited.inputs, f.limited.pairs, f.controls),
    snapshot = structuredClone(initial.reconstruction),
    refined = refineIncremental(f.inputs, f.pairs, initial, f.controls);
  assert.equal(initial.reconstruction.views.length, 4);
  assert.equal(initial.reconstruction.landmarks.length, 30);
  assert.equal(
    initial.reconstruction.landmarks.reduce((n, p) => n + p.observations.length, 0),
    110,
  );
  assert.equal(refined.reconstruction.views.length, 4);
  assert.equal(refined.reconstruction.landmarks.length, 60);
  assert.equal(
    refined.reconstruction.landmarks.reduce((n, p) => n + p.observations.length, 0),
    240,
  );
  assert.deepEqual(refined.diagnostics.initialPair, initial.diagnostics.initialPair);
  for (const id of refined.diagnostics.initialPair) {
    const a = refined.reconstruction.views.find((v) => v.id === id)!.pose,
      b = snapshot.views.find((v) => v.id === id)!.pose;
    assert.ok(a.R.every((x, i) => Math.abs(x - b.R[i]!) < 1e-10));
    assert.ok(a.t.every((x, i) => Math.abs(x - b.t[i]!) < 1e-10));
  }
  assert.deepEqual(initial.reconstruction, snapshot, 'accepted input map remains unchanged');
  assert.ok(refined.diagnostics.meanReprojectionErrorPx < 1e-5);
});

test('batch refinement rejects changed measurement indices, missing views and a new gauge', () => {
  const f = refinementFixture(),
    initial = reconstructIncremental(f.limited.inputs, f.limited.pairs, f.controls);
  const altered = f.inputs.map((v) => ({ ...v, keypoints: v.keypoints.map((x) => x + 0.1) }));
  assert.throws(() => refineIncremental(altered, f.pairs, initial), /stable feature indices/);
  assert.throws(
    () => refineIncremental(f.inputs, f.pairs, initial, { initialPair: [10, 30] }),
    /seed gauge/,
  );
  assert.throws(
    () =>
      refineIncremental(
        f.inputs.slice(0, 3),
        f.pairs.filter((p) => p.second !== 40),
        initial,
      ),
    /retain all/,
  );
});

test('measured refinement extends persistent raw tracks without renumbering, duplicate evidence or stale covariance', () => {
  const f = refinementFixture(),
    tracker = new InverseDepthTracker({ pixelNoiseStdDevPx: 0.25 }),
    callbacks: number[] = [],
    options = {
      ...f.controls,
      depthTracking: tracker,
      onReconstruction: (m: { landmarks: readonly unknown[] }) =>
        callbacks.push(m.landmarks.length),
    },
    initial = reconstructIncremental(f.limited.inputs, f.limited.pairs, options),
    original = structuredClone(initial),
    before = tracker.snapshot()!;
  assert.equal(before.diagnostics.uniqueObservations, 110);
  assert.equal(before.diagnostics.finite, 30);
  callbacks.length = 0;
  const refined = refineIncremental(f.inputs, f.pairs, initial, options),
    after = refined.inverseDepth!;
  assert.equal(after.diagnostics.uniqueObservations, 240);
  assert.equal(after.diagnostics.finite, 60);
  assert.equal(refined.reconstruction.landmarks.length, 60);
  assert.deepEqual(callbacks, [60], 'publish the completed admitted result once');
  assert.deepEqual(initial, original, 'refinement never mutates the accepted input');
  for (const old of before.tracks) {
    const retained = after.tracks.find((t) => t.id === old.id)!;
    assert.ok(retained, 'retained observation ownership keeps its track id');
    for (const o of old.observations)
      assert.ok(
        retained.observations.some((n) => n.viewId === o.viewId && n.featureId === o.featureId),
      );
  }
  for (const point of refined.reconstruction.landmarks) {
    const track = after.tracks.find((t) => t.id === point.trackId)!;
    assert.equal(track.status, 'finite');
    assert.ok(track.covariance && track.covariance[8]! > 0);
  }
  const anchorId = initial.diagnostics.initialPair[0],
    anchor = (m: typeof initial.reconstruction) => m.views.find((v) => v.id === anchorId)!.pose;
  assert.ok(
    anchor(refined.reconstruction).R.every(
      (v, i) => Math.abs(v - anchor(initial.reconstruction).R[i]!) < 1e-9,
    ),
  );
  assert.ok(
    anchor(refined.reconstruction).t.every(
      (v, i) => Math.abs(v - anchor(initial.reconstruction).t[i]!) < 1e-9,
    ),
  );
  const baseline = (m: typeof initial.reconstruction) => {
    const centers = initial.diagnostics.initialPair.map((id) =>
      cameraCenter(m.views.find((v) => v.id === id)!.pose),
    );
    return Math.hypot(...centers[0]!.map((v, i) => v - centers[1]![i]!));
  };
  assert.ok(Math.abs(baseline(refined.reconstruction) - baseline(initial.reconstruction)) < 1e-9);
  callbacks.length = 0;
  const repeated = refineIncremental(f.inputs, f.pairs, refined, options);
  assert.equal(repeated.inverseDepth!.diagnostics.uniqueObservations, 240);
  assert.deepEqual(
    repeated.inverseDepth!.tracks.map((t) => t.id),
    after.tracks.map((t) => t.id),
  );
  assert.deepEqual(callbacks, [60]);
  for (const old of after.tracks) {
    const current = repeated.inverseDepth!.tracks.find((t) => t.id === old.id)!;
    assert.ok(
      Math.abs(current.covariance![8]! / old.covariance![8]! - 1) < 1e-5,
      'reusing the same observations must not artificially shrink uncertainty',
    );
  }
});

test('refinement publishes pending depths rather than native XYZ when uncertainty rejects admission', () => {
  const f = refinementFixture(),
    initial = reconstructIncremental(f.limited.inputs, f.limited.pairs, f.controls),
    tracker = new InverseDepthTracker({ maxRelativeStddev: 1e-12, pixelNoiseStdDevPx: 0.25 }),
    callbacks: number[] = [],
    refined = refineIncremental(f.inputs, f.pairs, initial, {
      ...f.controls,
      depthTracking: tracker,
      onReconstruction: (m) => callbacks.push(m.landmarks.length),
    });
  assert.equal(initial.reconstruction.landmarks.length, 30);
  assert.equal(refined.reconstruction.views.length, 4);
  assert.equal(refined.reconstruction.landmarks.length, 0);
  assert.equal(refined.inverseDepth!.diagnostics.uniqueObservations, 240);
  assert.equal(refined.inverseDepth!.diagnostics.finite, 0);
  assert.equal(refined.tracking!.untriangulatedTrackIds.length, 60);
  assert.deepEqual(callbacks, [0]);
});
