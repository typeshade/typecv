import test from 'node:test';
import assert from 'node:assert/strict';
import { reconstructIncremental, refineIncremental } from '../src/index.ts';

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
