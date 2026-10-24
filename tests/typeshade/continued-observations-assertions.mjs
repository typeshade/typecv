import assert from 'node:assert/strict';

export function assertContinuedObservations(result) {
  assert.deepEqual(result.measurements, [384, 32]);
  assert.deepEqual(
    result.continued.masks,
    result.all.masks.map((mask) => [mask.at(-1)]),
  );
  assert.equal(
    result.continued.masks[0][0],
    0,
    'a newly rejected measurement must remain rejected',
  );
  assert.deepEqual(result.guarded, result.guardedAll);
  assert.deepEqual(result.weakContinued, result.weakAll);
  assert.ok(result.weakContinued.depthRejected.some((value) => value === 1));
  assert.ok(result.uploads[1] < result.uploads[0] / 3);
  assert.deepEqual(result.readbacks, [1552, 144]);
  assert.equal(result.guardedUploads[1], result.guardedUploads[0]);
  for (const row of result.invalid) {
    assert.match(row.failures[0], /^RangeError:/, row.kind);
    assert.equal(row.failures[1], row.failures[0], row.kind);
  }
  assert.equal(result.inputUnchanged, true);
  assert.deepEqual(result.snapshotted, result.reference);
  assert.equal(result.liveBytes, 0);
}
