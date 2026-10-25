import assert from 'node:assert/strict';

export function assertStagingCase(result) {
  assert.ok(result.whole.count > 0);
  assert.equal(result.boxes.count, result.whole.count);
  assert.deepEqual(result.boxes.channels, result.whole.channels);
  assert.equal(result.explicit.count, result.whole.count);
  assert.deepEqual(result.explicit.channels, result.whole.channels);
  assert.equal(result.explicit.tsdf.hostViewStagingBytes, 0);
  assert.equal(Object.keys(result.whole.channels).length, 7);
  assert.equal(result.before, result.after);
  assert.equal(result.liveBytes, 0);
  assert.equal(result.whole.tsdf.passes, 1);
  assert.ok(result.boxes.tsdf.passes > 1);
  for (const { tsdf } of [result.whole, result.boxes]) {
    assert.equal(tsdf.allocationPixels, 16 * 16 * 8);
    assert.equal(tsdf.hostViewStagingBytes, 4 * 16 * 16);
    assert.equal(tsdf.uploadBytes, 16 * 16 * 8 * (16 + 28 * tsdf.passes));
  }
  assert.equal(result.whole.tsdf.reusedBoxTables, 0);
  assert.equal(result.boxes.tsdf.reusedBoxTables, result.boxes.tsdf.passes);
  assert.equal(result.boxes.tsdf.blocks, result.whole.tsdf.blocks);
  assert.equal(result.boxes.tsdf.voxel, result.whole.tsdf.voxel);
  assert.equal(result.boxes.tsdf.levels, result.whole.tsdf.levels);
  for (const { tsdf } of [result.whole, result.boxes, result.explicit]) {
    assert.equal(Object.keys(tsdf.hostTimings).length, 5);
    for (const [name, value] of Object.entries(tsdf.hostTimings))
      assert.ok(Number.isFinite(value) && value >= 0, name);
  }
  for (const failure of result.failures) {
    assert.equal(failure.reached, true, failure.kind);
    assert.equal(failure.liveBytes, 0, failure.kind);
    assert.equal(
      failure.error,
      failure.kind === 'abort'
        ? 'AbortError'
        : failure.kind === 'budget'
          ? 'RangeError'
          : 'synthetic callback failure',
    );
    if (failure.kind === 'budget') assert.match(failure.message, /working budget/);
  }
}
