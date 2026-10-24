import assert from 'node:assert/strict';

export function assertInlineVotes(result) {
  assert.equal(result.liveBytes, 0);
  for (const {
    kind,
    boxed,
    runs: [legacy, inline],
    sourceUnchanged,
  } of result.results) {
    const label = `${kind}/${boxed ? 'boxes' : 'whole'}`;
    assert.ok(legacy.count > 0, label);
    assert.equal(inline.count, legacy.count, label);
    assert.deepEqual(inline.channels, legacy.channels, label);
    assert.equal(Object.keys(inline.channels).length, 7);
    assert.equal(sourceUnchanged, true, label);
    assert.equal(inline.liveBytes, 0, label);
    assert.equal(legacy.liveBytes, 0, label);
    assert.equal(legacy.groups, 0, label);
    assert.equal(inline.groups, 3 * inline.passes, label);
    assert.equal(inline.passes, legacy.passes, label);
    if (boxed) assert.ok(inline.passes > 1, label);
    assert.ok(inline.maxFree >= 2, `${label}: both independent contradictory groups contribute`);
    assert.ok(
      inline.maxFreeWeight > 1,
      `${label}: later weaker group adds evidence after early return`,
    );
    assert.ok(
      legacy.kernels.some((row) => row.name === 'tsdfResetVotes'),
      label,
    );
    assert.ok(!inline.kernels.some((row) => row.name === 'tsdfResetVotes'), label);
  }
}
