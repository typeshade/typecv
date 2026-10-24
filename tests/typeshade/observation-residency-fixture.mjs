import assert from 'node:assert/strict';
import { expansionFixture } from './observation-expansion-fixture.mjs';
import { consensusFixture } from './observation-surface-fixture.mjs';

/** These are image-tracking/API cases; the supplied finite model is not RGB-only truth input. */
export function residencyFixture() {
  const expansion = expansionFixture(),
    consensus = consensusFixture();
  return {
    cases: [
      { name: 'bidirectional-missed-seeds', frames: expansion.frames, pairs: [], radius: 2 },
      {
        name: 'original-witnesses',
        frames: consensus.frames,
        pairs: consensus.verifiedPairs,
        verifiedPairs: consensus.verifiedPairs,
        radius: 2,
        multiViewConsistency: true,
      },
      {
        name: 'finite-map-viewpoint',
        frames: consensus.geometryChoice.frames,
        pairs: [],
        model: consensus.geometryChoice.model,
        radius: 1,
        multiViewConsistency: true,
      },
    ],
  };
}

export function assertResidency(result, input, { tightBudget = false } = {}) {
  assert.equal(result.rows.length, input.cases.length);
  for (const row of result.rows) {
    const fixture = input.cases.find((entry) => entry.name === row.name),
      baseline = row.modes.find((mode) => mode.name === 'off'),
      automatic = row.modes.find((mode) => mode.name === 'auto'),
      legacy = row.modes.find((mode) => mode.name === 'three');
    assert.ok(baseline.quality.observations.length > 0, row.name + ': exercise real observations');
    for (const mode of row.modes) {
      assert.deepEqual(mode.quality, baseline.quality, row.name + ': ' + mode.name);
      assert.deepEqual(
        mode.repeatedQuality,
        baseline.quality,
        row.name + ': repeated ' + mode.name,
      );
      assert.deepEqual(
        mode.quality.images.map((image) => image.id),
        fixture.frames.map((frame) => frame.id),
        'every original frame remains represented',
      );
      assert.equal(
        mode.activeAfter,
        mode.sentinelBytes,
        'only unrelated caller allocation remains',
      );
      assert.equal(mode.activeAfterRepeat, mode.sentinelBytes);
      assert.equal(mode.activeAfterRelease, 0);
      assert.equal(mode.liveAfterDispose, 0);
      assert.equal(mode.pendingReads, 0);
      assert.equal(mode.originalGrayPreserved, true);
      assert.deepEqual(
        mode.originalKeypoints,
        fixture.frames.map((frame) => frame.keypoints),
      );
      assert.deepEqual(mode.sentinel, mode.originalSentinel, 'caller-owned evidence is unchanged');
      for (const image of mode.quality.images) {
        const original = fixture.frames.find((frame) => frame.id === image.id);
        assert.deepEqual(image.keypoints.slice(0, original.keypoints.length), original.keypoints);
      }
      for (const cache of [mode.cache, mode.repeatedCache]) {
        if (mode.name === 'off') assert.equal(cache, null);
        else {
          assert.equal(cache.frames, 0);
          assert.equal(cache.residentBytes, 0);
          assert.ok(cache.pyramidBuilds > 0);
          assert.equal(cache.peakFrames, mode.name === 'auto' || tightBudget ? 2 : 3);
        }
      }
      assert.equal(mode.invalidCacheOptions.length, 6);
      assert.ok(mode.invalidCacheOptions.every((error) => error === 'RangeError'));
      if (tightBudget) assert.ok(mode.peakActiveBytes <= mode.workingBytes);
    }
    assert.equal(
      automatic.peakActiveBytes,
      baseline.peakActiveBytes,
      'automatic reuse keeps the stateless two-pyramid active-storage peak',
    );
    if (!tightBudget)
      assert.ok(legacy.peakActiveBytes > automatic.peakActiveBytes, 'explicit true retains three');
    else assert.equal(legacy.peakActiveBytes, automatic.peakActiveBytes);
    if (row.name === 'original-witnesses') {
      assert.equal(automatic.quality.observations.length, 4);
      assert.ok(
        automatic.quality.observations.every((o) => o.corroboration?.status === 'verified'),
      );
    }
    if (row.name === 'finite-map-viewpoint') {
      assert.equal(automatic.quality.observations.length, 4);
      assert.ok(
        automatic.quality.observations.every(
          (o) => o.prediction === 'finite' && o.corroboration?.viewId === 3,
        ),
      );
    }
  }
}
