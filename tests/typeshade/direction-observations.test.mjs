import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/gpu.js';
import { runDirectionCases } from './direction-observations-cases.mjs';
import { assertDirectionCases, directionFixtures } from './direction-observations-fixture.mjs';

test('TypeShade CPU separates finite and directional observations without a geometry fallback', async () => {
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 }),
    fixtures = directionFixtures();
  try {
    assertDirectionCases(await runDirectionCases(api, device, fixtures), fixtures);
  } finally {
    await device.dispose();
  }
});

test('streaming directional mode keeps its option snapshot and unresolved tracks across updates', async () => {
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 }),
    fixture = directionFixtures().mixed,
    options = {
      separateDirectionalObservations: true,
      ransacIterations: 128,
      bundleIterations: 2,
      finalBundleIterations: 2,
    },
    session = new api.IncrementalSfmSession(device, options);
  options.separateDirectionalObservations = false;
  try {
    for (const image of fixture.images) {
      const result = await session.add(
        { ...image, keypoints: Float64Array.from(image.keypoints) },
        fixture.pairs
          .filter((p) => p.second === image.id)
          .map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
      );
      if (image.id === 0) assert.equal(result, undefined);
      else {
        assert.equal(result.reconstruction.views.length, image.id + 1);
        assert.equal(result.reconstruction.landmarks.length, 60);
        assert.equal(session.tracking.untriangulatedTrackIds.length, 60);
      }
    }
    const result = await session.finish();
    assert.equal(result.reconstruction.views.length, 5);
    assert.equal(result.reconstruction.landmarks.length, 60);
    assert.equal(result.tracking.untriangulatedTrackIds.length, 60);
    assert.ok(
      result.reconstruction.landmarks.every((p) => p.observations.every((o) => o.featureId < 60)),
    );
    assert.equal(device.statistics.liveBuffers, 0);
  } finally {
    await device.dispose();
  }
});
