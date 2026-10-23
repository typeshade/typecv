import test from 'node:test';
import assert from 'node:assert/strict';
import {
  reconstructIncremental,
  ReconstructionInitializationError,
} from '../src/reconstruction/mapper.ts';
import { reconstructIncrementalScene } from '../src/reconstruction/scene-mapper.ts';
import { generalVideoFixture } from './fixtures/general-video.ts';

// An infinite direction field follows orientation only, even while the physical camera translates.
// Known calibration and supplied point identities isolate geometry; no semantic classifier.
function directionalInput(mixed: boolean) {
  const fixture = generalVideoFixture('translation');
  const images = fixture.input.images.map((image, index) => {
    const { R } = fixture.truth.views[index]!.pose,
      k = image.camera;
    const directional = Array.from({ length: 60 }, (_, i) => {
      const d = [((i % 10) - 4.5) * 0.055, -0.25 - Math.floor(i / 10) * 0.02, 1];
      const q = [0, 1, 2].map((r) => R[r * 3]! * d[0]! + R[r * 3 + 1]! * d[1]! + R[r * 3 + 2]!);
      return [(k.fx * q[0]!) / q[2]! + k.cx, (k.fy * q[1]!) / q[2]! + k.cy];
    }).flat();
    return {
      ...image,
      keypoints: Float64Array.from([...(mixed ? image.keypoints : []), ...directional]),
    };
  });
  const pairs = fixture.input.pairs.map((p) => ({
    ...p,
    matches: Uint32Array.from(Array.from({ length: mixed ? 120 : 60 }, (_, i) => [i, i]).flat()),
  }));
  return { images, pairs };
}

test('infinite directions cannot initialize finite structure despite physical camera movement', () => {
  const { images, pairs } = directionalInput(false);
  for (const mapper of [reconstructIncremental, reconstructIncrementalScene]) {
    assert.throws(
      () => mapper(images, pairs, { ransacIterations: 128 }),
      (error: unknown) => {
        assert.ok(error instanceof ReconstructionInitializationError);
        assert.ok(error.tracking!.rotations.length > 0);
        assert.equal(error.tracking!.tracks.length, 60);
        assert.equal(error.tracking!.untriangulatedTrackIds.length, 60);
        assert.ok(error.tracking!.rotations.every((r) => r.rotation.depth === 'not-estimated'));
        return true;
      },
    );
  }
  // The uncalibrated mode must also defer; calibrated orientation tracking is not enabled there.
  assert.throws(
    () =>
      reconstructIncremental(images, pairs, {
        focalSearch: true,
        ransacIterations: 128,
      }),
    ReconstructionInitializationError,
  );
});

test(
  'finite structure and infinite directions retain separate depth outcomes',
  {
    todo: 'Pose recovery currently requires finite positive depth for 60% of all E inliers; split directional support before changing this guard',
  },
  () => {
    const { images, pairs } = directionalInput(true);
    for (const mapper of [reconstructIncremental, reconstructIncrementalScene]) {
      const result = mapper(images, pairs, {
        ransacIterations: 128,
        bundleIterations: 2,
        finalBundleIterations: 2,
      });
      assert.equal(result.reconstruction.views.length, 5);
      assert.equal(result.reconstruction.landmarks.length, 60);
      assert.ok(
        result.reconstruction.landmarks.every((p) => p.observations.every((o) => o.featureId < 60)),
      );
      assert.equal(result.tracking!.untriangulatedTrackIds.length, 60);
      assert.ok(
        result.tracking!.untriangulatedTrackIds.every((id) =>
          result.tracking!.tracks[id]!.observations.every((o) => o.featureId >= 60),
        ),
      );
    }
  },
);

test('featureless input defers without filling cameras or depths', () => {
  const { images } = directionalInput(false),
    empty = images.map((v) => ({ ...v, keypoints: new Float64Array() }));
  assert.throws(
    () => reconstructIncremental(empty, []),
    (error: unknown) => {
      assert.ok(error instanceof ReconstructionInitializationError);
      assert.equal(error.tracking!.tracks.length, 0);
      assert.equal(error.tracking!.rotations.length, 0);
      return true;
    },
  );
});
