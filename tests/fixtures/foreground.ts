import { projectPoint } from '../../src/geometry/pose.ts';
import { seededRandom } from '../../src/geometry/numerics.ts';
import { generalVideoFixture } from './general-video.ts';
import type { CameraObservationMask } from '../../src/reconstruction/observation-selection.ts';

/** Supplied identity geometry, NOT an RGB-only or automatic-segmentation experiment. */
export function foregroundFixture() {
  const f = generalVideoFixture('translation'),
    rng = seededRandom(7561),
    background = 60,
    moving = 180,
    hud = 60;
  const object = Array.from({ length: moving }, () => [
    (rng() - 0.5) * 3,
    (rng() - 0.5) * 2,
    5 + 3 * rng(),
  ]);
  const overlays = Array.from({ length: hud }, () => [20 + rng() * 600, 20 + rng() * 440]);
  const images = f.input.images.map((v, i) => ({
    ...v,
    keypoints: Float64Array.from([
      ...v.keypoints,
      ...object.flatMap((p) =>
        projectPoint(v.camera, f.truth.views[i]!.pose, [
          p[0]! + 0.65 * i,
          p[1]! + 0.05 * i,
          p[2]!,
        ])!,
      ),
      ...overlays.flat(),
    ]),
  }));
  const pairs = f.input.pairs.map((p) => ({
    ...p,
    matches: Uint32Array.from(
      Array.from({ length: background + moving + hud }, (_, i) => [i, i]).flat(),
    ),
  }));
  const masks: CameraObservationMask[] = images.map((v) => ({
    viewId: v.id,
    source: 'ground-truth',
    values: Uint8Array.from({ length: background + moving + hud }, (_, i) =>
      i < background ? 1 : 0,
    ),
  }));
  return { ...f, images, pairs, masks, background, moving, hud };
}
