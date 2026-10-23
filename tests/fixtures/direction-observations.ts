import { seededRandom } from '../../src/geometry/numerics.ts';
import { generalVideoFixture, type GeneralVideoCase } from './general-video.ts';

/** Supplied identities/known K isolate observability. Ground truth never enters mapper input. */
export function directionObservationFixture({
  finiteCount = 60,
  directionalCount = 60,
  noisePx = 0,
  scenario = 'translation',
  zoom = false,
  distantDepth = Infinity,
  screenFixed = false,
}: {
  finiteCount?: number;
  directionalCount?: number;
  noisePx?: number;
  scenario?: GeneralVideoCase;
  zoom?: boolean;
  distantDepth?: number;
  screenFixed?: boolean;
} = {}) {
  const base = generalVideoFixture(scenario),
    random = seededRandom(8173),
    bearings = Array.from({ length: directionalCount }, (_, i) => [
      ((i % 10) - 4.5) * 0.055,
      -0.25 - Math.floor(i / 10) * 0.02,
      1,
    ]);
  const views = base.truth.views.map((v, index) => ({
    ...v,
    camera: {
      ...v.camera,
      fx: v.camera.fx * (zoom ? 1 + index * 0.16 : 1),
      fy: v.camera.fy * (zoom ? 1 + index * 0.16 : 1),
    },
  }));
  const images = views.map((view, index) => {
    const { R, t } = view.pose,
      k = view.camera;
    const directional = bearings.flatMap((d) => {
      const q = [0, 1, 2].map((r) =>
        screenFixed
          ? d[r]!
          : R[r * 3]! * d[0]! +
            R[r * 3 + 1]! * d[1]! +
            R[r * 3 + 2]! +
            (Number.isFinite(distantDepth) ? t[r]! / distantDepth : 0),
      );
      return [(k.fx * q[0]!) / q[2]! + k.cx, (k.fy * q[1]!) / q[2]! + k.cy];
    });
    const oldK = base.input.images[index]!.camera,
      finite = Array.from(base.input.images[index]!.keypoints.slice(0, finiteCount * 2), (v, i) =>
        i % 2 ? ((v - oldK.cy) * k.fy) / oldK.fy + k.cy : ((v - oldK.cx) * k.fx) / oldK.fx + k.cx,
      );
    return {
      id: view.id,
      camera: { ...k },
      keypoints: Float64Array.from(
        [...finite, ...directional],
        (v) => v + (random() * 2 - 1) * noisePx,
      ),
    };
  });
  const pairs = base.input.pairs.map((p) => ({
    ...p,
    matches: Uint32Array.from(
      Array.from({ length: finiteCount + directionalCount }, (_, i) => [i, i]).flat(),
    ),
  }));
  return {
    input: { images, pairs },
    truth: { views, finitePoints: base.truth.points.slice(0, finiteCount), bearings },
    provenance: {
      mode: 'geometry-only',
      calibration: 'known-synthetic',
      correspondences: 'supplied-synthetic-identities',
      noise: { distribution: 'uniform', boundPx: noisePx, seed: 8173 },
    },
  };
}

export function directionPairPixels(f: ReturnType<typeof directionObservationFixture>, second = 4) {
  const first = f.input.images[0]!,
    last = f.input.images[second]!;
  return Float64Array.from(
    Array.from({ length: first.keypoints.length / 2 }, (_, i) => [
      first.keypoints[i * 2]!,
      first.keypoints[i * 2 + 1]!,
      last.keypoints[i * 2]!,
      last.keypoints[i * 2 + 1]!,
    ]).flat(),
  );
}
