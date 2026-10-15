import { rotationExp, transformPoint } from '../src/geometry/pose.ts';
import { seededRandom } from '../src/geometry/numerics.ts';
import { cameraProjection } from '../src/geometry/camera.ts';
import type { Camera } from '../src/geometry/camera.ts';
import type {
  ImageFeatures,
  PairMatches,
  SparseReconstruction,
} from '../src/reconstruction/types.ts';
export function scene({
  count = 90,
  views = 6,
  noise = 0,
  seed = 731,
  camera = { fx: 850, fy: 870, cx: 480, cy: 270 } as Camera,
} = {}) {
  const rng = seededRandom(seed),
    points = Array.from({ length: count }, () =>
      Float64Array.of((rng() - 0.5) * 4, (rng() - 0.5) * 3, 4 + rng() * 5),
    );
  const registered = Array.from({ length: views }, (_, i) => {
    const R = rotationExp([0.008 * i, -0.018 * i, 0.003 * i]),
      C = Float64Array.of(i * 0.32, Math.sin(i) * 0.04, 0.02 * i);
    return {
      id: i * 10 + 10,
      camera: { ...camera },
      pose: { R, t: transformPoint({ R, t: new Float64Array(3) }, C).map((x) => -x) },
    };
  });
  const inputs: ImageFeatures[] = registered.map((v) => ({
    id: v.id,
    camera: v.camera,
    keypoints: Float64Array.from(
      points.flatMap((X) => {
        const p = cameraProjection(camera, transformPoint(v.pose, X))!.pixel;
        return [p[0] + (rng() - 0.5) * noise * 2, p[1] + (rng() - 0.5) * noise * 2];
      }),
    ),
  }));
  const pairs: PairMatches[] = [];
  for (let i = 0; i < views; i++)
    for (let j = i + 1; j < views; j++)
      pairs.push({
        first: inputs[i]!.id,
        second: inputs[j]!.id,
        matches: Uint32Array.from(points.flatMap((_, k) => [k, k])),
      });
  const reconstruction: SparseReconstruction = {
    views: registered,
    landmarks: points.map((xyz, i) => ({
      trackId: i,
      xyz: xyz.slice(),
      observations: inputs.map((v) => ({
        viewId: v.id,
        featureId: i,
        pixel: [v.keypoints[2 * i]!, v.keypoints[2 * i + 1]!] as const,
      })),
    })),
  };
  return { points, registered, inputs, pairs, reconstruction, rng };
}
