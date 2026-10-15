import { reconstructIncremental, projectPoint, rotationExp } from 'typecv';

// A generated, calibrated scene. Production callers supply detected keypoints and raw matches.
let state = 731;
const random = () => {
  state ^= state << 13;
  state ^= state >>> 17;
  state ^= state << 5;
  return (state >>> 0) / 2 ** 32;
};
const camera = { fx: 900, fy: 910, cx: 480, cy: 270, k1: -0.03, k2: 0.004 };
const points = Array.from({ length: 80 }, () => [
  (random() - 0.5) * 4,
  (random() - 0.5) * 3,
  4 + random() * 5,
]);
const views = Array.from({ length: 5 }, (_, id) => {
  const R = rotationExp([0, -0.02 * id, 0]);
  const center = [0.35 * id, 0.02 * Math.sin(id), 0];
  const t = Float64Array.from(
    { length: 3 },
    (_, i) => -center.reduce((sum, x, j) => sum + R[i * 3 + j] * x, 0),
  );
  return {
    id,
    camera,
    keypoints: Float64Array.from(
      points.flatMap((X) => {
        const p = projectPoint(camera, { R, t }, X);
        return [p[0] + (random() - 0.5) * 0.3, p[1] + (random() - 0.5) * 0.3];
      }),
    ),
  };
});
const pairs = views.flatMap((a, i) =>
  views.slice(i + 1).map((b) => ({
    first: a.id,
    second: b.id,
    matches: Uint32Array.from(points.flatMap((_, i) => [i, i])),
  })),
);
const result = reconstructIncremental(views, pairs, { seed: 42 });
console.log(
  JSON.stringify(
    {
      registeredViews: result.reconstruction.views.length,
      landmarks: result.reconstruction.landmarks.length,
      initialPair: result.diagnostics.initialPair,
      meanReprojectionErrorPx: result.diagnostics.meanReprojectionErrorPx,
      unregistered: result.unregistered,
    },
    null,
    2,
  ),
);
