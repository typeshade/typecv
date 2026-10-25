import { poseFixture } from './geometry-cases.mjs';

/** Long static tracks with a few deliberately inconsistent measurements, independently projected. */
export function continuedMapperFixture() {
  const count = 24,
    views = 8;
  const source = poseFixture({ count, noise: 0, outliers: 0, radial: false });
  const images = Array.from({ length: views }, (_, i) => ({
    id: 10 + i * 10,
    camera: { ...source.camera },
    keypoints: Float64Array.from(
      Array.from({ length: count }, (_, p) => {
        const X = source.world.subarray(3 * p, 3 * p + 3);
        const bad = i > 2 && i % 3 === 0 && p % 11 === 0;
        return [
          (source.camera.fx * (X[0] - 0.3 * i)) / X[2] + source.camera.cx + (bad ? 12 : 0),
          (source.camera.fy * X[1]) / X[2] + source.camera.cy,
        ];
      }).flat(),
    ),
  }));
  const pairs = [];
  for (let i = 0; i < views; i++)
    for (let j = i + 1; j < views; j++)
      if (i === 0 || j === i + 1)
        pairs.push({
          first: images[i].id,
          second: images[j].id,
          matches: Uint32Array.from(Array.from({ length: count }, (_, p) => [p, p]).flat()),
        });
  return { images, pairs };
}

export function continuedMapperSignature(result, events, published) {
  const words = (a) => [...new Uint32Array(a.buffer, a.byteOffset, a.length * 2)];
  return {
    views: result.reconstruction.views.map((v) => ({
      id: v.id,
      camera: v.camera,
      R: words(v.pose.R),
      t: words(v.pose.t),
    })),
    points: result.reconstruction.landmarks.map((p) => ({
      trackId: p.trackId,
      xyz: words(p.xyz),
      observations: p.observations,
    })),
    diagnostics: result.diagnostics,
    unregistered: result.unregistered,
    events,
    published,
  };
}
