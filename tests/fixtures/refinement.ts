import { scene } from '../scene.ts';

// Geometry-only API regression; image measurement is tested separately by observation-expansion.
export function refinementFixture() {
  const full = scene({ count: 60, views: 4 });
  const inputs = full.inputs.map((v, i) => ({
    ...v,
    keypoints: i >= 2 ? v.keypoints.slice(0, 60) : v.keypoints.slice(),
  }));
  const pairs = full.pairs.map((p) => ({
    ...p,
    matches: Uint32Array.from(
      Array.from({ length: 30 }, (_, f) => f)
        .filter((f) => f < 20 || (p.first !== 30 && p.second !== 30))
        .flatMap((f) => [f, f]),
    ),
  }));
  return {
    ...full,
    limited: { inputs, pairs },
    controls: {
      ransacIterations: 100,
      bundleIterations: 1,
      finalBundleIterations: 2,
      initialPair: [10, 20] as const,
    },
  };
}
