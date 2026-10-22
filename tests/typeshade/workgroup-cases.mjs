import { floatDescriptorFixture } from './float-matching-cases.mjs';

/** Deterministic SfM-shaped descriptors, with narrow and wide epipolar bands. */
export function guidedWorkgroupFixture(queries, candidates, dimensions = 128, maxLinePx = 2) {
  const a = floatDescriptorFixture(queries, dimensions, 17);
  const b = floatDescriptorFixture(candidates, dimensions, 29);
  for (let i = 0; i < Math.min(queries, candidates); i++)
    for (let d = 0; d < dimensions; d++) b[i * dimensions + d] = a[i * dimensions + d] + 0.01;
  return {
    name: `${queries}x${candidates}x${dimensions} band ${maxLinePx}px`,
    queries: [...a],
    candidates: [...b],
    dimensions,
    queryPoints: Array.from({ length: queries }, (_, i) => [i % 640, (i * 37) % 360]).flat(),
    candidatePoints: Array.from({ length: candidates }, (_, i) => [
      (i % 640) + 14,
      (i * 37) % 360,
    ]).flat(),
    matrix: [0, 0, 0, 0, 0, -1, 0, 1, 0],
    maxLinePx,
  };
}
