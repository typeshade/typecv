// The browser bundle of the guided-matching probe: it runs `nearestFloatGuided` (the scalar
// oracle) or `nearestFloatGuidedTiles` on raw inputs and returns their raw outputs, so a test can
// hold the two kernels to each other bit for bit. `guided-probe-build.mjs` bundles it; the
// package does not export these kernels.
import { ShadeDevice } from '../../src/gpu/device.ts';
import {
  nearestFloatGuided,
  nearestFloatGuidedTiles,
} from '../../src/gpu/kernels/float-matching.shade.ts';

export { ShadeDevice };

/**
 * Nearest and second-nearest neighbours of each query under the epipolar gate. `matrix` is the
 * row-major 3 × 3 matrix that maps a query point to its line in the candidate image. The result
 * holds `indices` (best, second, invalid per query) and the f32 bits of `distances` (best,
 * second per query).
 */
export async function nearestGuided(device, kernel, input) {
  const { queries, candidates, dimensions, queryPoints, candidatePoints, matrix, maxLinePx } =
    input;
  const n = queries.length / dimensions,
    m = candidates.length / dimensions;
  return device.enqueue(async () => {
    const owned = [];
    const own = (buffer) => (owned.push(buffer), buffer);
    try {
      const floatQueries = own(device.allocate(queries.length, Float32Array.from(queries))),
        floatCandidates = own(device.allocate(candidates.length, Float32Array.from(candidates))),
        queryBuffer = own(device.allocate(queryPoints.length, Float32Array.from(queryPoints))),
        candidateBuffer = own(
          device.allocate(candidatePoints.length, Float32Array.from(candidatePoints)),
        ),
        nearestIndices = own(device.allocateU32(n * 3)),
        nearestDistances = own(device.allocate(n * 2));
      const m32 = Array.from(matrix, Math.fround);
      const bindings = {
        floatQueries,
        floatCandidates,
        queryPoints: queryBuffer,
        candidatePoints: candidateBuffer,
        nearestIndices,
        nearestDistances,
        floatMatchControl: {
          queries: n,
          candidates: m,
          dimensions,
          ratioSquared: Math.fround(0.64),
          maxDistanceSquared: Math.fround(1e38),
          mutual: 0,
        },
        epipolarGuide: {
          m0: m32[0],
          m1: m32[1],
          m2: m32[2],
          m3: m32[3],
          m4: m32[4],
          m5: m32[5],
          m6: m32[6],
          m7: m32[7],
          m8: m32[8],
          maxLinePx: Math.fround(maxLinePx),
        },
      };
      const report = await device.planSteps([
        [
          kernel,
          () =>
            kernel === 'tiles'
              ? nearestFloatGuidedTiles(bindings, Math.ceil(n / 8))
              : nearestFloatGuided(bindings, Math.ceil(n / 64)),
        ],
      ]);
      const indices = await device.read(nearestIndices, n * 3),
        distances = await device.read(nearestDistances, n * 2);
      return {
        tier: report.tier,
        indices: Array.from(indices.subarray(0, n * 3)),
        distanceBits: Array.from(
          new Uint32Array(Float32Array.from(distances.subarray(0, n * 2)).buffer),
        ),
      };
    } finally {
      for (const buffer of owned) device.drop(buffer);
    }
  });
}
