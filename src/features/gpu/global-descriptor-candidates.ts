/**
 * The baseline candidate of the global descriptor (`docs/multi-video.md`, 3.2), and the k-means
 * that trains a vocabulary. They are plain TypeScript on the host. The recall measurement of
 * step 3 (`scripts/global-descriptor.mjs recall`) ranks the candidates with them. Nothing in the
 * multi-video path calls the thumbnail. VLAD, the other candidate, is the default descriptor and
 * lives in `global-descriptor.ts` (`vladDescriptorReference`).
 */
import { seededRandom } from '../../geometry/numerics.ts';
import { GLOBAL_DIMENSIONS } from './global-descriptor.ts';

/**
 * The baseline: a small gray thumbnail (for example 16 × 16) with its mean removed and its L2
 * norm set to 1, so that the dot product of two is their normalized cross-correlation. A flat
 * thumbnail gives the zero vector.
 */
export function thumbnailDescriptor(gray: ArrayLike<number>): Float32Array {
  let mean = 0;
  for (let i = 0; i < gray.length; i++) mean += gray[i]!;
  mean /= gray.length || 1;
  let sum = 0;
  const out = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    out[i] = gray[i]! - mean;
    sum += out[i]! * out[i]!;
  }
  const norm = Math.sqrt(sum);
  if (norm > 0) for (let i = 0; i < out.length; i++) out[i] = out[i]! / norm;
  return out;
}

export interface KMeansOptions {
  /** Lloyd iterations; always this many, with no early stop. */
  readonly iterations: number;
  readonly seed: number;
  readonly dimensions?: number;
  /** Called after each iteration with the mean squared distance to the nearest center. */
  readonly onIteration?: (iteration: number, meanSquaredDistance: number) => void;
}

/**
 * Lloyd's k-means in f32 with a fixed iteration count. The initial centers are `k` distinct
 * points chosen with the seeded generator (a partial Fisher-Yates shuffle of the point indices).
 * A center that loses all its points keeps its place. The same input, `k`, iteration count and
 * seed give the same centers. The nearest center is found by `|c|² - 2 x·c`, ties to the lower index.
 */
export function kMeans(points: Float32Array, k: number, options: KMeansOptions): Float32Array {
  const dim = options.dimensions ?? GLOBAL_DIMENSIONS,
    n = points.length / dim;
  if (!Number.isInteger(n) || k < 1 || n < k)
    throw new RangeError('k-means needs at least k points');
  const random = seededRandom(options.seed),
    order = Uint32Array.from({ length: n }, (_, i) => i);
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(random() * (n - i)),
      t = order[i]!;
    order[i] = order[j]!;
    order[j] = t;
  }
  const centers = new Float32Array(k * dim);
  for (let c = 0; c < k; c++)
    centers.set(points.subarray(order[c]! * dim, order[c]! * dim + dim), c * dim);
  const assignment = new Uint32Array(n),
    norms = new Float64Array(k),
    sums = new Float64Array(k * dim),
    counts = new Uint32Array(k);
  for (let iteration = 0; iteration < options.iterations; iteration++) {
    for (let c = 0; c < k; c++) {
      let s = 0;
      for (let d = 0; d < dim; d++) s += centers[c * dim + d]! ** 2;
      norms[c] = s;
    }
    sums.fill(0);
    counts.fill(0);
    let error = 0;
    for (let i = 0; i < n; i++) {
      const base = i * dim;
      let best = Infinity,
        bestC = 0;
      for (let c = 0; c < k; c++) {
        let dot = 0;
        const cb = c * dim;
        for (let d = 0; d < dim; d++) dot += points[base + d]! * centers[cb + d]!;
        const score = norms[c]! - 2 * dot;
        if (score < best) {
          best = score;
          bestC = c;
        }
      }
      assignment[i] = bestC;
      counts[bestC]!++;
      for (let d = 0; d < dim; d++) sums[bestC * dim + d]! += points[base + d]!;
      let x2 = 0;
      for (let d = 0; d < dim; d++) x2 += points[base + d]! ** 2;
      error += x2 + best;
    }
    for (let c = 0; c < k; c++)
      if (counts[c])
        for (let d = 0; d < dim; d++) centers[c * dim + d] = sums[c * dim + d]! / counts[c]!;
    options.onIteration?.(iteration, error / n);
  }
  return centers;
}
