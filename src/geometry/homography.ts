import { multiply } from './linear.ts';
import { inverse3, nullspace, ransacBudget, sample, seededRandom } from './numerics.ts';

/** Normalized DLT, row-major H maps (x,y,1) to (u,v,1). */
export function fitHomography(matches: readonly (readonly number[])[]): Float64Array {
  if (matches.length < 4 || matches.some((m) => m.length !== 4 || !m.every(Number.isFinite)))
    throw new RangeError('Homography requires four finite correspondences');
  const normalize = (offset: number) => {
    const cx = matches.reduce((s, m) => s + m[offset]!, 0) / matches.length,
      cy = matches.reduce((s, m) => s + m[offset + 1]!, 0) / matches.length;
    const distance =
      matches.reduce((s, m) => s + Math.hypot(m[offset]! - cx, m[offset + 1]! - cy), 0) /
      matches.length;
    if (distance < 1e-12) throw new Error('Degenerate homography distribution');
    const s = Math.SQRT2 / distance;
    return { cx, cy, s, T: Float64Array.of(s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1) };
  };
  const a = normalize(0),
    b = normalize(2),
    rows: number[][] = [];
  for (const m of matches) {
    const x = (m[0]! - a.cx) * a.s,
      y = (m[1]! - a.cy) * a.s,
      u = (m[2]! - b.cx) * b.s,
      v = (m[3]! - b.cy) * b.s;
    rows.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]);
  }
  const fit = nullspace(rows, 9);
  if (fit.singularValues[1]! < 1e-9 * fit.singularValues[8]!)
    throw new Error('Degenerate homography rank');
  const inv = Float64Array.of(1 / b.s, 0, b.cx, 0, 1 / b.s, b.cy, 0, 0, 1);
  const H = multiply(multiply(inv, fit.vector, 3), a.T, 3),
    norm = Math.hypot(...H);
  return H.map((x) => x / norm);
}
function transfer(H: Float64Array, x: number, y: number, u: number, v: number): number {
  const z = H[6]! * x + H[7]! * y + H[8]!;
  if (Math.abs(z) < 1e-12) return Infinity;
  return (
    ((H[0]! * x + H[1]! * y + H[2]!) / z - u) ** 2 + ((H[3]! * x + H[4]! * y + H[5]!) / z - v) ** 2
  );
}
export function estimateHomography(
  matches: Float64Array,
  options: { thresholdPx?: number; iterations?: number; seed?: number; confidence?: number } = {},
):
  | {
      H: Float64Array;
      inliers: Uint8Array;
      inlierCount: number;
      iterations: number;
    }
  | undefined {
  const threshold = options.thresholdPx ?? 2,
    iterations = options.iterations ?? 500,
    confidence = options.confidence ?? 0.999,
    seed = options.seed ?? 17;
  if (
    matches.length % 4 ||
    !matches.every(Number.isFinite) ||
    !Number.isFinite(threshold) ||
    threshold <= 0 ||
    !Number.isSafeInteger(iterations) ||
    iterations < 1 ||
    !Number.isSafeInteger(seed) ||
    !(confidence > 0 && confidence < 1)
  )
    throw new RangeError('Invalid homography inputs/options');
  const n = matches.length / 4;
  if (n < 4) return undefined;
  const data = Array.from({ length: n }, (_, i) => Array.from(matches.subarray(4 * i, 4 * i + 4))),
    rng = seededRandom(seed);
  const score = (H: Float64Array) => {
    const inv = inverse3(H),
      mask = new Uint8Array(n);
    let count = 0,
      cost = 0;
    if (!inv) return { mask, count, cost: Infinity };
    data.forEach((m, i) => {
      const d =
        (transfer(H, m[0]!, m[1]!, m[2]!, m[3]!) + transfer(inv, m[2]!, m[3]!, m[0]!, m[1]!)) / 2;
      if (d <= threshold * threshold) {
        mask[i] = 1;
        count++;
        cost += d;
      }
    });
    return { mask, count, cost };
  };
  let best: Float64Array | undefined,
    mask = new Uint8Array(n),
    count = 0,
    cost = Infinity,
    budget = iterations,
    trials = 0;
  for (let i = 0; i < budget; i++) {
    trials++;
    try {
      const H = fitHomography(sample(n, 4, rng).map((i) => data[i]!)),
        s = score(H);
      if (s.count > count || (s.count === count && s.cost < cost)) {
        best = H;
        mask = s.mask;
        count = s.count;
        cost = s.cost;
        budget = Math.min(
          budget,
          Math.max(Math.min(32, iterations), ransacBudget(count, n, 4, confidence)),
        );
      }
    } catch {
      /* Singular minimal sets are expected. */
    }
  }
  if (!best || count < 4) return undefined;
  for (let pass = 0; pass < 3; pass++)
    try {
      const H = fitHomography(data.filter((_, i) => mask[i])),
        s = score(H);
      if (s.count < count || (s.count === count && s.cost >= cost)) break;
      best = H;
      mask = s.mask;
      count = s.count;
      cost = s.cost;
    } catch {
      break;
    }
  return { H: best, inliers: mask, inlierCount: count, iterations: trials };
}
