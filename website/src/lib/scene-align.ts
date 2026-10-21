// The placement of a reconstruction on the viewer's ground grid. A reconstruction has no
// gravity: its frame is the first camera's. These helpers find a ground plane, from three
// picked points or from the points themselves, and orient it toward the cameras.

export type Vec3 = [number, number, number];

export interface Plane {
  /** The unit normal. */
  readonly normal: Vec3;
  /** A point on the plane. */
  readonly point: Vec3;
}

const sub = (a: ArrayLike<number>, b: ArrayLike<number>): Vec3 => [
  a[0]! - b[0]!,
  a[1]! - b[1]!,
  a[2]! - b[2]!,
];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

/** The plane through three points, or nothing when they are (nearly) on one line. */
export function planeFromPoints(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  c: ArrayLike<number>,
): Plane | undefined {
  const u = sub(b, a),
    v = sub(c, a);
  const n = cross(u, v);
  const length = Math.hypot(...n);
  // The sine of the angle at `a` below 1e-3 counts as one line.
  if (!(length > 1e-3 * Math.hypot(...u) * Math.hypot(...v))) return undefined;
  return { normal: [n[0] / length, n[1] / length, n[2] / length], point: [a[0]!, a[1]!, a[2]!] };
}

/**
 * The plane with its normal flipped, when needed, so that most of `centers` (x, y, z per
 * camera) lie on its positive side: the cameras look at the ground from above it.
 */
export function orientTowards(plane: Plane, centers: ArrayLike<number>): Plane {
  let above = 0,
    below = 0;
  for (let i = 0; i + 2 < centers.length; i += 3) {
    const d = dot(plane.normal, sub([centers[i]!, centers[i + 1]!, centers[i + 2]!], plane.point));
    if (d > 0) above++;
    else if (d < 0) below++;
  }
  if (above >= below) return plane;
  const [x, y, z] = plane.normal;
  return { normal: [-x, -y, -z], point: plane.point };
}

/** A small seeded generator (mulberry32), so a plane fit is the same on every run. */
function generator(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The eigenvector of the smallest eigenvalue of a symmetric 3 × 3 matrix (Jacobi rotations). */
export function smallestEigenvector(m: number[]): Vec3 {
  const a = m.slice();
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 32; sweep++) {
    const off = a[1]! ** 2 + a[2]! ** 2 + a[5]! ** 2;
    if (off < 1e-30) break;
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ] as const) {
      const apq = a[3 * p + q]!;
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (a[3 * q + q]! - a[3 * p + p]!) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1),
        s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[3 * k + p]!,
          akq = a[3 * k + q]!;
        a[3 * k + p] = c * akp - s * akq;
        a[3 * k + q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[3 * p + k]!,
          aqk = a[3 * q + k]!;
        a[3 * p + k] = c * apk - s * aqk;
        a[3 * q + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[3 * k + p]!,
          vkq = v[3 * k + q]!;
        v[3 * k + p] = c * vkp - s * vkq;
        v[3 * k + q] = s * vkp + c * vkq;
      }
    }
  }
  let best = 0;
  for (let k = 1; k < 3; k++) if (a[4 * k]! < a[4 * best]!) best = k;
  const e: Vec3 = [v[best]!, v[3 + best]!, v[6 + best]!];
  const length = Math.hypot(...e);
  return [e[0] / length, e[1] / length, e[2] / length];
}

export interface DominantPlane extends Plane {
  /** The share of the sampled points within the tolerance of the plane. */
  readonly inlierShare: number;
  /** The distance that counted as on the plane. */
  readonly tolerance: number;
}

/**
 * The plane that holds the most points (RANSAC over at most `sample` points, then a least
 * squares fit to its inliers). The tolerance is `toleranceFactor` (1 %) of the diagonal of the
 * 5th to 95th percentile box of the points. Nothing for fewer than 3 points.
 */
export function dominantPlane(
  positions: ArrayLike<number>,
  count: number,
  { iterations = 400, sample = 20000, toleranceFactor = 0.01, seed = 1 } = {},
): DominantPlane | undefined {
  if (count < 3) return undefined;
  const stride = Math.max(1, Math.floor(count / sample));
  const n = Math.floor((count - 1) / stride) + 1;
  const pts = new Float64Array(3 * n);
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) pts[3 * i + a] = positions[3 * i * stride + a]!;
  // The robust box: the 5th and 95th percentile on each axis.
  let diagonal = 0;
  for (let a = 0; a < 3; a++) {
    const axis = Float64Array.from({ length: n }, (_, i) => pts[3 * i + a]!).sort();
    const low = axis[Math.floor(0.05 * (n - 1))]!,
      high = axis[Math.ceil(0.95 * (n - 1))]!;
    diagonal += (high - low) ** 2;
  }
  const tolerance = toleranceFactor * Math.sqrt(diagonal) || 1e-9;
  const random = generator(seed);
  let best: Plane | undefined,
    bestCount = 0;
  const at = (i: number) => pts.subarray(3 * i, 3 * i + 3);
  for (let it = 0; it < iterations; it++) {
    const plane = planeFromPoints(
      at(Math.floor(random() * n)),
      at(Math.floor(random() * n)),
      at(Math.floor(random() * n)),
    );
    if (!plane) continue;
    let inliers = 0;
    for (let i = 0; i < n; i++)
      if (Math.abs(dot(plane.normal, sub(at(i), plane.point))) <= tolerance) inliers++;
    if (inliers > bestCount) {
      bestCount = inliers;
      best = plane;
    }
  }
  if (!best) return undefined;
  // Least squares: the centroid of the inliers and the normal of their smallest spread.
  const mean = [0, 0, 0],
    cov = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  let m = 0;
  for (let i = 0; i < n; i++) {
    const p = at(i);
    if (Math.abs(dot(best.normal, sub(p, best.point))) > tolerance) continue;
    m++;
    for (let a = 0; a < 3; a++) mean[a]! += p[a]!;
  }
  for (let a = 0; a < 3; a++) mean[a]! /= m;
  for (let i = 0; i < n; i++) {
    const p = at(i);
    if (Math.abs(dot(best.normal, sub(p, best.point))) > tolerance) continue;
    const d = sub(p, mean);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) cov[3 * r + c]! += d[r]! * d[c]!;
  }
  let normal = smallestEigenvector(cov);
  if (dot(normal, best.normal) < 0) normal = [-normal[0], -normal[1], -normal[2]];
  return {
    normal,
    point: [mean[0]!, mean[1]!, mean[2]!],
    inlierShare: bestCount / n,
    tolerance,
  };
}

/** The `q` quantile (0 to 1) of a list, by sorting a copy. */
export function quantile(values: ArrayLike<number>, q: number): number {
  const sorted = Float64Array.from(values).sort();
  if (!sorted.length) return NaN;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))]!;
}
