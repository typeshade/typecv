/**
 * Similarity alignment of one set of camera poses to a reference set, on their common
 * views. The same method as `scripts/colmap-datasets/colmap_dataset.py` `align`: the
 * rotation comes from the camera orientations (the Procrustes solution over the
 * stacked rotation products), the scale and the translation from the camera centers.
 * Both pose sets use the world-to-camera convention: x_camera = R x_world + t, so the
 * camera center is -Rᵀ t.
 */
export interface CameraPose {
  readonly id: number;
  /** Row-major 3 × 3 rotation. */
  readonly R: ArrayLike<number>;
  readonly t: ArrayLike<number>;
}
export interface PoseAlignment {
  /** `aligned`, or the reason no alignment was possible. */
  readonly status: string;
  readonly commonViews: number;
  readonly rotationDeg: { readonly median: number; readonly max: number };
  /** Center error divided by the reference trajectory's radius about its centroid. */
  readonly centerOfRadius: { readonly median: number; readonly max: number };
  readonly perView: readonly {
    readonly id: number;
    readonly rotationDeg: number;
    readonly centerOfRadius: number;
  }[];
  readonly scale: number;
}

type Mat3 = [number, number, number, number, number, number, number, number, number];
type Vec3 = [number, number, number];

const transpose = (m: ArrayLike<number>): Mat3 => [
  m[0],
  m[3],
  m[6],
  m[1],
  m[4],
  m[7],
  m[2],
  m[5],
  m[8],
];
const multiply = (a: ArrayLike<number>, b: ArrayLike<number>): Mat3 => {
  const out = new Array<number>(9) as Mat3;
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return out;
};
const apply = (m: ArrayLike<number>, v: ArrayLike<number>): Vec3 => [
  m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
  m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
  m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
];
const det = (m: ArrayLike<number>) =>
  m[0] * (m[4] * m[8] - m[5] * m[7]) -
  m[1] * (m[3] * m[8] - m[5] * m[6]) +
  m[2] * (m[3] * m[7] - m[4] * m[6]);
const inverse = (m: ArrayLike<number>): Mat3 | undefined => {
  const d = det(m);
  if (!(Math.abs(d) > 1e-300)) return undefined;
  return [
    (m[4] * m[8] - m[5] * m[7]) / d,
    (m[2] * m[7] - m[1] * m[8]) / d,
    (m[1] * m[5] - m[2] * m[4]) / d,
    (m[5] * m[6] - m[3] * m[8]) / d,
    (m[0] * m[8] - m[2] * m[6]) / d,
    (m[2] * m[3] - m[0] * m[5]) / d,
    (m[3] * m[7] - m[4] * m[6]) / d,
    (m[1] * m[6] - m[0] * m[7]) / d,
    (m[0] * m[4] - m[1] * m[3]) / d,
  ];
};
/** The center of a world-to-camera pose. */
export const cameraCenter = (pose: { R: ArrayLike<number>; t: ArrayLike<number> }): Vec3 => {
  const c = apply(transpose(pose.R), pose.t);
  return [-c[0], -c[1], -c[2]];
};
/**
 * The orthogonal polar factor of a matrix with positive determinant: Higham's Newton
 * iteration X ← (X + X⁻ᵀ) / 2, which converges quadratically to the rotation nearest M.
 */
function polarRotation(m: Mat3): Mat3 | undefined {
  if (!(det(m) > 0)) return undefined;
  let x = m;
  for (let iteration = 0; iteration < 60; iteration++) {
    const inv = inverse(x);
    if (!inv) return undefined;
    const invT = transpose(inv);
    const next = x.map((value, i) => 0.5 * (value + invT[i])) as Mat3;
    const change = Math.max(...next.map((value, i) => Math.abs(value - x[i])));
    x = next;
    if (change < 1e-15) break;
  }
  return x;
}
const rotationAngleDeg = (m: ArrayLike<number>) =>
  (Math.acos(Math.max(-1, Math.min(1, (m[0] + m[4] + m[8] - 1) / 2))) * 180) / Math.PI;
const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};
const failure = (status: string, commonViews: number): PoseAlignment => ({
  status,
  commonViews,
  rotationDeg: { median: NaN, max: NaN },
  centerOfRadius: { median: NaN, max: NaN },
  perView: [],
  scale: NaN,
});

/** Align `source` poses to `reference` poses on the views both hold. */
export function alignPoses(
  source: readonly CameraPose[],
  reference: readonly CameraPose[],
): PoseAlignment {
  const byId = new Map(reference.map((pose) => [pose.id, pose]));
  const common = source.filter((pose) => byId.has(pose.id)).sort((a, b) => a.id - b.id);
  if (common.length < 3) return failure('fewer than three common cameras', common.length);
  // Q minimizes Σ ‖R_ref − Q R_src‖: the polar factor of Σ R_refᵀ R_src transposed.
  const sum = new Array<number>(9).fill(0) as Mat3;
  for (const pose of common) {
    const product = multiply(transpose(byId.get(pose.id)!.R), pose.R);
    for (let i = 0; i < 9; i++) sum[i] += product[i];
  }
  const q = polarRotation(sum);
  if (!q) return failure('camera orientations do not determine a rotation', common.length);
  // In Python: Q = u diag(1, 1, det) vt from the SVD of Σ R_refᵀ R_src; the polar factor is the same.
  const xs = common.map((pose) => apply(q, cameraCenter(pose)));
  const ys = common.map((pose) => cameraCenter(byId.get(pose.id)!));
  const centroid = (points: Vec3[]): Vec3 => {
    const c: Vec3 = [0, 0, 0];
    for (const p of points) for (let i = 0; i < 3; i++) c[i] += p[i] / points.length;
    return c;
  };
  const cx = centroid(xs),
    cy = centroid(ys);
  let variance = 0,
    covariance = 0,
    extent = 0;
  for (let k = 0; k < common.length; k++)
    for (let i = 0; i < 3; i++) {
      variance += (xs[k][i] - cx[i]) ** 2;
      covariance += (xs[k][i] - cx[i]) * (ys[k][i] - cy[i]);
    }
  for (const y of ys)
    extent = Math.max(extent, Math.hypot(y[0] - cy[0], y[1] - cy[1], y[2] - cy[2]));
  if (!(variance > 1e-20)) return failure('source trajectory has no extent', common.length);
  const scale = covariance / variance;
  if (!(scale > 0) || !Number.isFinite(scale))
    return failure('similarity scale is not positive', common.length);
  if (!(extent > 1e-12)) return failure('reference trajectory has no extent', common.length);
  const perView = common.map((pose, k) => {
    const ref = byId.get(pose.id)!;
    // D = R_ref (R_src Qᵀ)ᵀ, the residual rotation after alignment.
    const residual = multiply(ref.R, transpose(multiply(pose.R, transpose(q))));
    const aligned = xs[k].map((value, i) => scale * (value - cx[i]) + cy[i]);
    const centerError = Math.hypot(
      aligned[0] - ys[k][0],
      aligned[1] - ys[k][1],
      aligned[2] - ys[k][2],
    );
    return {
      id: pose.id,
      rotationDeg: rotationAngleDeg(residual),
      centerOfRadius: centerError / extent,
    };
  });
  const rotations = perView.map((view) => view.rotationDeg),
    centers = perView.map((view) => view.centerOfRadius);
  return {
    status: 'aligned',
    commonViews: common.length,
    rotationDeg: { median: median(rotations), max: Math.max(...rotations) },
    centerOfRadius: { median: median(centers), max: Math.max(...centers) },
    perView,
    scale,
  };
}
