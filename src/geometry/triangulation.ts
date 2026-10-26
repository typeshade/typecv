import { nullspace } from './numerics.ts';

/** Linear DLT triangulation from calibrated x1, x2 and X2=R X1+t. */
export function triangulate(
  x: number,
  y: number,
  u: number,
  v: number,
  R: ArrayLike<number>,
  t: ArrayLike<number>,
): Float64Array | undefined {
  const rows = [
    [-1, 0, x, 0],
    [0, -1, y, 0],
    [u * R[6]! - R[0]!, u * R[7]! - R[1]!, u * R[8]! - R[2]!, u * t[2]! - t[0]!],
    [v * R[6]! - R[3]!, v * R[7]! - R[4]!, v * R[8]! - R[5]!, v * t[2]! - t[1]!],
  ];
  if (
    R.length !== 9 ||
    t.length !== 3 ||
    ![x, y, u, v, ...Array.from(R), ...Array.from(t)].every(Number.isFinite)
  )
    throw new RangeError('Triangulation requires finite coordinates and valid pose dimensions');
  try {
    const { vector: h, singularValues: s } = nullspace(rows, 4);
    if (s[1]! < 1e-12 * s[3]! || Math.abs(h[3]!) < 1e-12) return undefined;
    const X = h.slice(0, 3).map((value) => value / h[3]!);
    return X.every(Number.isFinite) ? X : undefined;
  } catch {
    return undefined;
  }
}
