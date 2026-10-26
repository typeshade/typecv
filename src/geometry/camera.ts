import type { CameraIntrinsics } from './two-view.ts';

/** Undistorted pinhole if k1=k2=0; otherwise Brown radial distortion only. */
export interface Camera extends CameraIntrinsics {
  readonly k1?: number;
  readonly k2?: number;
}
export type Pixel = readonly [number, number];
export function validateCamera(k: Camera): void {
  if (
    ![k.fx, k.fy, k.cx, k.cy, k.k1 ?? 0, k.k2 ?? 0].every(Number.isFinite) ||
    k.fx <= 0 ||
    k.fy <= 0
  )
    throw new RangeError('Invalid camera: finite parameters and positive focal lengths required');
}

// First radial derivative zero bounds the injective branch connected to r=0.
export function radialLimitSquared(k1: number, k2: number): number {
  if (k2 === 0) return k1 < 0 ? -1 / (3 * k1) : Infinity;
  const A = 5 * k2,
    B = 3 * k1,
    D = B * B - 4 * A;
  if (D < 0) return Infinity;
  const q = -0.5 * (B + (B >= 0 ? 1 : -1) * Math.sqrt(D));
  const roots = [q / A, 1 / q].filter((x) => x > 0 && Number.isFinite(x));
  return roots.length ? Math.min(...roots) : Infinity;
}

/** Projection and analytic 2x3 derivative with respect to camera-frame XYZ. */
export function cameraProjection(
  k: Camera,
  q: ArrayLike<number>,
): { pixel: Pixel; jacobian: Float64Array } | undefined {
  validateCamera(k);
  const X = q[0]!,
    Y = q[1]!,
    Z = q[2]!;
  if (!(Z > 1e-12) || ![X, Y, Z].every(Number.isFinite)) return undefined;
  const x = X / Z,
    y = Y / Z,
    r2 = x * x + y * y,
    k1 = k.k1 ?? 0,
    k2 = k.k2 ?? 0;
  if (r2 >= radialLimitSquared(k1, k2)) return undefined;
  const d = 1 + k1 * r2 + k2 * r2 * r2,
    g = 2 * (k1 + 2 * k2 * r2);
  const xx = d + g * x * x,
    xy = g * x * y,
    yy = d + g * y * y;
  // Reject non-injective radial regions rather than silently project through a lens fold.
  if (!(d > 0 && 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2 > 0)) return undefined;
  const pixel: Pixel = [k.fx * x * d + k.cx, k.fy * y * d + k.cy];
  const jacobian = Float64Array.of(
    (k.fx * xx) / Z,
    (k.fx * xy) / Z,
    (-k.fx * (xx * x + xy * y)) / Z,
    (k.fy * xy) / Z,
    (k.fy * yy) / Z,
    (-k.fy * (xy * x + yy * y)) / Z,
  );
  return pixel.every(Number.isFinite) && jacobian.every(Number.isFinite)
    ? { pixel, jacobian }
    : undefined;
}

/** Invert radial distortion on the monotone branch connected to the optical axis. */
export function unprojectPixel(k: Camera, pixel: Pixel): Pixel | undefined {
  validateCamera(k);
  const xd = (pixel[0] - k.cx) / k.fx,
    yd = (pixel[1] - k.cy) / k.fy,
    rd = Math.hypot(xd, yd);
  if (!Number.isFinite(rd)) return undefined;
  if (rd === 0) return [0, 0];
  const k1 = k.k1 ?? 0,
    k2 = k.k2 ?? 0;
  const limit2 = radialLimitSquared(k1, k2);
  if (Number.isFinite(limit2) && rd >= Math.sqrt(limit2) * (1 + k1 * limit2 + k2 * limit2 * limit2))
    return undefined;
  let r = rd;
  for (let i = 0; i < 50; i++) {
    const r2 = r * r,
      d = 1 + k1 * r2 + k2 * r2 * r2,
      slope = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2;
    if (!(slope > 1e-10 && d > 0)) return undefined;
    const residual = r * d - rd;
    if (Math.abs(residual) <= 1e-12 * Math.max(1, rd)) return [(xd * r) / rd, (yd * r) / rd];
    // Backtracking keeps Newton iterations on the injective branch.
    let step = residual / slope,
      accepted = false;
    for (let j = 0; j < 25; j++) {
      const next = r - step,
        n2 = next * next;
      if (
        next >= 0 &&
        n2 < limit2 &&
        1 + 3 * k1 * n2 + 5 * k2 * n2 * n2 > 0 &&
        Math.abs(next * (1 + k1 * n2 + k2 * n2 * n2) - rd) < Math.abs(residual)
      ) {
        r = next;
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) return undefined;
  }
  return undefined;
}
