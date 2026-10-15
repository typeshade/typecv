"use typeshade";

/**
 * A TypeShade kernel in the same style as TypeNN's src/kernels/*.shade.ts.
 * Padded 4-tuples: [x1, y1, x2, y2] in *normalized* camera coordinates.
 * E is a row-major 3x3 Essential matrix in a Resident<Float32Array>.
 * Each iteration writes out[i], and can be independently dispatched.
 */
export function epipolarSampsonSquared(
  out: array<f32>,
  xy: array<f32>,
  E: array<f32>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const j = i * 4;
    const x = xy[j];
    const y = xy[j + 1];
    const u = xy[j + 2];
    const v = xy[j + 3];

    const ex0 = E[0] * x + E[1] * y + E[2];
    const ex1 = E[3] * x + E[4] * y + E[5];
    const ex2 = E[6] * x + E[7] * y + E[8];
    const et0 = E[0] * u + E[3] * v + E[6];
    const et1 = E[1] * u + E[4] * v + E[7];
    const residual = u * ex0 + v * ex1 + ex2;
    const denominator = ex0 * ex0 + ex1 * ex1 + et0 * et0 + et1 * et1;
    out[i] = select(1e20, residual * residual / denominator, denominator > 1e-12);
  }
}

/**
 * Converts packed pixel matches into normalized coordinates on-device. This separate kernel
 * preserves the simple out[i] write shape needed for TypeShade's loop-independence proof.
 */
export function normalizeMatches(
  out: array<f32>,
  pixels: array<f32>,
  fx1: f32, fy1: f32, cx1: f32, cy1: f32,
  fx2: f32, fy2: f32, cx2: f32, cy2: f32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const lane = i % 4;
    const value = pixels[i];
    let normalized: f32 = 0.0;
    if (lane === 0) normalized = (value - cx1) / fx1;
    else if (lane === 1) normalized = (value - cy1) / fy1;
    else if (lane === 2) normalized = (value - cx2) / fx2;
    else normalized = (value - cy2) / fy2;
    out[i] = normalized;
  }
}
