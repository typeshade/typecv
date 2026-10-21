/**
 * A synthetic orbit for the focal search (docs/auto-focal.md): 80 points in a box about 5 units
 * in front of 6 cameras that turn towards the box's centre, 640 × 480 pixels, a true focal of
 * 0.7 × width, and ±0.25 px of deterministic noise. Every view sees every point; each pair
 * matches every point.
 */
export function orbitFixture({ ratio = 0.7, start = 1.2, views = 6, points = 80 } = {}) {
  const width = 640,
    height = 480,
    f = ratio * width,
    cx = (width - 1) / 2,
    cy = (height - 1) / 2;
  let seed = 7;
  const random = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) * 2 - 1;
  const world = Array.from({ length: points }, () => [
    1.6 * random(),
    1.1 * random(),
    5 + 1.2 * random(),
  ]);
  const images = [];
  for (let c = 0; c < views; c++) {
    // The camera centre moves along an arc and the camera turns to keep (0, 0, 5) in view.
    const angle = 0.07 * c,
      C = [5 * Math.sin(angle), 0.1 * Math.sin(c), 5 - 5 * Math.cos(angle)];
    const ca = Math.cos(-angle),
      sa = Math.sin(-angle),
      R = [ca, 0, sa, 0, 1, 0, -sa, 0, ca];
    const keypoints = [];
    for (const X of world) {
      const d = X.map((x, i) => x - C[i]);
      const q = [0, 1, 2].map((r) => R[3 * r] * d[0] + R[3 * r + 1] * d[1] + R[3 * r + 2] * d[2]);
      keypoints.push(
        f * (q[0] / q[2]) + cx + 0.25 * random(),
        f * (q[1] / q[2]) + cy + 0.25 * random(),
      );
    }
    images.push({
      id: c,
      camera: { fx: start * width, fy: start * width, cx, cy, k1: 0, k2: 0 },
      keypoints: Float64Array.from(keypoints),
    });
  }
  const pairs = [];
  for (let b = 1; b < views; b++)
    for (let a = 0; a < b; a++)
      pairs.push({
        first: a,
        second: b,
        matches: Uint32Array.from(Array.from({ length: points }, (_, k) => [k, k]).flat()),
      });
  return { images, pairs, width, ratio };
}
