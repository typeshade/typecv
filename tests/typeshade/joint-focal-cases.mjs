// Independent image formation: no TypeCV geometry or TypeShade kernels.
export function jointFocalFixture({
  views = 6,
  points = 60,
  focalScale = 0.75,
  noise = 0,
  radial = true,
} = {}) {
  const multiply = (a, b) =>
    Array.from({ length: 9 }, (_, i) =>
      [0, 1, 2].reduce((s, k) => s + a[Math.floor(i / 3) * 3 + k] * b[k * 3 + (i % 3)], 0),
    );
  const rotation = (x, y, z) => {
    const cx = Math.cos(x),
      sx = Math.sin(x),
      cy = Math.cos(y),
      sy = Math.sin(y),
      cz = Math.cos(z),
      sz = Math.sin(z);
    return multiply(
      [cz, -sz, 0, sz, cz, 0, 0, 0, 1],
      multiply([cy, 0, sy, 0, 1, 0, -sy, 0, cy], [1, 0, 0, 0, cx, -sx, 0, sx, cx]),
    );
  };
  const camera = {
    fx: 800,
    fy: 840,
    cx: 511.5,
    cy: 383.5,
    k1: radial ? -0.04 : 0,
    k2: radial ? 0.004 : 0,
  };
  const truth = { views: [], landmarks: [] },
    input = { views: [], landmarks: [] };
  for (let c = 0; c < views; c++) {
    const angles = [0.03 * Math.sin(c), 0.075 * c, 0.035 * Math.sin(c * 0.7)],
      R = rotation(...angles),
      t = [-0.4 * c, 0.12 * Math.sin(c * 1.3), 0.08 * (Math.cos(c) - 1)];
    truth.views.push({
      id: 10 + 10 * c,
      camera: { ...camera },
      pose: { R: Float64Array.from(R), t: Float64Array.from(t) },
    });
    input.views.push({
      id: 10 + 10 * c,
      camera: { ...camera, fx: camera.fx * focalScale, fy: camera.fy * focalScale },
      pose: {
        R: Float64Array.from(
          rotation(...angles.map((v, j) => v + (c ? 0.003 * Math.sin(c + j) : 0))),
        ),
        t: Float64Array.from(t.map((v, j) => v + (c ? 0.012 * Math.sin(c + j) : 0))),
      },
    });
  }
  let seed = 117;
  const next = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
  for (let p = 0; p < points; p++) {
    const X = [(next() - 0.5) * 3, (next() - 0.5) * 2, 3 + next() * 6];
    const observations = truth.views.map((v, c) => {
      const q = [0, 1, 2].map(
        (j) => v.pose.t[j] + [0, 1, 2].reduce((s, k) => s + v.pose.R[j * 3 + k] * X[k], 0),
      );
      const x = q[0] / q[2],
        y = q[1] / q[2],
        r = x * x + y * y,
        d = 1 + camera.k1 * r + camera.k2 * r * r;
      return {
        viewId: v.id,
        featureId: p,
        pixel: [
          camera.fx * x * d + camera.cx + noise * Math.sin(p + c),
          camera.fy * y * d + camera.cy + noise * Math.cos(p * 2 + c),
        ],
      };
    });
    truth.landmarks.push({ trackId: p, xyz: Float64Array.from(X), observations });
    input.landmarks.push({
      trackId: p,
      xyz: Float64Array.from(X.map((v, j) => v + 0.025 * Math.sin(p + j))),
      observations,
    });
  }
  return { input, truth, expectedScale: 1 / focalScale };
}
