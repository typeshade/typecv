export function fixture(n = 513) {
  const first = { fx: 1200, fy: 1100, cx: 640, cy: 360 },
    second = { fx: 900, fy: 800, cx: 320, cy: 240 };
  let state = 606;
  const next = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 2 ** 32;
  };
  const pixels = new Float64Array(4 * n);
  for (let i = 0; i < n; i++) {
    const x = (next() - 0.5) * 4,
      y = (next() - 0.5) * 2,
      z = 4 + 4 * next();
    pixels.set(
      [
        (first.fx * x) / z + first.cx,
        (first.fy * y) / z + first.cy,
        (second.fx * (x + 0.4)) / z + second.cx,
        (second.fy * y) / z + second.cy + (i % 5 === 0 ? 40 : 0) + (next() - 0.5) * 0.08,
      ],
      4 * i,
    );
  }
  const correct = [0, 0, 0, 0, 0, -1, 0, 1, 0],
    wrong = [0, 0, 1, 0, 0, 0, -1, 0, 0];
  return {
    pixels,
    first,
    second,
    models: Float64Array.from(
      [
        new Array(9).fill(0),
        wrong,
        correct,
        correct.map((x) => 2 * x),
        correct.map((x) => x * 1e300),
      ].flat(),
    ),
    threshold: 0.001,
  };
}

export function reconstructionFixture(n = 321) {
  const { first, second } = fixture(1),
    angle = 0.07;
  const R = [Math.cos(angle), 0, Math.sin(angle), 0, 1, 0, -Math.sin(angle), 0, Math.cos(angle)];
  const t = [0.4, -0.03, 0.06],
    pixels = new Float64Array(n * 4);
  let state = 404;
  const next = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 2 ** 32;
  };
  for (let i = 0; i < n; i++) {
    const X = [(next() - 0.5) * 4, (next() - 0.5) * 2, 4 + next() * 5];
    const Y = t.map((v, k) => v + R[3 * k] * X[0] + R[3 * k + 1] * X[1] + R[3 * k + 2] * X[2]);
    const observation = [
      (first.fx * X[0]) / X[2] + first.cx,
      (first.fy * X[1]) / X[2] + first.cy,
      (second.fx * Y[0]) / Y[2] + second.cx,
      (second.fy * Y[1]) / Y[2] + second.cy,
    ];
    pixels.set(
      observation.map((v) => v + (next() - 0.5) * 0.16),
      i * 4,
    );
    if (i % 5 === 0) {
      pixels[4 * i + 2] = next() * 640;
      pixels[4 * i + 3] = next() * 480;
    }
  }
  return { pixels, first, second, R, t };
}

export function poseMetrics(result, truth) {
  const trace = result.R.reduce((sum, x, i) => sum + x * truth.R[i], 0);
  const rotationErrorDeg = (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI;
  const translationCosine =
    result.t.reduce((sum, x, i) => sum + x * truth.t[i], 0) / Math.hypot(...truth.t);
  return {
    rotationErrorDeg,
    translationCosine,
    valid: result.positiveDepthCount,
    errorPx: result.meanReprojectionErrorPx,
    execution: result.execution,
    trials: result.diagnostics.ransacIterations,
  };
}
// Independent scalar f64 reference; no TypeShade kernel or GPU runtime is imported.
export function reference({ pixels, first: a, second: b, models, threshold }) {
  const n = pixels.length / 4,
    h = models.length / 9,
    counts = [],
    costs = [],
    masks = [];
  for (let m = 0; m < h; m++) {
    const raw = Array.from(models.subarray(m * 9, m * 9 + 9)),
      scale = Math.max(...raw.map(Math.abs));
    const unit = scale ? raw.map((x) => x / scale) : raw,
      norm = Math.hypot(...unit),
      E = norm ? unit.map((x) => x / norm) : unit;
    let count = 0,
      cost = 0;
    const mask = [];
    for (let i = 0; i < n; i++) {
      const x = (pixels[i * 4] - a.cx) / a.fx,
        y = (pixels[i * 4 + 1] - a.cy) / a.fy,
        u = (pixels[i * 4 + 2] - b.cx) / b.fx,
        v = (pixels[i * 4 + 3] - b.cy) / b.fy;
      const ex = [
        E[0] * x + E[1] * y + E[2],
        E[3] * x + E[4] * y + E[5],
        E[6] * x + E[7] * y + E[8],
      ];
      const et = [E[0] * u + E[3] * v + E[6], E[1] * u + E[4] * v + E[7]];
      const den = ex[0] ** 2 + ex[1] ** 2 + et[0] ** 2 + et[1] ** 2,
        d = den > 1e-12 ? (u * ex[0] + v * ex[1] + ex[2]) ** 2 / den : Infinity;
      const inlier = d <= threshold ** 2 ? 1 : 0;
      mask.push(inlier);
      count += inlier;
      cost += Math.min(d / threshold ** 2, 1);
    }
    counts.push(count);
    costs.push(cost);
    masks.push(mask);
  }
  let best = 0;
  for (let i = 1; i < h; i++)
    if (counts[i] > counts[best] || (counts[i] === counts[best] && costs[i] < costs[best]))
      best = i;
  return { counts, costs, masks, best };
}
