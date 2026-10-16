// Independent scalar image formation and residuals. No TypeShade implementation imports.
function random(seed) {
  let s = seed;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 2 ** 32;
  };
}
export function homographyFixture({ n = 97, offset = 0, scale = 1, collinear = false } = {}) {
  const next = random(683),
    H = [1.03, 0.04, 27, -0.02, 0.98, -13, 0.00015, -0.00007, 1];
  const pixels = new Float64Array(4 * n),
    clean = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const x = next() * 600,
      y = collinear ? 2 * x + 1 : next() * 350,
      z = H[6] * x + H[7] * y + H[8];
    let u = (H[0] * x + H[1] * y + H[2]) / z,
      v = (H[3] * x + H[4] * y + H[5]) / z;
    clean[i] = i % 5 === 0 ? 0 : 1;
    if (clean[i]) {
      u += (next() - 0.5) * 0.04;
      v += (next() - 0.5) * 0.04;
    } else {
      u = next() * 600;
      v = next() * 350;
    }
    pixels.set(
      [offset + scale * x, -offset + scale * y, offset + scale * u, -offset + scale * v],
      4 * i,
    );
  }
  return { pixels, clean, scale };
}
export function symmetricTransferErrors(H, pixels) {
  const a = H,
    inv = [
      a[4] * a[8] - a[5] * a[7],
      a[2] * a[7] - a[1] * a[8],
      a[1] * a[5] - a[2] * a[4],
      a[5] * a[6] - a[3] * a[8],
      a[0] * a[8] - a[2] * a[6],
      a[2] * a[3] - a[0] * a[5],
      a[3] * a[7] - a[4] * a[6],
      a[1] * a[6] - a[0] * a[7],
      a[0] * a[4] - a[1] * a[3],
    ];
  const transfer = (h, x, y, u, v) => {
    const z = h[6] * x + h[7] * y + h[8];
    return (
      ((h[0] * x + h[1] * y + h[2]) / z - u) ** 2 + ((h[3] * x + h[4] * y + h[5]) / z - v) ** 2
    );
  };
  return Array.from({ length: pixels.length / 4 }, (_, i) => {
    const [x, y, u, v] = pixels.subarray(4 * i, 4 * i + 4);
    return (transfer(a, x, y, u, v) + transfer(inv, u, v, x, y)) / 2;
  });
}
function image(k, X) {
  const x = X[0] / X[2],
    y = X[1] / X[2],
    r = x * x + y * y,
    d = 1 + (k.k1 ?? 0) * r + (k.k2 ?? 0) * r * r;
  return [k.fx * x * d + k.cx, k.fy * y * d + k.cy];
}
export function radialPairFixture({
  n = 97,
  baseline = 1,
  noise = 0.12,
  outliers = true,
  planar = false,
  folded = false,
} = {}) {
  const first = {
      fx: 1100,
      fy: 900,
      cx: 640,
      cy: 360,
      k1: folded ? -0.3 : 0.08,
      k2: folded ? 0 : -0.004,
    },
    second = {
      fx: 970,
      fy: 1030,
      cx: 480,
      cy: 300,
      k1: folded ? -0.25 : -0.035,
      k2: folded ? 0 : 0.003,
    };
  const theta = 0.11,
    c = Math.cos(theta),
    s = Math.sin(theta),
    R = [c, 0, s, 0, 1, 0, -s, 0, c],
    t = [0.45 * baseline, -0.03 * baseline, 0.05 * baseline],
    next = random(951),
    pixels = new Float64Array(4 * n),
    clean = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const X = [(next() - 0.5) * 4, (next() - 0.5) * 2, planar ? 6 : 4 + next() * 5],
      Y = t.map((v, k) => v + R[3 * k] * X[0] + R[3 * k + 1] * X[1] + R[3 * k + 2] * X[2]);
    const obs = [...image(first, X), ...image(second, Y)].map((v) => v + (next() - 0.5) * noise);
    clean[i] = outliers && i % 5 === 0 ? 0 : 1;
    if (!clean[i]) {
      obs[2] = next() * 960;
      obs[3] = next() * 600;
    }
    pixels.set(obs, 4 * i);
  }
  return { pixels, first, second, R, t, clean };
}

export function compactedPairFixture(
  n = 197,
  { noise = 0.03, outliers = true, allInvalid = false } = {},
) {
  const f = radialPairFixture({ n, folded: true, noise, outliers }),
    validRows = [];
  for (let i = 0; i < n; i++) {
    if (allInvalid || i % 11 === 0 || i % 17 === 3 || (i >= 64 && i < 128) || i === n - 1)
      f.pixels[4 * i + (i % 2 === 0 ? 0 : 3)] = 1e6;
    else validRows.push(i);
  }
  return { ...f, validRows };
}
export function recoveredMetrics(r, fixture) {
  const trace = r.R.reduce((s, x, i) => s + x * fixture.R[i], 0),
    rotationErrorDeg = (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI,
    tNorm = Math.hypot(...r.t),
    translationCosine =
      r.t.reduce((s, x, i) => s + x * fixture.t[i], 0) / (tNorm * Math.hypot(...fixture.t));
  let sum = 0,
    count = 0,
    minDepth = Infinity,
    maxError = 0;
  for (let i = 0; i < r.validPoints.length; i++)
    if (r.validPoints[i]) {
      const X = Array.from(r.points.subarray(i * 3, i * 3 + 3)),
        Y = Array.from(
          r.t,
          (v, k) => v + r.R[3 * k] * X[0] + r.R[3 * k + 1] * X[1] + r.R[3 * k + 2] * X[2],
        ),
        a = image(fixture.first, X),
        b = image(fixture.second, Y),
        ea = Math.hypot(a[0] - fixture.pixels[4 * i], a[1] - fixture.pixels[4 * i + 1]),
        eb = Math.hypot(b[0] - fixture.pixels[4 * i + 2], b[1] - fixture.pixels[4 * i + 3]);
      sum += (ea + eb) / 2;
      count++;
      minDepth = Math.min(minDepth, X[2], Y[2]);
      maxError = Math.max(maxError, ea, eb);
    }
  return {
    rotationErrorDeg,
    translationCosine,
    tNorm,
    meanError: sum / count,
    count,
    minDepth,
    maxError,
  };
}

export function relativePoseCases() {
  const multiply = (a, b) =>
    Array.from(
      { length: 9 },
      (_, i) =>
        a[Math.floor(i / 3) * 3] * b[i % 3] +
        a[Math.floor(i / 3) * 3 + 1] * b[3 + (i % 3)] +
        a[Math.floor(i / 3) * 3 + 2] * b[6 + (i % 3)],
    );
  const translations = [
    [0.45, -0.04, 0.07],
    [-0.45, 0.02, 0.07],
    [0.01, -0.45, -0.02],
    [0.08, 0.02, 0.4],
    [0.15, 0.1, -0.38],
    [0.02, 0.43, 0.04],
  ];
  return Array.from({ length: 12 }, (_, id) => {
    const first = { fx: 1200, fy: 850, cx: 640, cy: 360 },
      second = { fx: 750, fy: 1450, cx: 480, cy: 300 },
      ax = (id - 5) * 0.035,
      ay = ((id % 4) - 2) * 0.13,
      az = ((id % 3) - 1) * 0.17,
      cx = Math.cos(ax),
      sx = Math.sin(ax),
      cy = Math.cos(ay),
      sy = Math.sin(ay),
      cz = Math.cos(az),
      sz = Math.sin(az),
      R = multiply(multiply([cz, -sz, 0, sz, cz, 0, 0, 0, 1], [cy, 0, sy, 0, 1, 0, -sy, 0, cy]), [
        1,
        0,
        0,
        0,
        cx,
        -sx,
        0,
        sx,
        cx,
      ]),
      t = translations[id % translations.length],
      next = random(1951 + id),
      pixels = new Float64Array(24 * 4),
      world = new Float64Array(24 * 3);
    for (let i = 0; i < 24; i++) {
      const X = [(next() - 0.5) * 4, (next() - 0.5) * 3, 4 + next() * 4],
        Y = t.map((v, k) => v + R[3 * k] * X[0] + R[3 * k + 1] * X[1] + R[3 * k + 2] * X[2]);
      world.set(X, i * 3);
      pixels.set([...image(first, X), ...image(second, Y)], i * 4);
    }
    return { id, first, second, R, t, pixels, world };
  });
}
