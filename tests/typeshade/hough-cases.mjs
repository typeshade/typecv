/** Test images for the Hough lines tests. Plain functions of integers, so the browser test can embed them. */

/** The settings the synthetic tests were written with: absolute thresholds 0.1 and 0.2, and 64 votes at 256 px. */
export const ABSOLUTE = { thresholdMode: 'absolute', threshold: 64 };

/** Gray `value` on every channel of a width x height RGBA image. */
const gray = (width, height, at) => {
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const v = at(x, y);
      rgba.set([v, v, v, 255], (y * width + x) * 4);
    }
  return rgba;
};

/**
 * Bright strokes 12 px thick on black. A stroke is the set of pixels with
 * `0 <= x cos(theta) + y sin(theta) - rho < 12`, so its two borders are the lines (theta, rho) and
 * (theta, rho + 12).
 */
export function strokes(width, height, list) {
  return gray(width, height, (x, y) => {
    for (const [degrees, rho] of list) {
      const t = (degrees * Math.PI) / 180,
        d = x * Math.cos(t) + y * Math.sin(t) - rho;
      if (d >= 0 && d < 12) return 220;
    }
    return 20;
  });
}

/** Two bars on one horizontal band, with a gap between them. */
export function brokenBar(width, height, [y0, x0, x1, x2, x3]) {
  return gray(width, height, (x, y) =>
    y >= y0 && y < y0 + 12 && ((x >= x0 && x < x1) || (x >= x2 && x < x3)) ? 220 : 20,
  );
}

/** A vertical step edge between columns `at - 1` and `at`. */
export const step = (width, height, at) => gray(width, height, (x) => (x < at ? 0 : 255));

/** Deterministic noise around mid-gray. */
export function noise(width, height, amplitude, seed = 11) {
  let state = seed;
  return gray(width, height, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return 128 + (((state >>> 8) % (2 * amplitude + 1)) - amplitude);
  });
}

/**
 * A photo-like image: slow shading, fine texture (three sinusoids of 5 to 9 px), noise of +-8 and
 * the strokes of `strokes` 70 gray levels brighter than the ground they cross.
 */
export function photo(width, height, list, seed = 5) {
  let state = seed;
  return gray(width, height, (x, y) => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    let v =
      110 +
      25 * Math.sin(x / 70) * Math.cos(y / 90) +
      9 * Math.sin(x / 1.1 + y / 2.3) +
      7 * Math.sin(x / 1.6 - y / 1.4) +
      6 * Math.sin(y / 1.3) +
      (((state >>> 8) % 17) - 8);
    for (const [degrees, rho] of list) {
      const t = (degrees * Math.PI) / 180,
        d = x * Math.cos(t) + y * Math.sin(t) - rho;
      if (d >= 0 && d < 12) v += 70;
    }
    return Math.max(0, Math.min(255, Math.round(v)));
  });
}

export const STROKES = [
  [0, 40],
  [30, 150],
  [90, 200],
  [135, -70],
];

/** The accumulator of an edge map, in the fixed-point arithmetic of the kernel but written out here. */
export function accumulate(edges, width, thetaBins, rhoCount, maxRho) {
  const acc = new Uint32Array(thetaBins * rhoCount);
  for (let p = 0; p < edges.length; p++) {
    if (!edges[p]) continue;
    const x = p % width,
      y = (p - x) / width;
    for (let t = 0; t < thetaBins; t++) {
      const c = Math.round(Math.cos((Math.PI * t) / thetaBins) * 65536),
        s = Math.round(Math.sin((Math.PI * t) / thetaBins) * 65536);
      acc[t * rhoCount + ((x * c + y * s + 32768) >> 16) + maxRho]++;
    }
  }
  return acc;
}

/** The share of equal entries. */
export function agreement(a, b) {
  let same = 0;
  for (let i = 0; i < a.length; i++) if (a[i] === b[i]) same++;
  return same / a.length;
}

/** The closest line to (degrees, rho): angle distance in degrees (mod 180) and rho distance. */
export function nearest(lines, degrees, rho) {
  let best = { dTheta: Infinity, dRho: Infinity };
  for (const line of lines) {
    const raw = Math.abs((line.theta * 180) / Math.PI - degrees),
      dTheta = Math.min(raw, 180 - raw),
      flipped = raw > 90 ? -line.rho : line.rho,
      dRho = Math.abs(flipped - rho);
    if (dTheta + dRho < best.dTheta + best.dRho) best = { dTheta, dRho };
  }
  return best;
}
