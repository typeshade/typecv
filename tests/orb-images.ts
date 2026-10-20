// Synthetic frames for the ORB tests: deterministic, integer-seeded, no files.

/** A deterministic generator in [0, 1). */
export function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/** Pack a gray image (0..255) into RGBA bytes. */
export function grayToRgba(gray: ArrayLike<number>, width: number, height: number): Uint8Array {
  const rgba = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    rgba[i * 4] =
      rgba[i * 4 + 1] =
      rgba[i * 4 + 2] =
        Math.max(0, Math.min(255, Math.round(gray[i]!)));
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

/**
 * A textured scene on a continuous domain, so that a warped copy can be sampled without a
 * resampling blur: smooth value noise at four scales, plus bars and discs with sharp edges.
 */
export function sceneSampler(seed: number): (x: number, y: number) => number {
  const random = lcg(seed),
    grid = (cell: number) => {
      const size = 64,
        values = Float64Array.from({ length: size * size }, () => random());
      return (x: number, y: number) => {
        const u = x / cell,
          v = y / cell,
          x0 = Math.floor(u),
          y0 = Math.floor(v),
          fx = u - x0,
          fy = v - y0,
          at = (i: number, j: number) =>
            values[(((j % size) + size) % size) * size + (((i % size) + size) % size)]!,
          sx = fx * fx * (3 - 2 * fx),
          sy = fy * fy * (3 - 2 * fy);
        return (
          (at(x0, y0) * (1 - sx) + at(x0 + 1, y0) * sx) * (1 - sy) +
          (at(x0, y0 + 1) * (1 - sx) + at(x0 + 1, y0 + 1) * sx) * sy
        );
      };
    },
    layers = [grid(5), grid(11), grid(23), grid(47)],
    shapes = Array.from({ length: 120 }, () => ({
      x: random() * 640,
      y: random() * 640,
      r: 4 + random() * 22,
      value: 20 + random() * 215,
      disc: random() < 0.5,
      angle: random() * Math.PI,
    }));
  return (x, y) => {
    let value =
      40 +
      30 * layers[0]!(x, y) +
      40 * layers[1]!(x, y) +
      40 * layers[2]!(x, y) +
      40 * layers[3]!(x, y);
    for (const s of shapes) {
      const dx = x - s.x,
        dy = y - s.y;
      if (s.disc) {
        if (dx * dx + dy * dy < s.r * s.r) value = s.value;
      } else {
        const c = Math.cos(s.angle),
          sn = Math.sin(s.angle),
          u = c * dx + sn * dy,
          v = -sn * dx + c * dy;
        if (Math.abs(u) < s.r && Math.abs(v) < s.r * 0.4) value = s.value;
      }
    }
    return Math.max(0, Math.min(255, value));
  };
}

/** Render `sampler` through the 3 x 3 homography `h` (row-major, source = h applied to target pixels). */
export function renderHomography(
  sampler: (x: number, y: number) => number,
  h: readonly number[],
  width: number,
  height: number,
): Uint8Array {
  const gray = new Float64Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const w = h[6]! * x + h[7]! * y + h[8]!;
      gray[y * width + x] = sampler(
        (h[0]! * x + h[1]! * y + h[2]!) / w,
        (h[3]! * x + h[4]! * y + h[5]!) / w,
      );
    }
  return grayToRgba(gray, width, height);
}

export function invert3(m: readonly number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = m as [
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
    ],
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  return [
    (e * i - f * h) / det,
    (c * h - b * i) / det,
    (b * f - c * e) / det,
    (f * g - d * i) / det,
    (a * i - c * g) / det,
    (c * d - a * f) / det,
    (d * h - e * g) / det,
    (b * g - a * h) / det,
    (a * e - b * d) / det,
  ];
}

export function mapPoint(m: readonly number[], x: number, y: number): [number, number] {
  const w = m[6]! * x + m[7]! * y + m[8]!;
  return [(m[0]! * x + m[1]! * y + m[2]!) / w, (m[3]! * x + m[4]! * y + m[5]!) / w];
}

/** A rotation by `angle` about (cx, cy), then a translation. */
export function similarity(angle: number, cx: number, cy: number, tx = 0, ty = 0): number[] {
  const c = Math.cos(angle),
    s = Math.sin(angle);
  return [c, -s, cx - c * cx + s * cy + tx, s, c, cy - s * cx - c * cy + ty, 0, 0, 1];
}

/** Hamming distance of two 8-word descriptors at the given word offsets. */
export function hamming(a: Uint32Array, ia: number, b: Uint32Array, ib: number): number {
  let distance = 0;
  for (let w = 0; w < 8; w++) {
    let v = (a[ia * 8 + w]! ^ b[ib * 8 + w]!) >>> 0;
    while (v) {
      distance += v & 1;
      v >>>= 1;
    }
  }
  return distance;
}

/**
 * A bright convex polygon (clockwise in image coordinates) on a dark field, edges anti-aliased by
 * 4 x 4 coverage and a 3 x 3 box blur, so that a corner has one strict score maximum; returns RGBA.
 */
export function polygonImage(
  width: number,
  height: number,
  vertices: readonly (readonly [number, number])[],
  inside = 220,
  outside = 30,
): Uint8Array {
  const covered = (px: number, py: number) => {
      for (let i = 0; i < vertices.length; i++) {
        const [x0, y0] = vertices[i]!,
          [x1, y1] = vertices[(i + 1) % vertices.length]!;
        if ((x1 - x0) * (py - y0) - (y1 - y0) * (px - x0) < 0) return false;
      }
      return true;
    },
    coverage = new Float64Array(width * height),
    gray = new Float64Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let hits = 0;
      for (let j = 0; j < 4; j++)
        for (let i = 0; i < 4; i++)
          if (covered(x - 0.5 + (i + 0.5) / 4, y - 0.5 + (j + 0.5) / 4)) hits++;
      coverage[y * width + x] = hits / 16;
    }
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++)
          sum +=
            coverage[
              Math.min(height - 1, Math.max(0, y + dy)) * width +
                Math.min(width - 1, Math.max(0, x + dx))
            ]!;
      gray[y * width + x] = outside + ((inside - outside) * sum) / 9;
    }
  return grayToRgba(gray, width, height);
}
