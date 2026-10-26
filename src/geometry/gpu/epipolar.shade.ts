'use typeshade';

export function normalizeMatches(
  out: array<f32>,
  pixels: array<f32>,
  fx1: f32,
  fy1: f32,
  cx1: f32,
  cy1: f32,
  fx2: f32,
  fy2: f32,
  cx2: f32,
  cy2: f32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const lane = i % 4;
    let v: f32 = 0;
    if (lane === 0) v = (pixels[i] - cx1) / fx1;
    else if (lane === 1) v = (pixels[i] - cy1) / fy1;
    else if (lane === 2) v = (pixels[i] - cx2) / fx2;
    else v = (pixels[i] - cy2) / fy2;
    out[i] = v;
  }
}

// One invocation per (hypothesis, match). Models are unit-Frobenius E matrices.
export function sampsonBatch(
  out: array<f32>,
  xy: array<f32>,
  models: array<f32>,
  count: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const m = (i / count) * 9;
    const j = (i % count) * 4;
    const x = xy[j];
    const y = xy[j + 1];
    const u = xy[j + 2];
    const v = xy[j + 3];
    const a = models[m] * x + models[m + 1] * y + models[m + 2];
    const b = models[m + 3] * x + models[m + 4] * y + models[m + 5];
    const c = models[m + 6] * x + models[m + 7] * y + models[m + 8];
    const d = models[m] * u + models[m + 3] * v + models[m + 6];
    const e = models[m + 1] * u + models[m + 4] * v + models[m + 7];
    const residual = u * a + v * b + c;
    const denominator = a * a + b * b + d * d + e * e;
    let error: f32 = 1e20;
    if (denominator > 1e-12) error = (residual * residual) / denominator;
    out[i] = error;
  }
}

export function inlierFlags(out: array<f32>, errors: array<f32>, thresholdSquared: f32): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = select(0, 1, errors[i] <= thresholdSquared);
  }
}

export function msacCosts(out: array<f32>, errors: array<f32>, thresholdSquared: f32): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = min(errors[i], thresholdSquared) / thresholdSquared;
  }
}

// Hierarchical reduction keeps long observations off a single serial invocation.
// [count partials for all models, cost partials for all models].
export function blockStats(
  out: array<f32>,
  flags: array<f32>,
  costs: array<f32>,
  count: u32,
  blocks: u32,
  blockSize: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const half = out.length / 2;
    const lane = i / half;
    const index = i % half;
    const model = index / blocks;
    const start = (index % blocks) * blockSize;
    let total: f32 = 0;
    for (let k: u32 = 0; k < blockSize; k++) {
      const row = start + k;
      if (row < count) {
        const j = model * count + row;
        if (lane === 0) total += flags[j];
        else total += costs[j];
      }
    }
    out[i] = total;
  }
}

export function modelStats(out: array<f32>, partial: array<f32>, blocks: u32): void {
  for (let i: u32 = 0; i < out.length; i++) {
    let total: f32 = 0;
    for (let k: u32 = 0; k < blocks; k++) total += partial[i * blocks + k];
    out[i] = total;
  }
}

// Summary remains resident. Only three floats need cross the host boundary.
export function selectBest(out: array<f32>, stats: array<f32>, models: u32): void {
  for (let i: u32 = 0; i < out.length; i++) {
    let winner: u32 = 0;
    let count: f32 = stats[0];
    let cost: f32 = stats[models];
    for (let m: u32 = 1; m < models; m++) {
      const c = stats[m];
      const v = stats[models + m];
      if (c > count || (c === count && v < cost)) {
        winner = m;
        count = c;
        cost = v;
      }
    }
    let value: f32 = f32(winner);
    if (i === 1) value = count;
    else if (i === 2) value = cost;
    out[i] = value;
  }
}

export function winningMask(
  out: array<f32>,
  flags: array<f32>,
  best: array<f32>,
  count: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = flags[u32(best[0]) * count + i];
  }
}
