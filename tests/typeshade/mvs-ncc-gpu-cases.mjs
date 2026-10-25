// Independent analytic patches and centered-f64 NCC on their exact uploaded Float32 samples.
// The cameras induce integer disparities +2/+4/+6; no shader estimates the ground truth.
export const NCC_ROW_FLOATS = 20;
const W = 64,
  H = 32,
  P = W * H,
  N = 4,
  X = 31,
  Y = 15,
  F = 32,
  CX = 31.5,
  CY = 15.5;
const mean = (a) => a.reduce((sum, x) => sum + x, 0) / a.length;
const offsets = Array.from({ length: 36 }, (_, i) => [(i % 6) * 2 - 5, Math.floor(i / 6) * 2 - 5]);
function normalized(a) {
  const mu = mean(a),
    spread = Math.sqrt(mean(a.map((x) => (x - mu) ** 2)));
  return a.map((x) => (x - mu) / spread);
}
const a = normalized(
  Array.from({ length: 36 }, (_, i) => Math.sin(0.73 * (i + 1)) + 0.37 * Math.cos(1.19 * (i + 1))),
);
let b = Array.from(
  { length: 36 },
  (_, i) => Math.cos(0.91 * (i + 1)) + 0.23 * Math.sin(1.37 * (i + 1)),
);
const bm = mean(b);
b = b.map((x) => x - bm);
const projection = mean(b.map((x, i) => x * a[i]));
b = normalized(b.map((x, i) => x - projection * a[i]));

function oracle(reference, source, geometryValid) {
  const am = mean(reference),
    bm = mean(source),
    av = mean(reference.map((v) => (v - am) ** 2)),
    bv = mean(source.map((v) => (v - bm) ** 2)),
    covariance = mean(reference.map((v, i) => (v - am) * (source[i] - bm))),
    referenceOk = av >= 1e-6,
    sourceOk = bv >= 1e-6;
  return {
    referenceMean: am,
    referenceVariance: av,
    sourceVariance: bv,
    referenceOk,
    sourceOk,
    cost:
      geometryValid && referenceOk && sourceOk
        ? Math.max(0, Math.min(2, 1 - covariance / Math.sqrt(av * bv)))
        : 2,
  };
}

function fixture(
  id,
  {
    brightness = 0.4,
    offset = 0.05,
    referenceSigma = 0.00101,
    sourceSigma = referenceSigma,
    rho = 1,
    semantic,
  } = {},
) {
  const cameras = new Float32Array(N * 24),
    gray = new Float32Array(N * P),
    sourceScale = ['center-hole', 'center-hole-opt-in', 'center-outlier'].includes(semantic)
      ? 2
      : 1;
  for (let s = 0; s < N; s++) {
    const center = -0.25 * s,
      f = F * (s ? sourceScale : 1);
    cameras.set(
      [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        -center,
        0,
        0,
        f,
        f,
        semantic === 'center-out' && s ? -100 : CX,
        CY,
        W,
        H,
        3,
        5,
        center,
        0,
        0,
        s,
      ],
      s * 24,
    );
    gray.fill(brightness + (s ? offset : 0), s * P, (s + 1) * P);
  }
  const reference = a.map((x) => Math.fround(brightness + referenceSigma * x)),
    source = a.map((x, i) =>
      Math.fround(brightness + offset + sourceSigma * (rho * x + Math.sqrt(1 - rho * rho) * b[i])),
    );
  offsets.forEach(([dx, dy], k) => {
    gray[(Y + dy) * W + X + dx] = reference[k];
    for (let s = 1; s < N; s++) {
      const u = sourceScale * (X + dx - CX) + CX + 2 * sourceScale * s,
        v = sourceScale * (Y + dy - CY) + CY,
        x = Math.floor(u),
        y = Math.floor(v);
      // Four equal bilinear cells also preserve the fixture at half-integer scaled coordinates.
      for (const oy of [0, 1])
        for (const ox of [0, 1]) gray[s * P + (y + oy) * W + x + ox] = source[k];
    }
  });
  if (semantic === 'center-hole' || semantic === 'center-hole-opt-in') {
    const u = 2 * (X - CX) + CX + 4,
      v = 2 * (Y - CY) + CY;
    gray[P + Math.floor(v) * W + Math.floor(u)] = -1;
  }
  if (semantic === 'center-outlier') {
    // The sparse default lattice never samples these four bilinear center cells.
    // A valid but unrepresentative anchor must not cancel the tiny sample variance.
    for (let s = 1; s < N; s++) {
      const u = 2 * (X - CX) + CX + 4 * s,
        v = 2 * (Y - CY) + CY;
      for (const oy of [0, 1])
        for (const ox of [0, 1]) gray[s * P + (Math.floor(v) + oy) * W + Math.floor(u) + ox] = 0.1;
    }
  }
  const expected = oracle(reference, source, semantic !== 'center-out');
  expected.sourceCosts = Array(3).fill(expected.cost);
  if (semantic === 'center-hole-opt-in') expected.sourceCosts[0] = 2;
  expected.aggregate = mean(expected.sourceCosts);
  return {
    id,
    parameters: { brightness, offset, referenceSigma, sourceSigma, rho, semantic },
    expected,
    mvsControl: {
      width: W,
      height: H,
      reference: 0,
      state: 0,
      color: 0,
      iteration: 0,
      seed: 0,
      bestSources: 3,
      windowRadius: 5,
      windowStep: 2,
      axisSamples: 6,
      minSamples: 9,
      bilateral: 0,
      boundaryAware: 0,
      minFiniteGain: semantic === 'center-hole-opt-in' ? 0.01 : 0,
      spatialFalloff: 0.02,
      colorFalloff: 12.5,
      minAngleCos: Math.cos(Math.PI / 180),
      perturbDepth: 1,
      perturbAngle: 0,
      geometric: 0,
      geometricWeight: 0.3,
      geometricMaxCost: 3,
      view: 0,
      relativePoseCache: 0,
    },
    mvsCameras: Array.from(cameras),
    mvsGray: Array.from(gray),
    mvsSources: [3, 1, 2, 3, 0, 0, 0],
    mvsDepth: Array(N * P).fill(0),
    hypothesis: [X, Y, 4, 0, 0, -1, 0, 0],
  };
}

export function nccGpuFixtures() {
  // Includes the observed source-variance cancellation: 1.0201e-6 becomes 9.5367e-7 on SwiftShader.
  const rows = [fixture('near-floor-offset')];
  for (const brightness of [0.1, 0.4, 0.8])
    for (const rho of [1, 0, -0.5])
      if (!(brightness === 0.4 && rho === 1))
        rows.push(fixture(`bright-${brightness}-rho-${rho}`, { brightness, rho }));
  rows.push(fixture('normal-contrast', { referenceSigma: 0.01, rho: 0.9 }));
  for (const [name, referenceSigma, sourceSigma] of [
    ['reference-below-floor', 0.0009, 0.003],
    ['source-below-floor', 0.003, 0.0009],
    ['both-below-floor', 0.0009, 0.0009],
    ['constant-reference', 0, 0.003],
    ['constant-source', 0.003, 0],
  ])
    rows.push(fixture(name, { referenceSigma, sourceSigma, rho: 0.9 }));
  rows.push(fixture('center-hole', { referenceSigma: 0.01, semantic: 'center-hole' }));
  rows.push(
    fixture('center-hole-opt-in', { referenceSigma: 0.01, semantic: 'center-hole-opt-in' }),
  );
  rows.push(fixture('center-out', { referenceSigma: 0.01, semantic: 'center-out' }));
  rows.push(fixture('center-outlier', { brightness: 0.8, semantic: 'center-outlier' }));
  rows.push(nccWeightedOutlierFixture());
  return rows;
}

export function nccWeightedOutlierFixture() {
  const row = fixture('weighted-first-sample-outlier'),
    gray = new Float32Array(N * P),
    reference = [],
    source = [],
    points = Array.from({ length: 25 }, (_, i) => [(i % 5) * 2 - 4, Math.floor(i / 5) * 2 - 4]);
  for (let s = 0; s < N; s++) gray.fill(s ? 0.15 : 0.1, s * P, (s + 1) * P);
  for (let k = 0; k < points.length; k++) {
    reference[k] = Math.fround(k === 0 ? 0.8 : k === 12 ? 0.1 : 0.1 + 0.0015 * a[k]);
    source[k] = Math.fround(
      k === 0 ? 0.85 : k === 12 ? 0.15 : 0.15 + 0.0015 * (0.5 * a[k] + Math.sqrt(0.75) * b[k]),
    );
    const [dx, dy] = points[k];
    gray[(Y + dy) * W + X + dx] = reference[k];
    for (let s = 1; s < N; s++)
      for (const oy of [0, 1])
        for (const ox of [0, 1]) gray[s * P + (Y + dy + oy) * W + X + dx + 2 * s + ox] = source[k];
  }
  const weights = points.map(([dx, dy], k) =>
      Math.exp(-0.02 * (dx * dx + dy * dy) - 50 * (reference[k] - reference[12]) ** 2),
    ),
    weight = weights.reduce((s, w) => s + w, 0),
    weightedMean = (v) => v.reduce((sum, x, k) => sum + weights[k] * x, 0) / weight,
    am = weightedMean(reference),
    bm = weightedMean(source),
    av = weightedMean(reference.map((x) => (x - am) ** 2)),
    bv = weightedMean(source.map((x) => (x - bm) ** 2)),
    covariance = weightedMean(reference.map((x, k) => (x - am) * (source[k] - bm))),
    cost = Math.max(0, Math.min(2, 1 - covariance / Math.sqrt(av * bv)));
  Object.assign(row.mvsControl, {
    boundaryAware: 1,
    bilateral: 1,
    axisSamples: 5,
    minSamples: 7,
    colorFalloff: 50,
  });
  row.mvsGray = Array.from(gray);
  row.expected = {
    referenceMean: am,
    referenceVariance: av,
    sourceVariance: bv,
    referenceOk: av >= 1e-6,
    sourceOk: bv >= 1e-6,
    cost,
    sourceCosts: [cost, cost, cost],
    aggregate: cost,
  };
  row.parameters = {
    role: 'weighted arithmetic stress',
    firstReferenceSample: reference[0],
    firstSourceSample: source[0],
    firstSampleWeight: weights[0],
    totalWeight: weight,
    sourcePatternMixCoefficient: 0.5,
    weightedCorrelation: 1 - cost,
  };
  return row;
}

export function legacyNccControl(source) {
  const pattern = /function stableNcc\(\): bool \{[^}]*\}/;
  if ((source.match(/function stableNcc\(/g) ?? []).length !== 1)
    throw new Error('NCC control site changed');
  // Only the numeric formulation changes, with default feature flags still false/0.
  return source.replace(pattern, 'function stableNcc(): bool { return false; }');
}

export const NCC_PROBE = `
declare const nccProbeHypothesis: storage<array<f32>>;
declare const nccProbeOutput: storage<array<f32>, 'read_write'>;
@compute([1])
export function nccGpuProbe(): void {
  const x: u32 = u32(nccProbeHypothesis[0]);
  const y: u32 = u32(nccProbeHypothesis[1]);
  const depth: f32 = nccProbeHypothesis[2];
  const n: vec3 = vec3(nccProbeHypothesis[3], nccProbeHypothesis[4], nccProbeHypothesis[5]);
  const stats: MvsWindow = referenceWindow(i32(x), i32(y));
  let patch: MvsReferencePatch = {values: fill<f32, 121>(0.), weights: fill<f32, 121>(0.)};
  if (stats.ok !== 0) prepareReferencePatch(patch, i32(x), i32(y), stats.center);
  nccProbeOutput[0] = stats.mean;
  nccProbeOutput[1] = stats.variance;
  nccProbeOutput[2] = stats.weight;
  nccProbeOutput[3] = stats.center;
  nccProbeOutput[4] = f32(stats.ok);
  nccProbeOutput[5] = hypothesisCost(x, y, depth, n, stats);
  nccProbeOutput[6] = hypothesisCostCached(patch, x, y, depth, n, stats);
  const count: u32 = min(mvsSources[0], 6);
  nccProbeOutput[7] = f32(count);
  const X: vec3 = referenceRay(f32(x), f32(y)) * depth;
  for (let k: u32 = 0; k < 6; k++) {
    let scalar: f32 = NO_COST;
    let cached: f32 = NO_COST;
    if (k < count && stats.ok !== 0) {
      const s: u32 = mvsSources[1 + k];
      scalar = sourceCost(s, i32(x), i32(y), X, n, stats);
      cached = sourceCostCached(patch, s, i32(x), i32(y), X, n, stats);
    }
    nccProbeOutput[8 + k] = scalar;
    nccProbeOutput[14 + k] = cached;
  }
}
`;
