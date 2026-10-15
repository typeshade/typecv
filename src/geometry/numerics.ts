/** Small f64 numerical primitives. No normal equations in projective null-space fits. */
export function nullspace(
  rows: readonly (readonly number[])[],
  columns: number,
): { vector: Float64Array; singularValues: Float64Array } {
  if (
    rows.length < columns - 1 ||
    columns < 2 ||
    rows.some((r) => r.length !== columns || !r.every(Number.isFinite))
  )
    throw new RangeError('Invalid projective design matrix');
  let scale = 0;
  for (const row of rows) for (const x of row) scale = Math.max(scale, Math.abs(x));
  if (!(scale > 0)) throw new Error('Zero design matrix');
  const a = rows.map((r) => Float64Array.from(r, (x) => x / scale));
  const v = Array.from({ length: columns }, (_, i) =>
    Float64Array.from({ length: columns }, (_, j) => (i === j ? 1 : 0)),
  );
  let converged = false;
  for (let sweep = 0; sweep < 160; sweep++) {
    let changed = false;
    for (let p = 0; p < columns; p++)
      for (let q = p + 1; q < columns; q++) {
        let aa = 0,
          bb = 0,
          ab = 0;
        for (const row of a) {
          aa += row[p]! * row[p]!;
          bb += row[q]! * row[q]!;
          ab += row[p]! * row[q]!;
        }
        if (Math.abs(ab) <= 1e-13 * Math.sqrt(aa * bb) || aa < 1e-30 || bb < 1e-30) continue;
        const tau = (bb - aa) / (2 * ab),
          t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.hypot(1, tau));
        const c = 1 / Math.hypot(1, t),
          s = c * t;
        for (const matrix of [a, v])
          for (const row of matrix) {
            const x = row[p]!,
              y = row[q]!;
            row[p] = c * x - s * y;
            row[q] = s * x + c * y;
          }
        changed = true;
      }
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) throw new Error('Projective SVD did not converge');
  const values = Array.from({ length: columns }, (_, j) =>
    Math.sqrt(a.reduce((s, r) => s + r[j]! * r[j]!, 0)),
  );
  const order = values.map((s, i) => ({ s, i })).sort((a, b) => a.s - b.s);
  return {
    vector: Float64Array.from(v, (row) => row[order[0]!.i]!),
    singularValues: Float64Array.from(order, (x) => x.s * scale),
  };
}

/** Pivoted elimination; returns undefined for a singular/nonfinite system. Inputs are preserved. */
export function solveSystem(
  matrix: ArrayLike<number>,
  rhs: ArrayLike<number>,
): Float64Array | undefined {
  const n = rhs.length;
  if (matrix.length !== n * n) throw new RangeError('Invalid linear system dimensions');
  const a = Array.from({ length: n }, (_, i) =>
    Float64Array.from([...Array.from({ length: n }, (_, j) => matrix[i * n + j]!), rhs[i]!]),
  );
  if (a.some((row) => !row.every(Number.isFinite))) return undefined;
  const scale = Math.max(0, ...a.map((row) => Math.max(...row.subarray(0, n).map(Math.abs))));
  if (!(scale > 0)) return n === 0 ? new Float64Array() : undefined;
  for (let j = 0; j < n; j++) {
    let p = j;
    for (let i = j + 1; i < n; i++) if (Math.abs(a[i]![j]!) > Math.abs(a[p]![j]!)) p = i;
    if (Math.abs(a[p]![j]!) < 1e-14 * scale) return undefined;
    [a[p], a[j]] = [a[j]!, a[p]!];
    for (let i = j + 1; i < n; i++) {
      const f = a[i]![j]! / a[j]![j]!;
      for (let k = j; k <= n; k++) a[i]![k] -= f * a[j]![k]!;
    }
  }
  const out = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let b = a[i]![n]!;
    for (let j = i + 1; j < n; j++) b -= a[i]![j]! * out[j]!;
    out[i] = b / a[i]![i]!;
  }
  return out.every(Number.isFinite) ? out : undefined;
}

export function inverse3(a: ArrayLike<number>): Float64Array | undefined {
  const out = new Float64Array(9);
  for (let j = 0; j < 3; j++) {
    const b = new Float64Array(3);
    b[j] = 1;
    const x = solveSystem(a, b);
    if (!x) return undefined;
    for (let i = 0; i < 3; i++) out[i * 3 + j] = x[i]!;
  }
  return out;
}

export function seededRandom(seed: number): () => number {
  let s = seed >>> 0 || 0x9e3779b9;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 2 ** 32;
  };
}
export function sample(n: number, count: number, rng: () => number): number[] {
  if (count > n) throw new RangeError('Insufficient sample population');
  const out = new Set<number>();
  while (out.size < count) out.add(Math.floor(rng() * n));
  return [...out];
}
export function ransacBudget(
  inliers: number,
  total: number,
  size: number,
  confidence: number,
): number {
  const p = (inliers / total) ** size;
  return p >= 1 ? 1 : p <= 0 ? Infinity : Math.ceil(Math.log1p(-confidence) / Math.log1p(-p));
}
