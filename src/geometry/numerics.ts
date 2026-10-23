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
  const a = new Float64Array(rows.length * columns),
    v = new Float64Array(columns * columns);
  for (let i = 0; i < rows.length; i++)
    for (let j = 0; j < columns; j++) a[i * columns + j] = rows[i]![j]! / scale;
  for (let i = 0; i < columns; i++) v[i * columns + i] = 1;
  let converged = false;
  for (let sweep = 0; sweep < 160; sweep++) {
    let changed = false;
    for (let p = 0; p < columns; p++)
      for (let q = p + 1; q < columns; q++) {
        let aa = 0,
          bb = 0,
          ab = 0;
        for (let at = 0; at < a.length; at += columns) {
          aa += a[at + p]! * a[at + p]!;
          bb += a[at + q]! * a[at + q]!;
          ab += a[at + p]! * a[at + q]!;
        }
        if (Math.abs(ab) <= 1e-13 * Math.sqrt(aa * bb) || aa < 1e-30 || bb < 1e-30) continue;
        const tau = (bb - aa) / (2 * ab),
          t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.hypot(1, tau));
        const c = 1 / Math.hypot(1, t),
          s = c * t;
        for (let at = 0; at < a.length; at += columns) {
          const x = a[at + p]!,
            y = a[at + q]!;
          a[at + p] = c * x - s * y;
          a[at + q] = s * x + c * y;
        }
        for (let at = 0; at < v.length; at += columns) {
          const x = v[at + p]!,
            y = v[at + q]!;
          v[at + p] = c * x - s * y;
          v[at + q] = s * x + c * y;
        }
        changed = true;
      }
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) throw new Error('Projective SVD did not converge');
  const values = Array.from({ length: columns }, (_, j) => {
    let squared = 0;
    for (let at = 0; at < a.length; at += columns) squared += a[at + j]! * a[at + j]!;
    return Math.sqrt(squared);
  });
  const order = values.map((s, i) => ({ s, i })).sort((a, b) => a.s - b.s);
  return {
    vector: Float64Array.from({ length: columns }, (_, i) => v[i * columns + order[0]!.i]!),
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

/** Closed-form 3x3 inverse by the adjugate; undefined for a singular or nonfinite matrix. */
export function inverse3(a: ArrayLike<number>): Float64Array | undefined {
  if (a.length !== 9) throw new RangeError('Invalid 3x3 matrix');
  const a00 = a[0]!,
    a01 = a[1]!,
    a02 = a[2]!,
    a10 = a[3]!,
    a11 = a[4]!,
    a12 = a[5]!,
    a20 = a[6]!,
    a21 = a[7]!,
    a22 = a[8]!;
  const c00 = a11 * a22 - a12 * a21,
    c01 = a02 * a21 - a01 * a22,
    c02 = a01 * a12 - a02 * a11;
  const det = a00 * c00 + a10 * c01 + a20 * c02;
  if (!Number.isFinite(det) || det === 0) return undefined;
  const out = Float64Array.of(
    c00 / det,
    c01 / det,
    c02 / det,
    (a12 * a20 - a10 * a22) / det,
    (a00 * a22 - a02 * a20) / det,
    (a02 * a10 - a00 * a12) / det,
    (a10 * a21 - a11 * a20) / det,
    (a01 * a20 - a00 * a21) / det,
    (a00 * a11 - a01 * a10) / det,
  );
  return out.every(Number.isFinite) ? out : undefined;
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
