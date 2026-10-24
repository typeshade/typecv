/** Symmetric scalar sparsity; structural zeros are retained for selected inverse closure. */
export class SparseSymmetric {
  readonly rows: Map<number, number>[];
  entries = 0;
  readonly size: number;
  readonly maxEntries: number;
  /** Dense backing accumulates the upper triangle; dense() expands an owned symmetric copy. */
  private values?: Float64Array;
  private pattern?: Uint8Array;
  get denseStorage(): boolean {
    return !!this.values;
  }
  constructor(size: number, maxEntries: number, dense = false) {
    this.size = size;
    this.maxEntries = maxEntries;
    this.rows = Array.from({ length: size }, () => new Map<number, number>());
    if (dense) {
      this.values = new Float64Array(size * size);
      this.pattern = new Uint8Array(size * size);
    }
  }
  add(i: number, j: number, value: number): void {
    if (i > j) [i, j] = [j, i];
    if (this.values) {
      const at = i * this.size + j;
      if (!this.pattern![at]) {
        if (this.entries >= this.maxEntries) throw new RangeError('Schur nonzero budget exceeded');
        this.pattern![at] = 1;
        this.entries++;
      }
      this.values[at] += value;
      return;
    }
    const row = this.rows[i]!;
    if (!row.has(j)) {
      if (this.entries >= this.maxEntries) throw new RangeError('Schur nonzero budget exceeded');
      this.entries++;
    }
    row.set(j, (row.get(j) ?? 0) + value);
  }
  get(i: number, j: number): number {
    if (i > j) [i, j] = [j, i];
    if (this.values) return this.values[i * this.size + j]!;
    return this.rows[i]!.get(j) ?? 0;
  }
  copy(): SparseSymmetric {
    const out = new SparseSymmetric(this.size, this.maxEntries);
    if (this.values) {
      out.values = this.values.slice();
      out.pattern = this.pattern!.slice();
      out.entries = this.entries;
      return out;
    }
    for (let i = 0; i < this.size; i++)
      for (const [j, value] of this.rows[i]!) out.add(i, j, value);
    return out;
  }
  dense(): Float64Array {
    if (this.values) {
      const out = this.values.slice();
      for (let i = 0; i < this.size; i++)
        for (let j = i + 1; j < this.size; j++) out[j * this.size + i] = out[i * this.size + j]!;
      return out;
    }
    const out = new Float64Array(this.size * this.size);
    for (let i = 0; i < this.size; i++)
      for (const [j, v] of this.rows[i]!) out[i * this.size + j] = out[j * this.size + i] = v;
    return out;
  }
}

export interface SchurFactor {
  readonly storage: 'dense' | 'sparse';
  readonly entries: number;
  readonly selectedEntries: number;
  solve(rhs: Float64Array): Float64Array;
  /** Only entries in the Schur/fill pattern are requested. No absent value is treated as zero. */
  covariance(i: number, j: number): number;
}
interface Column {
  rows: Int32Array;
  values: Float64Array;
}

/** Exact sparse Cholesky with greedy minimum-degree ordering, diagonal equilibration and
 * Takahashi selected inversion. No camera is fixed/removed to meet the storage budget.
 * Limits count logical entries, not JS Map overhead or process RSS.
 */
export function factorSparseSchur(
  A: SparseSymmetric,
  maxFactorEntries: number,
): SchurFactor | undefined {
  if (A.denseStorage) throw new Error('Sparse factor requires sparse matrix storage');
  if (A.entries > maxFactorEntries) throw new RangeError('Schur factor nonzero budget exceeded');
  const n = A.size,
    scale = Float64Array.from({ length: n }, (_, i) => Math.sqrt(A.get(i, i)));
  if (scale.some((v) => !(v > 0 && Number.isFinite(v)))) return undefined;
  const work = Array.from({ length: n }, () => new Map<number, number>()),
    diagonal = new Float64Array(n).fill(1);
  for (let i = 0; i < n; i++)
    for (const [j, value] of A.rows[i]!)
      if (i !== j) {
        const v = value / (scale[i]! * scale[j]!);
        work[i]!.set(j, v);
        work[j]!.set(i, v);
      }
  const order = new Int32Array(n),
    position = new Int32Array(n).fill(-1),
    pivots = new Float64Array(n),
    columns: Column[] = [];
  let entries = n,
    structuralEntries = A.entries;
  if (entries > maxFactorEntries) throw new RangeError('Schur factor nonzero budget exceeded');
  for (let step = 0; step < n; step++) {
    let pivot = -1,
      degree = Infinity;
    for (let i = 0; i < n; i++)
      if (position[i] === -1 && work[i]!.size < degree) {
        degree = work[i]!.size;
        pivot = i;
      }
    if (pivot < 0 || !(diagonal[pivot]! > 1e-10 && Number.isFinite(diagonal[pivot])))
      return undefined;
    order[step] = pivot;
    position[pivot] = step;
    const root = Math.sqrt(diagonal[pivot]!),
      neighbors = [...work[pivot]!.keys()].sort((a, b) => a - b),
      values = Float64Array.from(neighbors, (j) => work[pivot]!.get(j)! / root);
    entries += neighbors.length;
    if (entries > maxFactorEntries) throw new RangeError('Schur factor nonzero budget exceeded');
    pivots[step] = root;
    columns.push({ rows: Int32Array.from(neighbors), values });
    for (let a = 0; a < neighbors.length; a++) {
      const i = neighbors[a]!,
        li = values[a]!;
      diagonal[i] -= li * li;
      work[i]!.delete(pivot);
      for (let b = a + 1; b < neighbors.length; b++) {
        const j = neighbors[b]!,
          old = work[i]!.get(j);
        if (old === undefined && ++structuralEntries > maxFactorEntries)
          throw new RangeError('Schur fill nonzero budget exceeded');
        const v = (old ?? 0) - li * values[b]!;
        work[i]!.set(j, v);
        work[j]!.set(i, v);
      }
    }
    work[pivot]!.clear();
  }
  let selected: Map<number, number>[] | undefined;
  const rawCovariance = (i: number, j: number): number => {
    if (position[i]! > position[j]!) [i, j] = [j, i];
    const value = selected![i]!.get(j);
    if (value === undefined)
      throw new Error('Selected inverse request is outside the factor pattern');
    return value;
  };
  const select = () => {
    if (selected) return;
    selected = Array.from({ length: n }, () => new Map<number, number>());
    for (let step = n - 1; step >= 0; step--) {
      const pivot = order[step]!,
        column = columns[step]!,
        root = pivots[step]!;
      for (const j of column.rows) {
        let value = 0;
        for (let k = 0; k < column.rows.length; k++)
          value -= (column.values[k]! * rawCovariance(column.rows[k]!, j)) / root;
        selected[pivot]!.set(j, value);
      }
      let value = 1 / (root * root);
      for (let k = 0; k < column.rows.length; k++)
        value -= (column.values[k]! * rawCovariance(pivot, column.rows[k]!)) / root;
      if (!(value > 0 && Number.isFinite(value)))
        throw new RangeError('Selected Schur inverse is numerically invalid');
      selected[pivot]!.set(pivot, value);
    }
  };
  return {
    storage: 'sparse',
    entries,
    get selectedEntries() {
      return selected ? entries : 0;
    },
    solve(rhs) {
      const x = rhs.map((v, i) => v / scale[i]!);
      for (let step = 0; step < n; step++) {
        const pivot = order[step]!,
          column = columns[step]!;
        x[pivot] /= pivots[step]!;
        for (let j = 0; j < column.rows.length; j++)
          x[column.rows[j]!] -= column.values[j]! * x[pivot]!;
      }
      for (let step = n - 1; step >= 0; step--) {
        const pivot = order[step]!,
          column = columns[step]!;
        for (let j = 0; j < column.rows.length; j++)
          x[pivot] -= column.values[j]! * x[column.rows[j]!]!;
        x[pivot] /= pivots[step]!;
      }
      return x.map((v, i) => v / scale[i]!);
    },
    covariance(i, j) {
      select();
      return rawCovariance(i, j) / (scale[i]! * scale[j]!);
    },
  };
}
