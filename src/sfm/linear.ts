/** Internal f64 matrix machinery. All matrices are row-major. No runtime dependencies. */
export function identity(n: number): Float64Array {
  const m = new Float64Array(n * n);
  for (let i = 0; i < n; i++) m[i * n + i] = 1;
  return m;
}

export function multiply(a: ArrayLike<number>, b: ArrayLike<number>, n: number): Float64Array {
  const m = new Float64Array(n * n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += a[r * n + k]! * b[k * n + c]!;
    m[r * n + c] = s;
  }
  return m;
}

export function transpose(a: ArrayLike<number>, n: number): Float64Array {
  const result = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) result[i * n + j] = a[j * n + i]!;
  return result;
}

/** Jacobi diagonalization of real symmetric n × n matrix; ascending eigenvalues. */
export function symmetricEigen(matrix: ArrayLike<number>, n: number): {
  values: Float64Array;
  vectors: Float64Array;
} {
  const a = Float64Array.from(matrix);
  if (a.length !== n * n) throw new RangeError('Invalid symmetric matrix dimensions');
  const v = identity(n);
  const sweeps = 120 * n * n;
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let p = 0, q = 1, maxOff = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const magnitude = Math.abs(a[i * n + j]!);
      if (magnitude > maxOff) { maxOff = magnitude; p = i; q = j; }
    }
    const maxDiagonal = Math.max(1, ...Array.from({ length: n }, (_, i) => Math.abs(a[i * n + i]!)));
    if (maxOff < 1e-14 * maxDiagonal) break;
    const apq = a[p * n + q]!;
    const tau = (a[q * n + q]! - a[p * n + p]!) / (2 * apq);
    const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t), s = t * c;
    const app = a[p * n + p]!, aqq = a[q * n + q]!;
    a[p * n + p] = app - t * apq;
    a[q * n + q] = aqq + t * apq;
    a[p * n + q] = a[q * n + p] = 0;
    for (let k = 0; k < n; k++) {
      if (k !== p && k !== q) {
        const akp = a[k * n + p]!, akq = a[k * n + q]!;
        a[k * n + p] = a[p * n + k] = c * akp - s * akq;
        a[k * n + q] = a[q * n + k] = s * akp + c * akq;
      }
      const vkp = v[k * n + p]!, vkq = v[k * n + q]!;
      v[k * n + p] = c * vkp - s * vkq;
      v[k * n + q] = s * vkp + c * vkq;
    }
  }
  const permutation = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i * n + i]! - a[j * n + j]!);
  const values = new Float64Array(n), vectors = new Float64Array(n * n);
  for (let col = 0; col < n; col++) {
    const src = permutation[col]!;
    values[col] = a[src * n + src]!;
    for (let row = 0; row < n; row++) vectors[row * n + col] = v[row * n + src]!;
  }
  return { values, vectors };
}

function normalize3(v: number[]): number[] {
  const len = Math.hypot(...v);
  if (!(len > 1e-14)) throw new Error('Degenerate 3D basis');
  return v.map(x => x / len);
}
function cross3(a: number[], b: number[]): number[] {
  return [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
}
function dot3(a: number[], b: number[]): number { return a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!; }

/** Small 3×3 SVD E = U diag(s) Vt; columns of U,V are right-handed. */
export function svd3(a: ArrayLike<number>): {
  U: Float64Array; S: Float64Array; Vt: Float64Array;
} {
  const ata = multiply(transpose(a, 3), a, 3);
  const eig = symmetricEigen(ata, 3);
  const v0 = normalize3([eig.vectors[2]!, eig.vectors[5]!, eig.vectors[8]!]);
  const v1raw = [eig.vectors[1]!, eig.vectors[4]!, eig.vectors[7]!];
  const projection = dot3(v0, v1raw);
  const v1 = normalize3(v1raw.map((x, i) => x - projection * v0[i]!));
  const v2 = cross3(v0, v1);
  const V = Float64Array.from([v0[0],v1[0],v2[0], v0[1],v1[1],v2[1], v0[2],v1[2],v2[2]]);
  const S = Float64Array.of(Math.sqrt(Math.max(0, eig.values[2]!)), Math.sqrt(Math.max(0, eig.values[1]!)), Math.sqrt(Math.max(0, eig.values[0]!)));
  if (S[1]! < 1e-12 * S[0]!) throw new Error('Essential matrix lacks rank two');
  const product = multiply(a, V, 3);
  const u0 = normalize3([product[0]! / S[0]!,product[3]! / S[0]!,product[6]! / S[0]!]);
  const u1raw = [product[1]! / S[1]!,product[4]! / S[1]!,product[7]! / S[1]!];
  const proj = dot3(u0, u1raw);
  const u1 = normalize3(u1raw.map((x,i) => x - proj * u0[i]!));
  const u2 = cross3(u0, u1);
  const U = Float64Array.from([u0[0],u1[0],u2[0],u0[1],u1[1],u2[1],u0[2],u1[2],u2[2]]);
  return { U, S, Vt: transpose(V, 3) };
}

export function determinant3(m: ArrayLike<number>): number {
  return m[0]! * (m[4]! * m[8]! - m[5]! * m[7]!) - m[1]! * (m[3]! * m[8]! - m[5]! * m[6]!) + m[2]! * (m[3]! * m[7]! - m[4]! * m[6]!);
}
