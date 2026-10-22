import test from 'node:test';
import assert from 'node:assert/strict';
import { nullspace } from '../src/geometry/numerics.ts';

// Frozen from f7013d62420d63ea3a05522589dc84ddaffebbfc:src/geometry/numerics.ts.
// Keep the legacy row storage and arithmetic independent of the optimized implementation.
function legacyNullspace(
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

function compareLegacy(rows: readonly (readonly number[])[], columns: number, label: string) {
  const input = Object.freeze(rows.map((row) => Object.freeze([...row]))),
    snapshot = structuredClone(input);
  let expected: ReturnType<typeof nullspace>;
  try {
    expected = legacyNullspace(input, columns);
  } catch (error) {
    assert.ok(error instanceof Error);
    assert.throws(
      () => nullspace(input, columns),
      (actual) =>
        actual instanceof Error &&
        actual.constructor === error.constructor &&
        actual.message === error.message,
      label,
    );
    assert.deepEqual(input, snapshot, label + ' preserves rejected input');
    return false;
  }
  assert.deepEqual(input, snapshot, label + ' legacy preserves input');
  const actual = nullspace(input, columns);
  assert.ok(actual.vector instanceof Float64Array);
  assert.ok(actual.singularValues instanceof Float64Array);
  // Compare bytes, including signed zeros, rather than a numerical tolerance.
  assert.deepEqual(
    new Uint8Array(actual.vector.buffer),
    new Uint8Array(expected.vector.buffer),
    label + ' vector bits',
  );
  assert.deepEqual(
    new Uint8Array(actual.singularValues.buffer),
    new Uint8Array(expected.singularValues.buffer),
    label + ' singular value bits',
  );
  assert.deepEqual(input, snapshot, label + ' preserves input');
  return true;
}

function design(rowCount: number, columns: number, seed: number) {
  let state = seed;
  return Array.from({ length: rowCount }, (_, i) =>
    Array.from({ length: columns }, (_, j) => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return (state / 2 ** 32 - 0.5) * 2 + Math.sin(i * 0.7 + j * 0.3);
    }),
  );
}

for (const columns of [4, 9, 12]) {
  for (const rowCount of [columns - 1, columns, columns * 3 + 1]) {
    test(`flat nullspace matches legacy bits for ${rowCount}x${columns} fits`, () => {
      const base = design(rowCount, columns, 731 + rowCount + columns);
      const cases = [
        { name: 'general', rows: base },
        {
          name: 'rank deficient',
          rows: base.map((r) => r.map((v, j) => (j === columns - 1 ? r[0]! * 2 : v))),
        },
        { name: 'repeated rows', rows: base.map((_r, i) => [...base[i % 2]!]) },
        {
          name: 'near collinear',
          rows: base.map((r, i) =>
            r.map((v, j) => (j === columns - 1 ? r[0]! * (1 + 1e-10) + 1e-12 * Math.sin(i) : v)),
          ),
        },
        {
          name: 'dynamic range',
          rows: base.map((r) => r.map((v, j) => v * 10 ** (-12 * (j % 4)))),
        },
        {
          name: 'repeated singular values',
          rows: base.map((_r, i) => Array.from({ length: columns }, (_, j) => (i === j ? 1 : 0))),
        },
      ];
      let successful = 0;
      for (const entry of cases)
        for (const scale of [1, -7, 1e-120, 1e120]) {
          if (
            compareLegacy(
              entry.rows.map((r) => r.map((x) => x * scale)),
              columns,
              `${entry.name}, scale ${scale}`,
            )
          )
            successful++;
        }
      assert.ok(successful >= 4, 'includes successful results rather than only matching failures');
    });
  }
}

test('flat nullspace preserves legacy validation errors and rejected inputs', () => {
  const valid = design(4, 4, 119);
  const invalid = [
    { name: 'too few rows', rows: valid.slice(0, 2), columns: 4 },
    { name: 'too few columns', rows: [[1]], columns: 1 },
    { name: 'wrong row width', rows: valid.map((r, i) => (i ? r : r.slice(0, 3))), columns: 4 },
    {
      name: 'NaN',
      rows: valid.map((r, i) => r.map((x, j) => (i === 1 && j === 2 ? NaN : x))),
      columns: 4,
    },
    {
      name: 'infinity',
      rows: valid.map((r, i) => r.map((x, j) => (i === 2 && j === 1 ? Infinity : x))),
      columns: 4,
    },
    { name: 'zero matrix', rows: valid.map((r) => r.map(() => 0)), columns: 4 },
    { name: 'signed zero matrix', rows: valid.map((r) => r.map(() => -0)), columns: 4 },
  ];
  for (const entry of invalid)
    assert.equal(compareLegacy(entry.rows, entry.columns, entry.name), false, entry.name);
});
