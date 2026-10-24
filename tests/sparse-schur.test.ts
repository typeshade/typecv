import test from 'node:test';
import assert from 'node:assert/strict';
import { SparseSymmetric, factorSparseSchur } from '../src/reconstruction/sparse-schur.ts';
import { solveSystem } from '../src/geometry/numerics.ts';

test('dense upper storage materializes symmetric owned copies without changing addition order or budgets', () => {
  const dense = new SparseSymmetric(4, 5, true),
    sparse = new SparseSymmetric(4, 5),
    additions = [
      [2, 0, 1e16],
      [0, 2, -1e16],
      [2, 0, 1],
      [1, 1, 3],
      [1, 3, 0],
      [3, 1, -2],
      [2, 2, 4],
      [3, 3, 5],
    ] as const;
  for (const [i, j, value] of additions) {
    dense.add(i, j, value);
    sparse.add(i, j, value);
  }
  assert.equal(dense.entries, 5);
  assert.equal(dense.get(0, 2), 1);
  assert.equal(dense.get(2, 0), 1);
  assert.deepEqual(dense.dense(), sparse.dense());
  const before = dense.dense(),
    copy = dense.copy();
  assert.equal(copy.denseStorage, true);
  assert.equal(copy.entries, dense.entries);
  copy.add(0, 2, 7);
  assert.equal(copy.get(2, 0), 8);
  assert.equal(dense.get(2, 0), 1);
  dense.add(3, 1, 1);
  assert.equal(dense.get(1, 3), -1);
  assert.equal(copy.get(1, 3), -2);
  assert.equal(before[3 * 4 + 1], -2);
  before[0 * 4 + 2] = 100;
  before[2 * 4 + 0] = 200;
  assert.equal(dense.get(0, 2), 1);
  assert.equal(copy.get(2, 0), 8);
  assert.throws(() => dense.add(0, 0, 0), /budget/);
  assert.throws(() => copy.add(0, 0, 0), /budget/);
  assert.equal(dense.entries, 5);
  assert.equal(copy.entries, 5);
  for (const matrix of [dense, copy]) {
    const materialized = matrix.dense();
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) assert.equal(materialized[4 * i + j], matrix.get(i, j));
  }
});

test('sparse Cholesky solves and selected inverse entries match dense equations with fill and scaling', () => {
  const n = 19,
    A = new SparseSymmetric(n, 1000);
  for (let row = 0; row < 50; row++) {
    const indices = [...new Set([row % n, (row * 7 + 3) % n, (row + 11) % n])],
      values = indices.map((_, j) => Math.sin(1.2 * row + j) * 10 ** ((row % 3) - 1));
    for (let a = 0; a < indices.length; a++)
      for (let b = a; b < indices.length; b++)
        A.add(indices[a]!, indices[b]!, values[a]! * values[b]!);
  }
  for (let i = 0; i < n; i++) A.add(i, i, 1);
  const factor = factorSparseSchur(A, 1000)!,
    rhs = Float64Array.from({ length: n }, (_, i) => Math.cos(i)),
    expected = solveSystem(A.dense(), rhs)!,
    actual = factor.solve(rhs);
  assert.ok(actual.every((x, i) => Math.abs(x - expected[i]!) < 1e-10));
  assert.equal(factor.selectedEntries, 0, 'optimization must not build the inverse');
  for (let i = 0; i < n; i++) {
    const unit = new Float64Array(n);
    unit[i] = 1;
    const column = solveSystem(A.dense(), unit)!;
    for (const j of A.rows[i]!.keys())
      assert.ok(Math.abs(factor.covariance(i, j) - column[j]!) < 1e-10);
  }
  assert.equal(factor.selectedEntries, factor.entries);
});

test('banded selected inverse remains sparse and rejects budget/rank failure without truncation', () => {
  const n = 300,
    A = new SparseSymmetric(n, n * 3);
  for (let i = 0; i < n; i++) {
    A.add(i, i, 3);
    if (i + 1 < n) A.add(i, i + 1, -1);
  }
  const factor = factorSparseSchur(A, n * 3)!;
  assert.ok(factor.entries < n * 3);
  assert.ok(factor.covariance(150, 151) > 0);
  assert.throws(() => factor.covariance(0, n - 1), /outside the factor pattern/);
  assert.throws(() => factorSparseSchur(A, 10), /budget/);
  const singular = new SparseSymmetric(2, 3);
  singular.add(0, 0, 1);
  singular.add(0, 1, 1);
  singular.add(1, 1, 1);
  assert.equal(factorSparseSchur(singular, 3), undefined);
  const small = new SparseSymmetric(2, 1);
  small.add(0, 0, 1);
  assert.throws(() => small.add(1, 1, 1), /budget/);
});
