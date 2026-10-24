import test from 'node:test';
import assert from 'node:assert/strict';
import { SparseSymmetric, factorSparseSchur } from '../src/reconstruction/sparse-schur.ts';
import { solveSystem } from '../src/geometry/numerics.ts';

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
