import assert from 'node:assert/strict';
import { affine, gridPoints, renderScene } from '../klt-cases.ts';

export function compactInput() {
  const width = 160,
    height = 120,
    points = Array.from(gridPoints(width, height, 35, 30, 5, 4)).slice(0, 34),
    warp = { degrees: 9, scale: 1.1 },
    theta = (warp.degrees * Math.PI) / 180,
    matrix = [
      warp.scale * Math.cos(theta),
      -warp.scale * Math.sin(theta),
      warp.scale * Math.sin(theta),
      warp.scale * Math.cos(theta),
    ],
    transform = affine(width, height, warp),
    initialTargets = Array.from({ length: points.length / 2 }, (_, i) =>
      transform.forward(points[2 * i], points[2 * i + 1]),
    )
      .flat()
      .map((v, i) => v + (i % 2 ? -0.25 : 0.4));
  return {
    width,
    height,
    points,
    first: Array.from(renderScene({ width, height })),
    shifted: Array.from(renderScene({ width, height }, { dx: 2.5, dy: -1.5 })),
    warped: Array.from(renderScene({ width, height }, warp)),
    affineWarps: Array.from({ length: 2 * points.length }, (_, i) => matrix[i % 4]),
    initialTargets,
  };
}

export function assertCompact(result, input) {
  const n = input.points.length / 2;
  assert.equal(result.liveBytes, 0);
  assert.equal(result.rows.length, 6);
  for (const row of result.rows) {
    const old = row.legacy,
      compact = row.compact;
    assert.equal(old.output.ncc, null, row.name + ': legacy NCC API unchanged');
    assert.deepEqual(compact.output.records, old.output.records, row.name + ': checked records');
    assert.deepEqual(compact.output.back, old.output.back, row.name + ': optional backward output');
    assert.deepEqual(
      compact.output.pyramids,
      old.output.pyramids,
      row.name + ': pyramid snapshots',
    );
    assert.deepEqual(compact.output.forward, [], row.name + ': omit forward read');
    assert.deepEqual(
      compact.output.ncc,
      old.output.forward.filter((_, i) => i % 4 === 2),
      row.name + ': exact forward NCC copied to checked tail',
    );
    if (!row.count) {
      assert.equal(old.counts.reads, 0);
      assert.equal(compact.counts.reads, 0);
      assert.deepEqual(compact.output.ncc, []);
      assert.equal(compact.dispatches, 0);
      continue;
    }
    assert.equal(compact.counts.reads, old.counts.reads - 1, row.name + ': one fewer read');
    assert.equal(compact.counts.waves, 1);
    assert.equal(old.counts.bytes - compact.counts.bytes, 12 * n, row.name + ': logical bytes');
    assert.equal(
      old.counts.snapshots - compact.counts.snapshots,
      (row.affine ? 28 : 12) * n,
      row.name + ': full Resident snapshot bytes',
    );
    assert.equal(compact.output.records.length, 4 * n);
    assert.equal(compact.output.ncc.length, n);
    assert.equal(
      compact.sharedSnapshot,
      true,
      'records/NCC retain only their independent read snapshot',
    );
  }
  const batch = result.rows.find((row) => row.name === 'affine batch'),
    ended = result.rows.find((row) => row.name === 'ended forward tracks');
  assert.equal(batch.legacy.counts.reads, 2);
  assert.equal(batch.compact.counts.reads, 1);
  assert.equal(batch.legacy.counts.bytes, 32 * n);
  assert.equal(batch.legacy.counts.snapshots, 48 * n);
  assert.equal(batch.compact.counts.bytes, 20 * n);
  assert.equal(batch.compact.counts.snapshots, 20 * n);
  assert.ok(
    batch.compact.output.records.some((v, i) => i % 4 === 3 && (v & 1) === 1),
    'affine transport parity must include accepted image measurements',
  );
  assert.ok(
    batch.legacy.output.forward.some((v, i) => i % 4 === 2 && v > 0.8),
    'the NCC comparison must exercise nonzero photometric evidence',
  );
  assert.ok(
    ended.compact.output.records.some((v, i) => i % 4 === 3 && (v & 1) === 0),
    'the failure fixture must exercise ended records, not only accepted tracks',
  );
}
