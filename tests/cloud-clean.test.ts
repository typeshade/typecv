// Verifies: docs/dense-mvs.md section 3.12 (the CPU reference of the dense cloud cleanup).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildPointGrid,
  cleanDenseCloud,
  dbscan,
  decodeOctahedral,
  encodeOctahedral,
  hashCell,
  keepLargestClusters,
  meanNeighbourDistances,
  medianPointSpacing,
  nearestSquaredDistances,
  outlierMask,
  removeStatisticalOutliers,
  selectClusters,
  voxelDownsample,
  type CleanCloud,
} from '../src/reconstruction/cloud-clean.ts';

/** A small deterministic generator (mulberry32) so a failure reproduces. */
function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function cloudOf(points: number[][], color = 128): CleanCloud {
  const positions = new Float32Array(points.length * 3);
  points.forEach((p, i) => positions.set(p, 3 * i));
  return { count: points.length, positions, colors: new Uint8Array(points.length * 3).fill(color) };
}

/** A jittered plane z = 0 of `side` × `side` points at `spacing`, then `outliers` points scattered. */
export function planeWithOutliers(side = 40, spacing = 0.01, outliers = 30, seed = 7) {
  const r = random(seed);
  const points: number[][] = [];
  for (let y = 0; y < side; y++)
    for (let x = 0; x < side; x++)
      points.push([
        x * spacing + (r() - 0.5) * 0.2 * spacing,
        y * spacing + (r() - 0.5) * 0.2 * spacing,
        (r() - 0.5) * 0.2 * spacing,
      ]);
  const planeCount = points.length;
  for (let o = 0; o < outliers; o++)
    points.push([r() * side * spacing, r() * side * spacing, 0.15 + r() * 0.5]);
  return { cloud: cloudOf(points), planeCount };
}

test('the grid finds the same k nearest distances as a brute-force scan', () => {
  const r = random(3);
  // A dense blob, a thin slab and far singletons: the rings, the empty cells and the fallback.
  const points: number[][] = [];
  for (let i = 0; i < 400; i++) points.push([r(), r(), r()]);
  for (let i = 0; i < 300; i++) points.push([5 + r() * 2, r() * 0.01, r() * 2]);
  points.push([100, 100, 100], [-80, 40, 7]);
  const cloud = cloudOf(points);
  const k = 7;
  for (const cellSize of [0.05, 0.3, 4]) {
    const grid = buildPointGrid(cloud.positions, cellSize);
    const best = new Float64Array(k);
    for (let s = 0; s < grid.count; s++) {
      nearestSquaredDistances(grid, s, best);
      const d: number[] = [];
      for (let j = 0; j < grid.count; j++) {
        if (j === s) continue;
        d.push(
          (grid.points[3 * j]! - grid.points[3 * s]!) ** 2 +
            (grid.points[3 * j + 1]! - grid.points[3 * s + 1]!) ** 2 +
            (grid.points[3 * j + 2]! - grid.points[3 * s + 2]!) ** 2,
        );
      }
      d.sort((a, b) => a - b);
      for (let m = 0; m < k; m++) assert.equal(best[m], d[m], `cell ${cellSize} point ${s} #${m}`);
    }
  }
});

test('the grid keeps every point once and the input order is recoverable', () => {
  const { cloud } = planeWithOutliers(12, 0.1, 5);
  const grid = buildPointGrid(cloud.positions, 0.2);
  assert.equal(new Set(grid.order).size, cloud.count);
  for (let s = 0; s < grid.count; s++) {
    assert.equal(grid.rank[grid.order[s]!], s);
    for (let a = 0; a < 3; a++)
      assert.equal(grid.points[3 * s + a], cloud.positions[3 * grid.order[s]! + a]);
  }
  assert.ok(hashCell(1, 2, 3) < 2 ** 32);
});

test('no cell is below 1, and a far search finds every neighbour the brute-force scan finds', () => {
  // The points of the minimum on an axis sit at exactly one cell from the origin, which rounding
  // can turn into 0.99999. Such a point was in cell 0, and the lookups skip cells below 1.
  const rnd = random(2024);
  const positions = new Float32Array(3 * 3000);
  for (let i = 0; i < positions.length; i++) positions[i] = (rnd() - 0.5) * 30;
  for (const cellSize of [0.0862430750350345, 0.37, 1.1, 3.3]) {
    const grid = buildPointGrid(positions, cellSize);
    for (let i = 0; i < grid.cells.length; i++) assert.ok(grid.cells[i]! >= 1, `cell ${i}`);
    // A ring limit far past the farthest ring needed must give the brute-force answer too.
    const k = 8,
      ring = new Float64Array(k),
      full = new Float64Array(k);
    for (let s = 0; s < grid.count; s += 37) {
      nearestSquaredDistances(grid, s, ring, 1000);
      nearestSquaredDistances(grid, s, full, 0);
      assert.deepEqual(ring, full, `point ${s}, cell size ${cellSize}`);
    }
  }
});

test('median point spacing of a regular lattice is its pitch', () => {
  const points: number[][] = [];
  for (let z = 0; z < 8; z++)
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) points.push([x, y, z]);
  assert.equal(medianPointSpacing(cloudOf(points).positions), 1);
  assert.equal(medianPointSpacing(new Float32Array([1, 2, 3])), 0);
});

test('statistical outlier removal drops the scattered points and keeps the plane', () => {
  const { cloud, planeCount } = planeWithOutliers();
  const result = removeStatisticalOutliers(cloud, { k: 16, stdRatio: 2 });
  // Every outlier is above the plane. The plane's own points are all kept (their mean distance
  // is nearly constant), except possibly a few at the border.
  const z = result.cloud.positions;
  let outliersLeft = 0,
    planeLeft = 0;
  for (let i = 0; i < result.cloud.count; i++) z[3 * i + 2]! > 0.05 ? outliersLeft++ : planeLeft++;
  assert.equal(outliersLeft, 0);
  assert.ok(planeLeft >= planeCount - 40, `kept ${planeLeft} of ${planeCount} plane points`);
  assert.equal(result.removed, cloud.count - result.cloud.count);
  assert.equal(result.cloud.colors!.length, 3 * result.cloud.count);
});

test('the outlier mask follows mean + ratio · sample σ exactly', () => {
  const distances = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10];
  const mean = 1.9,
    std = Math.sqrt(distances.reduce((s, d) => s + (d - mean) ** 2, 0) / 9);
  const mask = outlierMask(distances, 2);
  assert.ok(Math.abs(mask.mean - mean) < 1e-12 && Math.abs(mask.std - std) < 1e-12);
  assert.equal(mask.threshold, mean + 2 * std);
  assert.deepEqual(Array.from(mask.keep), [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]);
  // At the threshold itself a point stays.
  assert.equal(outlierMask([0, 2], 0).keep.join(), '1,0');
});

test('mean neighbour distance of a lattice with k = 6 is the pitch', () => {
  const points: number[][] = [];
  for (let z = 0; z < 5; z++)
    for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) points.push([x, y, z]);
  const d = meanNeighbourDistances(cloudOf(points).positions, 6);
  // The interior point (2, 2, 2) has six neighbours at distance 1.
  assert.equal(d[2 * 25 + 2 * 5 + 2], 1);
  assert.ok(d.every((v) => v >= 1));
});

test('voxel counts are exact: 3 × 3 × 3 voxels of 2 × 2 × 2 lattice points', () => {
  const points: number[][] = [];
  // 6 × 6 × 6 lattice points at pitch 1; voxel size 2 holds 8 points each, 27 voxels.
  for (let z = 0; z < 6; z++)
    for (let y = 0; y < 6; y++)
      for (let x = 0; x < 6; x++) points.push([x + 0.25, y + 0.25, z + 0.25]);
  const colors = new Uint8Array(points.length * 3);
  points.forEach((_, i) => colors.set([i % 256, 255 - (i % 256), 7], 3 * i));
  const cloud = { ...cloudOf(points), colors };
  const { cloud: down, voxelSize } = voxelDownsample(cloud, { voxelSize: 2 });
  assert.equal(voxelSize, 2);
  assert.equal(down.count, 27);
  // The first voxel holds lattice points 0 and 1 on each axis: its mean is (1.0 + 0.25 - 0.5).
  assert.deepEqual(Array.from(down.positions.slice(0, 3)), [0.75, 0.75, 0.75]);
  // Every voxel averages 8 points: the sum of positions is preserved.
  const sum = [0, 0, 0],
    sumIn = [0, 0, 0];
  for (let i = 0; i < 27; i++)
    for (let a = 0; a < 3; a++) sum[a]! += down.positions[3 * i + a]! * 8;
  for (let i = 0; i < 216; i++)
    for (let a = 0; a < 3; a++) sumIn[a]! += cloud.positions[3 * i + a]!;
  for (let a = 0; a < 3; a++) assert.ok(Math.abs(sum[a]! - sumIn[a]!) < 1e-3);
  // The third channel is constant 7 and survives the mean.
  assert.ok(
    Array.from(down.colors!)
      .filter((_, i) => i % 3 === 2)
      .every((c) => c === 7),
  );
  // A larger voxel merges all: one point at the centroid.
  assert.equal(voxelDownsample(cloud, { voxelSize: 100 }).cloud.count, 1);
  // A default voxel size comes from the spacing and keeps fewer points.
  assert.ok(voxelDownsample(cloud).cloud.count < 216);
  assert.throws(() => voxelDownsample(cloud, { voxelSize: 1e-9 }), RangeError);
});

test('voxel downsampling averages normals and keeps unit length', () => {
  const [a, b] = encodeOctahedral(0, 0, 1);
  const normals = new Int16Array([a, b, a, b]);
  const cloud: CleanCloud = {
    count: 2,
    positions: new Float32Array([0, 0, 0, 0.1, 0, 0]),
    colors: new Uint8Array([10, 20, 30, 20, 30, 40]),
    normals,
  };
  const { cloud: down } = voxelDownsample(cloud, { voxelSize: 1 });
  assert.equal(down.count, 1);
  assert.deepEqual(Array.from(down.colors!), [15, 25, 35]);
  const n = decodeOctahedral(down.normals![0]!, down.normals![1]!);
  assert.ok(Math.abs(n[2]! - 1) < 1e-3);
});

test('octahedral encode inverts decode', () => {
  const r = random(11);
  for (let i = 0; i < 200; i++) {
    const v = [r() - 0.5, r() - 0.5, r() - 0.5];
    const l = Math.hypot(v[0]!, v[1]!, v[2]!);
    const [px, py] = encodeOctahedral(v[0]! / l, v[1]! / l, v[2]! / l);
    const d = decodeOctahedral(px, py);
    for (let a = 0; a < 3; a++) assert.ok(Math.abs(d[a]! - v[a]! / l) < 1e-3);
  }
});

function blob(center: number[], count: number, radius: number, r: () => number): number[][] {
  const out: number[][] = [];
  while (out.length < count) {
    const p = [r() * 2 - 1, r() * 2 - 1, r() * 2 - 1];
    if (Math.hypot(p[0]!, p[1]!, p[2]!) <= 1) out.push(p.map((v, a) => center[a]! + v * radius));
  }
  return out;
}

test('DBSCAN labels two separated blobs and the noise between them', () => {
  const r = random(5);
  const points = [
    ...blob([0, 0, 0], 200, 0.3, r),
    ...blob([5, 0, 0], 120, 0.3, r),
    [2.5, 3, 0],
    [2.5, -3, 1],
  ];
  const result = dbscan(cloudOf(points).positions, { eps: 0.15, minPoints: 5 });
  assert.equal(result.clusters, 2);
  assert.deepEqual(Array.from(result.sizes).sort((a, b) => b - a).length, 2);
  const first = result.labels[0]!,
    second = result.labels[200]!;
  assert.deepEqual([first, second], [0, 1]);
  for (let i = 0; i < 200; i++) assert.ok(result.labels[i] === 0 || result.labels[i] === -1);
  for (let i = 200; i < 320; i++) assert.ok(result.labels[i] === 1 || result.labels[i] === -1);
  assert.equal(result.labels[320], -1);
  assert.equal(result.labels[321], -1);
  assert.ok(result.sizes[0]! > 150 && result.sizes[1]! > 80);
  // The largest cluster alone is the first blob; two clusters keep both.
  const largest = keepLargestClusters(cloudOf(points), { eps: 0.15, minPoints: 5, keep: 1 });
  assert.equal(largest.cloud.count, result.sizes[0]);
  const both = keepLargestClusters(cloudOf(points), { eps: 0.15, minPoints: 5, keep: 2 });
  assert.equal(both.cloud.count, result.sizes[0]! + result.sizes[1]!);
  // minFraction keeps the second blob (120 / 200 ≥ 0.5) and not above 0.9.
  assert.deepEqual(selectClusters(result.sizes, { keep: 1, minFraction: 0.5 }), [0, 1]);
  assert.deepEqual(selectClusters(result.sizes, { keep: 1, minFraction: 0.9 }), [0]);
});

test('DBSCAN follows the scikit-learn definition on a line', () => {
  // Points at 0, 1, 2, 3 and a far point. eps = 1, min_samples = 3 (the point included):
  // points 1 and 2 are core (three points within 1), 0 and 3 are border, 10 is noise.
  const cloud = cloudOf([
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 0],
    [3, 0, 0],
    [10, 0, 0],
  ]);
  const result = dbscan(cloud.positions, { eps: 1, minPoints: 3 });
  assert.deepEqual(Array.from(result.labels), [0, 0, 0, 0, -1]);
  assert.equal(result.corePoints, 2);
  // min_samples = 4: no core point, every point is noise.
  const none = dbscan(cloud.positions, { eps: 1, minPoints: 4 });
  assert.deepEqual(Array.from(none.labels), [-1, -1, -1, -1, -1]);
  assert.equal(keepLargestClusters(cloud, { eps: 1, minPoints: 4 }).cloud.count, 5);
  // A border point within eps of a core point of each of two clusters takes the lower label.
  const line = (xs: number[]) => xs.map((x) => [x, 0, 0]);
  const tie = cloudOf(line([-0.8, -0.6, -0.4, -0.2, 0, 2, 2.2, 2.4, 2.6, 2.8, 1]));
  assert.deepEqual(
    Array.from(dbscan(tie.positions, { eps: 1, minPoints: 5 }).labels),
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
  );
  // eps is inclusive (distance ≤ eps).
  assert.equal(
    dbscan(
      cloudOf([
        [0, 0, 0],
        [1, 0, 0],
      ]).positions,
      { eps: 1, minPoints: 2 },
    ).clusters,
    1,
  );
});

test('the pipeline cleans a plane with outliers and reports each stage', async () => {
  const { cloud, planeCount } = planeWithOutliers(50, 0.01, 60, 21);
  const { cloud: clean, report } = await cleanDenseCloud(cloud);
  assert.equal(report.tier, 'cpu');
  assert.equal(report.before, cloud.count);
  assert.ok(report.afterOutliers < report.before && report.afterOutliers >= planeCount - 60);
  assert.ok(report.afterVoxel < report.afterOutliers && report.voxelSize > 0);
  assert.ok(report.after <= report.afterVoxel && report.clusters >= 1 && report.eps > 0);
  assert.equal(report.after, clean.count);
  for (let i = 0; i < clean.count; i++) assert.ok(clean.positions[3 * i + 2]! < 0.05);
  // Each stage turns off by itself.
  const off = await cleanDenseCloud(cloud, { outliers: false, voxel: false, cluster: false });
  assert.equal(off.cloud, cloud);
  assert.equal(off.report.after, cloud.count);
  const only = await cleanDenseCloud(cloud, { voxel: false, cluster: false });
  assert.equal(only.report.afterVoxel, only.report.afterOutliers);
  assert.equal(only.report.after, only.report.afterOutliers);
});

test('degenerate clouds pass through', async () => {
  const empty = cloudOf([]);
  assert.equal((await cleanDenseCloud(empty)).cloud.count, 0);
  const one = cloudOf([[1, 2, 3]]);
  assert.equal((await cleanDenseCloud(one)).cloud.count, 1);
  const same = cloudOf(Array.from({ length: 30 }, () => [1, 1, 1]));
  const out = await cleanDenseCloud(same);
  assert.ok(out.cloud.count >= 1);
  assert.throws(() => removeStatisticalOutliers(same, { k: 0 }), RangeError);
  assert.throws(() => removeStatisticalOutliers(same, { k: 33 }), RangeError);
  assert.throws(() => dbscan(same.positions, { eps: 0 }), RangeError);
});
