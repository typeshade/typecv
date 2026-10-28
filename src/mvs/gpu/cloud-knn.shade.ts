'use typeshade';

// The k nearest neighbour distances of a dense point cloud (`docs/dense-mvs.md`, section 3.12),
// for the statistical outlier removal. The host sorts the points by the bucket of their grid cell
// (`buildPointGrid` in `src/reconstruction/cloud-clean.ts`). One invocation takes one sorted point,
// visits the rings of cells around its cell until the k-th distance is inside the ring's inner
// radius, and writes the mean of the k distances. A row or a cell whose nearest corner is farther
// than the k-th distance is skipped. Past `maxRing` it scans every point. A lane past
// its chunk takes the chunk's last point, runs the same control flow and writes nothing.
//
// - `gridPoints`: f32 × 3 per sorted point.
// - `gridCells`: u32 × 3 per sorted point, the cell coordinates (at least 1) from the host.
// - `visitOrder`: u32 per lane, the sorted point of the lane (cell order, so that lanes of a group
//   work on points that are close).
// - `bucketStart`: u32, `mask + 2` offsets of the buckets into the sorted points.
// - `knnMean`: f32 per sorted point, the mean distance to its k nearest neighbours.

class CloudControl {
  count: u32;
  k: u32;
  mask: u32;
  maxRing: u32;
  cellSize: f32;
  originX: f32;
  originY: f32;
  originZ: f32;
}
class CloudRange {
  first: u32;
  end: u32;
}
declare const cloudControl: uniform<CloudControl>;
declare const cloudRange: uniform<CloudRange>;
declare const gridPoints: storage<array<f32>>;
declare const gridCells: storage<array<u32>>;
declare const bucketStart: storage<array<u32>>;
declare const visitOrder: storage<array<u32>>;
declare const knnMean: storage<array<f32>, 'read_write'>;

const NO_DISTANCE: f32 = 1e30;
/** A cell is skipped only when its lower bound exceeds the k-th distance by this margin. */
const PRUNE_SAFETY: f32 = 0.999;

/** The bucket of a cell: `hashCell` of `src/reconstruction/cloud-clean.ts`. */
function cellBucket(x: u32, y: u32, z: u32): u32 {
  let h: u32 = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
  h = h ^ (h >> 16);
  h = h * 0x7feb352d;
  h = h ^ (h >> 15);
  h = h * 0x846ca68b;
  return h ^ (h >> 16);
}

/** The gap along one axis, in cells, from a point at `f` inside its cell to the cell `d` away. */
function cellGap(d: i32, f: f32): f32 {
  if (d > 0) return f32(d) - f;
  if (d < 0) return f + f32(-d) - 1.;
  return 0.;
}

/** Insert `d2` in place into the `k` smallest squared distances of `best`: the k-th falls off. */
function insertSorted(@inout best: array<f32, 32>, k: u32, d2: f32): void {
  let s: u32 = k - 1;
  while (s > 0 && best[s - 1] > d2) {
    best[s] = best[s - 1];
    s = s - 1;
  }
  best[s] = d2;
}

/** Insert the points of cell (`x`, `y`, `z`) other than point `i` that are closer than the k-th. */
function scanBucket(
  @inout best: array<f32, 32>,
  k: u32,
  i: u32,
  px: f32,
  py: f32,
  pz: f32,
  x: u32,
  y: u32,
  z: u32,
): void {
  const bucket: u32 = cellBucket(x, y, z) & cloudControl.mask;
  const end: u32 = bucketStart[bucket + 1];
  for (let j: u32 = bucketStart[bucket]; j < end; j++) {
    if (j !== i && gridCells[j * 3] === x && gridCells[j * 3 + 1] === y && gridCells[j * 3 + 2] === z) {
      const ex: f32 = gridPoints[j * 3] - px;
      const ey: f32 = gridPoints[j * 3 + 1] - py;
      const ez: f32 = gridPoints[j * 3 + 2] - pz;
      const d2: f32 = ex * ex + ey * ey + ez * ez;
      if (d2 < best[k - 1]) insertSorted(best, k, d2);
    }
  }
}

/** The mean of the square roots of the `k` smallest squared distances. */
function meanOf(@inout best: array<f32, 32>, k: u32): f32 {
  let sum: f32 = 0.;
  for (let m: u32 = 0; m < k; m++) sum += sqrt(best[m]);
  return sum / f32(k);
}

@compute([64])
export function knnMeanDistance(@builtin('global_invocation_id') gid: vec3u): void {
  let lane: u32 = cloudRange.first + gid.x;
  const active: bool = lane < cloudRange.end;
  if (!active) lane = max(cloudRange.end, 1) - 1;
  const i: u32 = visitOrder[lane];
  const k: u32 = cloudControl.k;
  const h: f32 = cloudControl.cellSize;
  const px: f32 = gridPoints[i * 3];
  const py: f32 = gridPoints[i * 3 + 1];
  const pz: f32 = gridPoints[i * 3 + 2];
  const cx: i32 = i32(gridCells[i * 3]);
  const cy: i32 = i32(gridCells[i * 3 + 1]);
  const cz: i32 = i32(gridCells[i * 3 + 2]);
  // The position inside the own cell, in cells (0 to 1) on each axis.
  const fx: f32 = clamp((px - cloudControl.originX) / h - f32(cx), 0., 1.);
  const fy: f32 = clamp((py - cloudControl.originY) / h - f32(cy), 0., 1.);
  const fz: f32 = clamp((pz - cloudControl.originZ) / h - f32(cz), 0., 1.);
  let best: array<f32, 32> = fill<f32, 32>(NO_DISTANCE);
  for (let r: u32 = 0; r <= cloudControl.maxRing; r++) {
    const ring: i32 = i32(r);
    for (let dz: i32 = -ring; dz <= ring; dz++) {
      const gz: f32 = cellGap(dz, fz);
      for (let dy: i32 = -ring; dy <= ring; dy++) {
        const gy: f32 = cellGap(dy, fy);
        // A row whose nearest cell is farther than the k-th distance holds no closer point.
        // The factor leaves room for the rounding of the cell position.
        const rowGap: f32 = (gz * gz + gy * gy) * h * h * PRUNE_SAFETY;
        if (rowGap <= best[k - 1]) {
          const onShell: bool = abs(dz) === ring || abs(dy) === ring;
          const cellsInRow: i32 = onShell ? 2 * ring + 1 : 2;
          for (let t: i32 = 0; t < cellsInRow; t++) {
            const dx: i32 = onShell ? t - ring : t * 2 * ring - ring;
            const gx: f32 = cellGap(dx, fx);
            const gap: f32 = (gz * gz + gy * gy + gx * gx) * h * h * PRUNE_SAFETY;
            const nx: i32 = cx + dx;
            const ny: i32 = cy + dy;
            const nz: i32 = cz + dz;
            if (gap <= best[k - 1] && nx >= 1 && ny >= 1 && nz >= 1)
              scanBucket(best, k, i, px, py, pz, u32(nx), u32(ny), u32(nz));
          }
        }
      }
    }
    // The k-th distance is inside the ring's inner radius: no farther cell holds a closer point.
    const reach: f32 = f32(r) * h;
    if (best[k - 1] <= reach * reach) {
      if (active) knnMean[i] = meanOf(best, k);
      return;
    }
  }
  best = fill<f32, 32>(NO_DISTANCE);
  for (let j: u32 = 0; j < cloudControl.count; j++) {
    if (j !== i) {
      const ex: f32 = gridPoints[j * 3] - px;
      const ey: f32 = gridPoints[j * 3 + 1] - py;
      const ez: f32 = gridPoints[j * 3 + 2] - pz;
      const d2: f32 = ex * ex + ey * ey + ez * ez;
      if (d2 < best[k - 1]) insertSorted(best, k, d2);
    }
  }
  if (active) knnMean[i] = meanOf(best, k);
}
