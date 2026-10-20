'use typeshade';

// The k nearest neighbour distances of a dense point cloud (`docs/dense-mvs.md`, section 3.12),
// for the statistical outlier removal. The host sorts the points by the bucket of their grid cell
// (`buildPointGrid` in `src/reconstruction/cloud-clean.ts`). One invocation takes one sorted point,
// visits the rings of cells around its cell until the k-th distance is inside the ring's inner
// radius, and writes the mean of the k distances. Past `maxRing` it scans every point. A lane past
// its chunk takes the chunk's last point, runs the same control flow and writes nothing.
//
// - `gridPoints`: f32 × 3 per sorted point.
// - `gridCells`: u32 × 3 per sorted point, the cell coordinates (at least 1) from the host.
// - `bucketStart`: u32, `mask + 2` offsets of the buckets into the sorted points.
// - `knnMean`: f32 per sorted point, the mean distance to its k nearest neighbours.

class CloudControl {
  count: u32;
  k: u32;
  mask: u32;
  maxRing: u32;
  cellSize: f32;
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
declare const knnMean: storage<array<f32>, 'read_write'>;

const NO_DISTANCE: f32 = 1e30;

/** The bucket of a cell: `hashCell` of `src/reconstruction/cloud-clean.ts`. */
function cellBucket(x: u32, y: u32, z: u32): u32 {
  let h: u32 = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
  h = h ^ (h >> 16);
  h = h * 0x7feb352d;
  h = h ^ (h >> 15);
  h = h * 0x846ca68b;
  return h ^ (h >> 16);
}

/** `list` (ascending, k used) with `d2` inserted; the k-th value falls off. */
function insertDistance(list: array<f32, 32>, d2: f32, k: u32): array<f32, 32> {
  let out: array<f32, 32> = list;
  let s: u32 = k - 1;
  let moving: bool = true;
  for (let step: u32 = 0; step < 31; step++) {
    if (moving && s > 0 && out[s - 1] > d2) {
      out[s] = out[s - 1];
      s = s - 1;
    } else {
      moving = false;
    }
  }
  out[s] = d2;
  return out;
}

@compute([64])
export function knnMeanDistance(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = cloudRange.first + gid.x;
  const active: bool = i < cloudRange.end;
  if (!active) i = max(cloudRange.end, 1) - 1;
  const k: u32 = cloudControl.k;
  const px: f32 = gridPoints[i * 3];
  const py: f32 = gridPoints[i * 3 + 1];
  const pz: f32 = gridPoints[i * 3 + 2];
  const cx: i32 = i32(gridCells[i * 3]);
  const cy: i32 = i32(gridCells[i * 3 + 1]);
  const cz: i32 = i32(gridCells[i * 3 + 2]);
  let best: array<f32, 32> = fill<f32, 32>(NO_DISTANCE);
  let done: bool = false;
  for (let r: u32 = 0; r <= cloudControl.maxRing; r++) {
    if (!done) {
      const ring: i32 = i32(r);
      for (let dz: i32 = -ring; dz <= ring; dz++) {
        for (let dy: i32 = -ring; dy <= ring; dy++) {
          for (let dx: i32 = -ring; dx <= ring; dx++) {
            const nx: i32 = cx + dx;
            const ny: i32 = cy + dy;
            const nz: i32 = cz + dz;
            if (max(abs(dx), max(abs(dy), abs(dz))) === ring && nx >= 1 && ny >= 1 && nz >= 1) {
              const bucket: u32 =
                cellBucket(u32(nx), u32(ny), u32(nz)) & cloudControl.mask;
              const end: u32 = bucketStart[bucket + 1];
              for (let j: u32 = bucketStart[bucket]; j < end; j++) {
                if (
                  j !== i &&
                  gridCells[j * 3] === u32(nx) &&
                  gridCells[j * 3 + 1] === u32(ny) &&
                  gridCells[j * 3 + 2] === u32(nz)
                ) {
                  const ex: f32 = gridPoints[j * 3] - px;
                  const ey: f32 = gridPoints[j * 3 + 1] - py;
                  const ez: f32 = gridPoints[j * 3 + 2] - pz;
                  const d2: f32 = ex * ex + ey * ey + ez * ez;
                  if (d2 < best[k - 1]) best = insertDistance(best, d2, k);
                }
              }
            }
          }
        }
      }
      const reach: f32 = f32(r) * cloudControl.cellSize;
      if (best[k - 1] <= reach * reach) done = true;
    }
  }
  if (!done) {
    best = fill<f32, 32>(NO_DISTANCE);
    for (let j: u32 = 0; j < cloudControl.count; j++) {
      if (j !== i) {
        const ex: f32 = gridPoints[j * 3] - px;
        const ey: f32 = gridPoints[j * 3 + 1] - py;
        const ez: f32 = gridPoints[j * 3 + 2] - pz;
        const d2: f32 = ex * ex + ey * ey + ez * ez;
        if (d2 < best[k - 1]) best = insertDistance(best, d2, k);
      }
    }
  }
  let sum: f32 = 0.;
  for (let m: u32 = 0; m < k; m++) sum += sqrt(best[m]);
  if (active) knnMean[i] = sum / f32(k);
}
