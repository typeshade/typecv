/**
 * Cleanup of a fused dense point cloud (`docs/dense-mvs.md`, section 3.12): statistical outlier
 * removal, voxel downsampling and DBSCAN clustering, each on a uniform-grid spatial hash. This is
 * the CPU reference of every filter. The GPU variants in `src/gpu/cloud-clean.ts` replace the two
 * steps that pay on a GPU (the k nearest neighbour distances and the voxel binning) and share
 * everything else with this file, so a result differs only where the two tiers round differently.
 *
 * The semantics follow the libraries the options are named after:
 * - Statistical outlier removal is PCL's `StatisticalOutlierRemoval`: the mean distance of a
 *   point to its `k` nearest neighbours (the point itself is not one), the mean and the sample
 *   standard deviation σ of those means over the cloud, and a point stays when its mean distance
 *   is at most mean + `stdRatio` · σ. Open3D's `remove_statistical_outlier(nb_neighbors, ...)`
 *   counts the point itself among its neighbours, so `nb_neighbors = k + 1` there is the same
 *   filter.
 * - Voxel downsampling is Open3D's `voxel_down_sample`: one point per occupied voxel at the mean
 *   position and the mean color. The normal is the normalized mean of the normals.
 * - DBSCAN is scikit-learn's `DBSCAN(eps, min_samples)`: a point is core when at least
 *   `minPoints` points (itself included) lie within `eps` (distance ≤ eps), a cluster is a set of
 *   core points connected through such neighbourhoods plus the border points they reach, and a
 *   point in no cluster is noise (label −1). Clusters are numbered by their first core point in
 *   input order, and a border point takes the lowest-numbered cluster that reaches it, as
 *   scikit-learn does.
 */

/** A point cloud with the arrays `GpuDensePoints` of `typecv/gpu` carries. */
export interface CleanCloud {
  readonly count: number;
  /** x, y, z per point. */
  readonly positions: Float32Array;
  /** r, g, b per point. */
  readonly colors?: Uint8Array;
  /** The octahedral normal per point as two snorm16 values (the layout of `GpuDensePoints`). */
  readonly normals?: Int16Array;
}

// ---------------------------------------------------------------------------------------------
// Octahedral normals

const SNORM = 32767;

/** The unit normal of two snorm16 values of the octahedral map of `GpuDensePoints.normals`. */
export function decodeOctahedral(
  px16: number,
  py16: number,
  out: Float64Array | number[] = [0, 0, 0],
): Float64Array | number[] {
  const px = Math.max(px16 / SNORM, -1),
    py = Math.max(py16 / SNORM, -1);
  const z = 1 - Math.abs(px) - Math.abs(py),
    t = Math.max(-z, 0);
  const x = px + (px >= 0 ? -t : t),
    y = py + (py >= 0 ? -t : t);
  const length = Math.hypot(x, y, z) || 1;
  out[0] = x / length;
  out[1] = y / length;
  out[2] = z / length;
  return out;
}

/** The two snorm16 values of a unit normal, the inverse of `decodeOctahedral` up to rounding. */
export function encodeOctahedral(x: number, y: number, z: number): [number, number] {
  const l1 = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
  let px = x / l1,
    py = y / l1;
  if (z < 0) {
    const ox = px;
    px = (1 - Math.abs(py)) * (ox >= 0 ? 1 : -1);
    py = (1 - Math.abs(ox)) * (py >= 0 ? 1 : -1);
  }
  const q = (v: number) => Math.max(-SNORM, Math.min(SNORM, Math.round(v * SNORM)));
  return [q(px), q(py)];
}

// ---------------------------------------------------------------------------------------------
// The uniform-grid spatial hash

/** The largest cell coordinate of an axis. A grid raises its cell size to stay below it. */
export const GRID_MAX_CELL = (1 << 20) - 8;

/** The bucket hash of a cell: three prime multipliers and a 32-bit avalanche (`lowbias32`). */
export function hashCell(x: number, y: number, z: number): number {
  let h = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  h = Math.imul(h, 0x7feb352d) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  h = Math.imul(h, 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

/**
 * The points sorted by the bucket of their cell. A cell is a cube of side `cellSize`. The cell of
 * a point is `floor((p − origin) / cellSize)` on each axis, always at least 1. A bucket holds the
 * points of every cell that hashes to it, so a lookup of one cell compares `cells` too.
 */
export interface PointGrid {
  readonly count: number;
  readonly cellSize: number;
  readonly origin: readonly [number, number, number];
  /** `tableSize − 1`, the table size being a power of two. */
  readonly mask: number;
  /** `tableSize + 1` offsets: the points of bucket b are `bucketStart[b]..bucketStart[b + 1]`. */
  readonly bucketStart: Uint32Array;
  /** The input index of each sorted point. */
  readonly order: Uint32Array;
  /** The sorted position of each input point. */
  readonly rank: Uint32Array;
  /** x, y, z per sorted point. */
  readonly points: Float32Array;
  /** The cell coordinates per sorted point. */
  readonly cells: Uint32Array;
}

export function boundingBox(positions: Float32Array): { min: number[]; max: number[] } {
  const min = [Infinity, Infinity, Infinity],
    max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3)
    for (let a = 0; a < 3; a++) {
      const v = positions[i + a]!;
      if (v < min[a]!) min[a] = v;
      if (v > max[a]!) max[a] = v;
    }
  return { min, max };
}

export function buildPointGrid(positions: Float32Array, cellSize: number): PointGrid {
  const n = positions.length / 3;
  const { min, max } = boundingBox(positions);
  const extent = n === 0 ? 0 : Math.max(max[0]! - min[0]!, max[1]! - min[1]!, max[2]! - min[2]!);
  let h = cellSize;
  if (!(h > 0)) h = extent > 0 ? extent / Math.max(1, Math.cbrt(n)) : 1;
  h = Math.max(h, extent / GRID_MAX_CELL);
  const origin = [min[0]! - h, min[1]! - h, min[2]! - h] as const;
  let table = 16;
  while (table < n) table *= 2;
  const mask = table - 1;
  const cells = new Uint32Array(3 * n),
    bucketOf = new Uint32Array(n),
    bucketStart = new Uint32Array(table + 1);
  for (let i = 0; i < n; i++) {
    // The rounding of `min − h` can put a point of the minimum on an axis at 0.99999 cells. The
    // lookups skip cells below 1, so the cell is raised to 1 and no point is lost to a search.
    const cx = Math.max(1, Math.floor((positions[3 * i]! - origin[0]) / h)),
      cy = Math.max(1, Math.floor((positions[3 * i + 1]! - origin[1]) / h)),
      cz = Math.max(1, Math.floor((positions[3 * i + 2]! - origin[2]) / h));
    cells[3 * i] = cx;
    cells[3 * i + 1] = cy;
    cells[3 * i + 2] = cz;
    const b = hashCell(cx, cy, cz) & mask;
    bucketOf[i] = b;
    bucketStart[b + 1]!++;
  }
  for (let b = 0; b < table; b++) bucketStart[b + 1]! += bucketStart[b]!;
  const fill = bucketStart.slice(0, table),
    order = new Uint32Array(n),
    rank = new Uint32Array(n),
    points = new Float32Array(3 * n),
    sortedCells = new Uint32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const s = fill[bucketOf[i]!]!++;
    order[s] = i;
    rank[i] = s;
    for (let a = 0; a < 3; a++) {
      points[3 * s + a] = positions[3 * i + a]!;
      sortedCells[3 * s + a] = cells[3 * i + a]!;
    }
  }
  return {
    count: n,
    cellSize: h,
    origin,
    mask,
    bucketStart,
    order,
    rank,
    points,
    cells: sortedCells,
  };
}

/**
 * The sorted points in the order of their cells, rows of x within rows of y within layers of z.
 * Lanes that are next to each other then work on points that are next to each other, so they
 * visit the same cells and leave their loops together. Only the order of the lanes changes: each
 * point still gets the distance of the same search.
 */
export function cellVisitOrder(cells: Uint32Array, count: number): Uint32Array {
  let order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;
  // Least significant digit first: x, then y, then z.
  for (let axis = 0; axis < 3; axis++) {
    let top = 0;
    for (let i = 0; i < count; i++) top = Math.max(top, cells[3 * i + axis]!);
    const offsets = new Uint32Array(top + 2);
    for (let i = 0; i < count; i++) offsets[cells[3 * i + axis]! + 1]!++;
    for (let c = 0; c <= top; c++) offsets[c + 1]! += offsets[c]!;
    const next = new Uint32Array(count);
    for (let s = 0; s < count; s++) {
      const point = order[s]!;
      next[offsets[cells[3 * point + axis]!]!++] = point;
    }
    order = next;
  }
  return order;
}

/** Calls `visit(j)` for each sorted point j of the cell (cx, cy, cz). */
export function eachInCell(
  grid: PointGrid,
  cx: number,
  cy: number,
  cz: number,
  visit: (j: number) => void,
): void {
  if (cx < 1 || cy < 1 || cz < 1) return;
  const b = hashCell(cx, cy, cz) & grid.mask;
  const end = grid.bucketStart[b + 1]!;
  const cells = grid.cells;
  for (let j = grid.bucketStart[b]!; j < end; j++)
    if (cells[3 * j] === cx && cells[3 * j + 1] === cy && cells[3 * j + 2] === cz) visit(j);
}

/** The farthest ring of cells a nearest-neighbour search visits before it scans every point. */
export const KNN_MAX_RING = 6;

/**
 * The squared distances of the `k` nearest points to sorted point `i`, ascending, with `i` itself
 * left out. Rings of cells grow around the point's cell until the k-th distance is within the
 * ring's inner radius, and past `maxRing` the search scans every point. `best` has `k` entries,
 * `Infinity` where fewer than k other points exist.
 */
export function nearestSquaredDistances(
  grid: PointGrid,
  i: number,
  best: Float64Array,
  maxRing = KNN_MAX_RING,
): void {
  const k = best.length;
  best.fill(Infinity);
  const p = grid.points;
  const px = p[3 * i]!,
    py = p[3 * i + 1]!,
    pz = p[3 * i + 2]!;
  const consider = (j: number) => {
    if (j === i) return;
    const dx = p[3 * j]! - px,
      dy = p[3 * j + 1]! - py,
      dz = p[3 * j + 2]! - pz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 >= best[k - 1]!) return;
    let s = k - 1;
    while (s > 0 && best[s - 1]! > d2) {
      best[s] = best[s - 1]!;
      s--;
    }
    best[s] = d2;
  };
  const cx = grid.cells[3 * i]!,
    cy = grid.cells[3 * i + 1]!,
    cz = grid.cells[3 * i + 2]!;
  for (let r = 0; r <= maxRing; r++) {
    for (let dz = -r; dz <= r; dz++)
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== r) continue;
          eachInCell(grid, cx + dx, cy + dy, cz + dz, consider);
        }
    const reach = r * grid.cellSize;
    if (best[k - 1]! <= reach * reach) return;
  }
  best.fill(Infinity);
  for (let j = 0; j < grid.count; j++) consider(j);
}

/** The median distance of a point to its nearest neighbour, from at most `sample` points. */
export function medianPointSpacing(positions: Float32Array, sample = 512): number {
  const n = positions.length / 3;
  if (n < 2) return 0;
  const { min, max } = boundingBox(positions);
  const floor = 1e-3 * Math.max(max[0]! - min[0]!, max[1]! - min[1]!, max[2]! - min[2]!);
  const volume =
    Math.max(max[0]! - min[0]!, floor) *
    Math.max(max[1]! - min[1]!, floor) *
    Math.max(max[2]! - min[2]!, floor);
  const grid = buildPointGrid(positions, Math.cbrt(volume / n));
  const distances: number[] = [],
    best = new Float64Array(1);
  const step = Math.max(1, Math.floor(n / sample));
  for (let s = 0; s < n; s += step) {
    nearestSquaredDistances(grid, grid.rank[s]!, best);
    if (Number.isFinite(best[0]!)) distances.push(Math.sqrt(best[0]!));
  }
  distances.sort((a, b) => a - b);
  return distances.length ? distances[distances.length >> 1]! : 0;
}

// ---------------------------------------------------------------------------------------------
// Selecting points

/** The points whose `keep` entry is non-zero, in input order. */
export function selectPoints(cloud: CleanCloud, keep: ArrayLike<number>): CleanCloud {
  let count = 0;
  for (let i = 0; i < cloud.count; i++) if (keep[i]) count++;
  if (count === cloud.count) return cloud;
  const positions = new Float32Array(3 * count),
    colors = cloud.colors ? new Uint8Array(3 * count) : undefined,
    normals = cloud.normals ? new Int16Array(2 * count) : undefined;
  let o = 0;
  for (let i = 0; i < cloud.count; i++) {
    if (!keep[i]) continue;
    for (let a = 0; a < 3; a++) {
      positions[3 * o + a] = cloud.positions[3 * i + a]!;
      if (colors) colors[3 * o + a] = cloud.colors![3 * i + a]!;
    }
    if (normals) {
      normals[2 * o] = cloud.normals![2 * i]!;
      normals[2 * o + 1] = cloud.normals![2 * i + 1]!;
    }
    o++;
  }
  return { count, positions, colors, normals };
}

// ---------------------------------------------------------------------------------------------
// 1. Statistical outlier removal

export interface OutlierOptions {
  /** The neighbours of the mean distance. Default 16, at most 32 (the GPU kernel's list). */
  readonly k?: number;
  /** The threshold is mean + stdRatio · σ. Default 2. */
  readonly stdRatio?: number;
}
export const OUTLIER_DEFAULTS = { k: 16, stdRatio: 2 } as const;
export const OUTLIER_MAX_K = 32;

export function checkedOutlierOptions(options: OutlierOptions = {}) {
  const k = options.k ?? OUTLIER_DEFAULTS.k,
    stdRatio = options.stdRatio ?? OUTLIER_DEFAULTS.stdRatio;
  if (!Number.isInteger(k) || k < 1 || k > OUTLIER_MAX_K)
    throw new RangeError(`k must be an integer from 1 to ${OUTLIER_MAX_K}`);
  if (!(stdRatio >= 0) || !Number.isFinite(stdRatio))
    throw new RangeError('stdRatio must be a finite number of at least 0');
  return { k, stdRatio };
}

/**
 * The cell size of the nearest-neighbour grid: about √k nearest-neighbour distances, so a ring of
 * one or two cells holds the k nearest points of a surface sample. Falls back to the cloud's
 * volume when the median spacing is 0 (a cloud of repeated points).
 */
export function neighbourCellSize(positions: Float32Array, k: number): number {
  const spacing = medianPointSpacing(positions);
  if (spacing > 0) return spacing * Math.max(2, Math.sqrt(k));
  const { min, max } = boundingBox(positions);
  const extent = Math.max(max[0]! - min[0]!, max[1]! - min[1]!, max[2]! - min[2]!);
  return extent > 0 ? extent / Math.max(1, Math.cbrt(positions.length / 3)) : 1;
}

/** The mean distance of each point to its `k` nearest neighbours, in input order (the CPU tier). */
export function meanNeighbourDistances(positions: Float32Array, k: number): Float64Array {
  const n = positions.length / 3;
  const out = new Float64Array(n);
  if (n < 2) return out;
  const kk = Math.min(k, n - 1);
  const grid = buildPointGrid(positions, neighbourCellSize(positions, kk));
  const best = new Float64Array(kk);
  for (let s = 0; s < n; s++) {
    nearestSquaredDistances(grid, s, best);
    let sum = 0;
    for (let m = 0; m < kk; m++) sum += Math.sqrt(best[m]!);
    out[grid.order[s]!] = sum / kk;
  }
  return out;
}

export interface OutlierMask {
  readonly keep: Uint8Array;
  readonly mean: number;
  readonly std: number;
  readonly threshold: number;
}

/** Keep the points whose mean distance is at most mean + stdRatio · σ (σ with n − 1). */
export function outlierMask(meanDistances: ArrayLike<number>, stdRatio: number): OutlierMask {
  const n = meanDistances.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += meanDistances[i]!;
  const mean = n ? sum / n : 0;
  let squares = 0;
  for (let i = 0; i < n; i++) squares += (meanDistances[i]! - mean) ** 2;
  const std = n > 1 ? Math.sqrt(squares / (n - 1)) : 0;
  const threshold = mean + stdRatio * std;
  const keep = new Uint8Array(n);
  for (let i = 0; i < n; i++) keep[i] = meanDistances[i]! <= threshold ? 1 : 0;
  return { keep, mean, std, threshold };
}

export interface OutlierResult {
  readonly cloud: CleanCloud;
  readonly removed: number;
  readonly mean: number;
  readonly std: number;
  readonly threshold: number;
}

export function removeStatisticalOutliers(
  cloud: CleanCloud,
  options: OutlierOptions = {},
): OutlierResult {
  const { k, stdRatio } = checkedOutlierOptions(options);
  return applyOutlierMask(cloud, meanNeighbourDistances(cloud.positions, k), stdRatio);
}

/** The mask of `meanDistances` applied to the cloud (shared by the CPU and the GPU tier). */
export function applyOutlierMask(
  cloud: CleanCloud,
  meanDistances: ArrayLike<number>,
  stdRatio: number,
): OutlierResult {
  const { keep, mean, std, threshold } = outlierMask(meanDistances, stdRatio);
  const kept = selectPoints(cloud, keep);
  return { cloud: kept, removed: cloud.count - kept.count, mean, std, threshold };
}

// ---------------------------------------------------------------------------------------------
// 2. Voxel downsampling

export interface VoxelOptions {
  /** The side of a voxel, in the cloud's units. Default: `spacingFactor` · median point spacing. */
  readonly voxelSize?: number;
  /** The multiple of the median point spacing of the default voxel size. Default 2. */
  readonly spacingFactor?: number;
}
export const VOXEL_DEFAULTS = { spacingFactor: 2 } as const;
/** The most voxels along an axis (a voxel coordinate is below 2^21). */
export const VOXEL_MAX_CELLS = 1 << 21;

export function defaultVoxelSize(positions: Float32Array, spacingFactor: number): number {
  return spacingFactor * medianPointSpacing(positions);
}

/** The f32 constants of the voxel coordinates: `floor((p − origin) · inverse)` on each axis. */
export interface VoxelFrame {
  readonly origin: readonly [number, number, number];
  readonly inverse: number;
  readonly voxelSize: number;
}

export function voxelFrame(positions: Float32Array, voxelSize: number): VoxelFrame {
  if (!(voxelSize > 0) || !Number.isFinite(voxelSize))
    throw new RangeError('The voxel size must be a positive number');
  const { min, max } = boundingBox(positions);
  const extent = Math.max(max[0]! - min[0]!, max[1]! - min[1]!, max[2]! - min[2]!);
  if (extent / voxelSize >= VOXEL_MAX_CELLS - 2)
    throw new RangeError('The voxel size is too small for the extent of the cloud');
  return {
    origin: [Math.fround(min[0]!), Math.fround(min[1]!), Math.fround(min[2]!)],
    inverse: Math.fround(1 / voxelSize),
    voxelSize,
  };
}

/**
 * The representative of each point's voxel: the first point of the voxel in input order (the CPU
 * tier). The GPU tier names any member, and `reduceVoxels` accepts both.
 */
export function voxelRepresentatives(positions: Float32Array, frame: VoxelFrame): Uint32Array {
  const n = positions.length / 3;
  const rep = new Uint32Array(n),
    seen = new Map<string, number>();
  const f = Math.fround;
  for (let i = 0; i < n; i++) {
    const c = [0, 1, 2].map((a) =>
      Math.floor(f(f(positions[3 * i + a]! - frame.origin[a]!) * frame.inverse)),
    );
    const key = `${c[0]},${c[1]},${c[2]}`;
    const first = seen.get(key);
    if (first === undefined) {
      seen.set(key, i);
      rep[i] = i;
    } else rep[i] = first;
  }
  return rep;
}

/**
 * One point per voxel from `rep` (`rep[i]` is a member of point i's voxel, one value per voxel).
 * The voxels come in the order of their first member. Position and color are means, the normal is
 * the normalized mean of the normals.
 */
export function reduceVoxels(cloud: CleanCloud, rep: ArrayLike<number>): CleanCloud {
  const n = cloud.count;
  const id = new Int32Array(n).fill(-1);
  let voxels = 0;
  const slot = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const r = rep[i]!;
    if (id[r]! < 0) id[r] = voxels++;
    slot[i] = id[r]!;
  }
  const position = new Float64Array(3 * voxels),
    color = cloud.colors ? new Float64Array(3 * voxels) : undefined,
    normal = cloud.normals ? new Float64Array(3 * voxels) : undefined,
    members = new Uint32Array(voxels),
    first = new Int32Array(voxels).fill(-1);
  const direction: number[] = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const v = slot[i]!;
    members[v]!++;
    if (first[v]! < 0) first[v] = i;
    for (let a = 0; a < 3; a++) {
      position[3 * v + a]! += cloud.positions[3 * i + a]!;
      if (color) color[3 * v + a]! += cloud.colors![3 * i + a]!;
    }
    if (normal) {
      decodeOctahedral(cloud.normals![2 * i]!, cloud.normals![2 * i + 1]!, direction);
      for (let a = 0; a < 3; a++) normal[3 * v + a]! += direction[a]!;
    }
  }
  const positions = new Float32Array(3 * voxels),
    colors = color ? new Uint8Array(3 * voxels) : undefined,
    normals = normal ? new Int16Array(2 * voxels) : undefined;
  for (let v = 0; v < voxels; v++) {
    const m = members[v]!;
    for (let a = 0; a < 3; a++) {
      positions[3 * v + a] = position[3 * v + a]! / m;
      if (colors) colors[3 * v + a] = Math.round(color![3 * v + a]! / m);
    }
    if (normals) {
      const x = normal![3 * v]!,
        y = normal![3 * v + 1]!,
        z = normal![3 * v + 2]!;
      if (Math.hypot(x, y, z) > 1e-6) {
        const [u, w] = encodeOctahedral(x, y, z);
        normals[2 * v] = u;
        normals[2 * v + 1] = w;
      } else {
        // Opposite normals cancel: keep the first member's.
        normals[2 * v] = cloud.normals![2 * first[v]!]!;
        normals[2 * v + 1] = cloud.normals![2 * first[v]! + 1]!;
      }
    }
  }
  return { count: voxels, positions, colors, normals };
}

export interface VoxelResult {
  readonly cloud: CleanCloud;
  readonly voxelSize: number;
}

export function voxelDownsample(cloud: CleanCloud, options: VoxelOptions = {}): VoxelResult {
  const voxelSize =
    options.voxelSize ??
    defaultVoxelSize(cloud.positions, options.spacingFactor ?? VOXEL_DEFAULTS.spacingFactor);
  if (cloud.count === 0) return { cloud, voxelSize };
  const frame = voxelFrame(cloud.positions, voxelSize);
  return {
    cloud: reduceVoxels(cloud, voxelRepresentatives(cloud.positions, frame)),
    voxelSize,
  };
}

// ---------------------------------------------------------------------------------------------
// 3. DBSCAN

export interface DbscanOptions {
  /** The neighbourhood radius (distance ≤ eps), in the cloud's units. */
  readonly eps: number;
  /** The least points in a neighbourhood, the point itself included, for a core point. Default 5. */
  readonly minPoints?: number;
}
export const DBSCAN_DEFAULTS = { minPoints: 5 } as const;

export interface DbscanResult {
  /** The cluster of each point in input order, −1 for noise. */
  readonly labels: Int32Array;
  readonly clusters: number;
  /** The points of each cluster, noise left out. */
  readonly sizes: Uint32Array;
  readonly corePoints: number;
}

export function dbscan(positions: Float32Array, options: DbscanOptions): DbscanResult {
  const eps = options.eps,
    minPoints = options.minPoints ?? DBSCAN_DEFAULTS.minPoints;
  if (!(eps > 0) || !Number.isFinite(eps)) throw new RangeError('eps must be a positive number');
  if (!Number.isInteger(minPoints) || minPoints < 1)
    throw new RangeError('minPoints must be an integer of at least 1');
  const n = positions.length / 3;
  const grid = buildPointGrid(positions, eps),
    p = grid.points,
    eps2 = eps * eps;
  const reach = Math.max(1, Math.ceil(eps / grid.cellSize));
  /** Calls `visit(j)` for each sorted point j within eps of sorted point i, i itself included. */
  const neighbours = (i: number, visit: (j: number) => void) => {
    const cx = grid.cells[3 * i]!,
      cy = grid.cells[3 * i + 1]!,
      cz = grid.cells[3 * i + 2]!;
    const px = p[3 * i]!,
      py = p[3 * i + 1]!,
      pz = p[3 * i + 2]!;
    for (let dz = -reach; dz <= reach; dz++)
      for (let dy = -reach; dy <= reach; dy++)
        for (let dx = -reach; dx <= reach; dx++)
          eachInCell(grid, cx + dx, cy + dy, cz + dz, (j) => {
            const ex = p[3 * j]! - px,
              ey = p[3 * j + 1]! - py,
              ez = p[3 * j + 2]! - pz;
            if (ex * ex + ey * ey + ez * ez <= eps2) visit(j);
          });
  };
  const core = new Uint8Array(n);
  let corePoints = 0;
  for (let i = 0; i < n; i++) {
    let count = 0;
    neighbours(i, () => count++);
    if (count >= minPoints) {
      core[i] = 1;
      corePoints++;
    }
  }
  // scikit-learn's `dbscan_inner`: clusters grow from core points in input order.
  const sortedLabels = new Int32Array(n).fill(-1);
  const stack: number[] = [];
  let clusters = 0;
  for (let o = 0; o < n; o++) {
    const start = grid.rank[o]!;
    if (sortedLabels[start] !== -1 || !core[start]) continue;
    stack.push(start);
    while (stack.length) {
      const i = stack.pop()!;
      if (sortedLabels[i] !== -1) continue;
      sortedLabels[i] = clusters;
      if (core[i])
        neighbours(i, (j) => {
          if (sortedLabels[j] === -1) stack.push(j);
        });
    }
    clusters++;
  }
  const labels = new Int32Array(n),
    sizes = new Uint32Array(clusters);
  for (let s = 0; s < n; s++) {
    labels[grid.order[s]!] = sortedLabels[s]!;
    if (sortedLabels[s]! >= 0) sizes[sortedLabels[s]!]!++;
  }
  return { labels, clusters, sizes, corePoints };
}

export interface ClusterSelection {
  /** The largest clusters to keep. Default 1. */
  readonly keep?: number;
  /** A cluster of at least this fraction of the largest one is kept too. Default 0. */
  readonly minFraction?: number;
}

/** The clusters kept by `keep` and `minFraction`, largest first (a tie takes the lower label). */
export function selectClusters(
  sizes: ArrayLike<number>,
  selection: ClusterSelection = {},
): number[] {
  const keep = selection.keep ?? 1,
    minFraction = selection.minFraction ?? 0;
  const ids = Array.from({ length: sizes.length }, (_, c) => c).sort(
    (a, b) => sizes[b]! - sizes[a]! || a - b,
  );
  if (!ids.length) return [];
  const largest = sizes[ids[0]!]!;
  return ids.filter(
    (c, rank) => rank < keep || (minFraction > 0 && sizes[c]! >= minFraction * largest),
  );
}

export interface ClusterResult extends DbscanResult {
  readonly cloud: CleanCloud;
  readonly removed: number;
  readonly kept: readonly number[];
}

/**
 * DBSCAN, then the points of the kept clusters. A cloud with no cluster at all (every point is
 * noise) comes back unchanged, since no structure was found to prefer.
 */
export function keepLargestClusters(
  cloud: CleanCloud,
  options: DbscanOptions & ClusterSelection,
): ClusterResult {
  const result = dbscan(cloud.positions, options);
  return applyClusters(cloud, result, options);
}

export function applyClusters(
  cloud: CleanCloud,
  result: DbscanResult,
  selection: ClusterSelection = {},
): ClusterResult {
  const kept = selectClusters(result.sizes, selection);
  if (!kept.length) return { ...result, cloud, removed: 0, kept };
  const wanted = new Uint8Array(result.clusters);
  for (const c of kept) wanted[c] = 1;
  const keep = new Uint8Array(cloud.count);
  for (let i = 0; i < cloud.count; i++) {
    const label = result.labels[i]!;
    keep[i] = label >= 0 && wanted[label] ? 1 : 0;
  }
  const next = selectPoints(cloud, keep);
  return { ...result, cloud: next, removed: cloud.count - next.count, kept };
}

// ---------------------------------------------------------------------------------------------
// The pipeline

export interface DenseCleanOptions {
  /** Statistical outlier removal; `false` skips it. On by default. */
  readonly outliers?: false | OutlierOptions;
  /** Voxel downsampling; `false` skips it. On by default. */
  readonly voxel?: false | VoxelOptions;
  /** DBSCAN and the kept clusters; `false` skips it. On by default. */
  readonly cluster?:
    | false
    | (Partial<DbscanOptions> &
        ClusterSelection & {
          /** The default `eps` is this multiple of the median point spacing. Default 4. */
          readonly epsFactor?: number;
        });
}
export const CLUSTER_DEFAULTS = { epsFactor: 4, keep: 1, minFraction: 0 } as const;

export interface DenseCleanReport {
  readonly tier: 'cpu' | 'webgpu';
  readonly before: number;
  readonly afterOutliers: number;
  readonly afterVoxel: number;
  readonly after: number;
  /** The mean + stdRatio · σ threshold of the outlier stage; 0 when it did not run. */
  readonly outlierThreshold: number;
  /** The voxel side used; 0 when the stage did not run. */
  readonly voxelSize: number;
  /** The DBSCAN radius used and the clusters it found; 0 when the stage did not run. */
  readonly eps: number;
  readonly clusters: number;
  readonly keptClusters: number;
  /** Milliseconds of each stage. */
  readonly ms: { readonly outliers: number; readonly voxel: number; readonly cluster: number };
}

export interface DenseCleanResult {
  readonly cloud: CleanCloud;
  readonly report: DenseCleanReport;
}

/** The two primitives a tier supplies; everything else of the pipeline is shared. */
export interface CleanPrimitives {
  readonly tier: 'cpu' | 'webgpu';
  meanDistances(positions: Float32Array, k: number): Promise<ArrayLike<number>>;
  voxelRepresentatives(positions: Float32Array, frame: VoxelFrame): Promise<ArrayLike<number>>;
}

export const cpuPrimitives: CleanPrimitives = {
  tier: 'cpu',
  async meanDistances(positions, k) {
    return meanNeighbourDistances(positions, k);
  },
  async voxelRepresentatives(positions, frame) {
    return voxelRepresentatives(positions, frame);
  },
};

/** Outliers, then voxels, then clusters, each stage on the output of the one before. */
export async function cleanCloudWith(
  primitives: CleanPrimitives,
  input: CleanCloud,
  options: DenseCleanOptions = {},
): Promise<DenseCleanResult> {
  const now = () => performance.now();
  let cloud = input;
  const ms = { outliers: 0, voxel: 0, cluster: 0 };
  let outlierThreshold = 0,
    voxelSize = 0,
    eps = 0,
    clusters = 0,
    keptClusters = 0;
  let t = now();
  if (options.outliers !== false && cloud.count > 1) {
    const { k, stdRatio } = checkedOutlierOptions(options.outliers);
    const distances = await primitives.meanDistances(cloud.positions, Math.min(k, cloud.count - 1));
    const result = applyOutlierMask(cloud, distances, stdRatio);
    cloud = result.cloud;
    outlierThreshold = result.threshold;
  }
  const afterOutliers = cloud.count;
  ms.outliers = now() - t;
  t = now();
  if (options.voxel !== false && cloud.count > 1) {
    const voxel = options.voxel ?? {};
    voxelSize =
      voxel.voxelSize ??
      defaultVoxelSize(cloud.positions, voxel.spacingFactor ?? VOXEL_DEFAULTS.spacingFactor);
    if (voxelSize > 0) {
      const frame = voxelFrame(cloud.positions, voxelSize);
      cloud = reduceVoxels(cloud, await primitives.voxelRepresentatives(cloud.positions, frame));
    }
  }
  const afterVoxel = cloud.count;
  ms.voxel = now() - t;
  t = now();
  if (options.cluster !== false && cloud.count > 1) {
    const cluster = options.cluster ?? {};
    eps =
      cluster.eps ??
      (cluster.epsFactor ?? CLUSTER_DEFAULTS.epsFactor) * medianPointSpacing(cloud.positions);
    if (eps > 0) {
      const result = keepLargestClusters(cloud, {
        eps,
        minPoints: cluster.minPoints,
        keep: cluster.keep ?? CLUSTER_DEFAULTS.keep,
        minFraction: cluster.minFraction ?? CLUSTER_DEFAULTS.minFraction,
      });
      cloud = result.cloud;
      clusters = result.clusters;
      keptClusters = result.kept.length;
    }
  }
  ms.cluster = now() - t;
  return {
    cloud,
    report: {
      tier: primitives.tier,
      before: input.count,
      afterOutliers,
      afterVoxel,
      after: cloud.count,
      outlierThreshold,
      voxelSize,
      eps,
      clusters,
      keptClusters,
      ms,
    },
  };
}

/** The whole pipeline on the CPU tier. */
export function cleanDenseCloud(
  cloud: CleanCloud,
  options: DenseCleanOptions = {},
): Promise<DenseCleanResult> {
  return cleanCloudWith(cpuPrimitives, cloud, options);
}
