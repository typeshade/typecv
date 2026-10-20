/**
 * Cleanup of a fused dense point cloud on the GPU (`docs/dense-mvs.md`, section 3.12). The two
 * steps that pay on a GPU run as TypeShade kernels: the mean distance to the k nearest neighbours
 * (`cloud-knn.shade.ts`, for the statistical outlier removal) and the voxel binning
 * (`cloud-voxel.shade.ts`). The host builds the uniform-grid spatial hash, applies the threshold,
 * sums each voxel in double and runs DBSCAN (`src/reconstruction/cloud-clean.ts`, the CPU
 * reference of every filter, which these functions share with the CPU tier).
 */
import type { ShadeDevice } from './device.ts';
import { knnMeanDistance } from './cloud-knn.shade.ts';
import { voxelClaim } from './cloud-voxel.shade.ts';
import {
  KNN_MAX_RING,
  applyOutlierMask,
  buildPointGrid,
  checkedOutlierOptions,
  cleanCloudWith,
  cpuPrimitives,
  neighbourCellSize,
  reduceVoxels,
  defaultVoxelSize,
  voxelFrame,
  VOXEL_DEFAULTS,
  type CleanCloud,
  type CleanPrimitives,
  type DenseCleanOptions,
  type DenseCleanResult,
  type OutlierOptions,
  type OutlierResult,
  type VoxelFrame,
  type VoxelOptions,
  type VoxelResult,
} from '../reconstruction/cloud-clean.ts';

const LANES = 64;

/** The chunks of one pass over `total` points: sized by the device from the rate of `key`. */
export function* chunkedPass(
  device: ShadeDevice,
  key: string,
  name: string,
  total: number,
  dispatch: (range: { first: number; end: number }, groups: number) => void,
) {
  const warm = { id: key, run: () => dispatch({ first: 0, end: 0 }, 0) };
  for (let first = 0; first < total;) {
    const remaining = total - first;
    let n = Math.min(remaining, device.chunkSize(key, remaining, Math.min(remaining, LANES)));
    if (n < remaining) n = Math.max(LANES, n - (n % LANES));
    n = Math.min(n, remaining);
    const range = { first, end: first + n };
    yield [
      name,
      () => dispatch(range, Math.ceil(n / LANES)),
      undefined,
      { key, units: n, warm },
    ] as const;
    first += n;
  }
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

/** The mean distance to the k nearest neighbours of each point, in input order, on `device`. */
export async function meanNeighbourDistancesGpu(
  device: ShadeDevice,
  positions: Float32Array,
  k: number,
): Promise<Float32Array> {
  const n = positions.length / 3;
  const out = new Float32Array(n);
  if (n < 2) return out;
  const kk = Math.min(k, n - 1);
  const grid = buildPointGrid(positions, neighbourCellSize(positions, kk));
  const arrays = [
    device.allocate(3 * n, grid.points),
    device.allocateU32(3 * n, grid.cells),
    device.allocateU32(grid.bucketStart.length, grid.bucketStart),
    device.allocateU32(n, cellVisitOrder(grid.cells, n)),
    device.allocate(n),
  ] as const;
  const [gridPoints, gridCells, bucketStart, visitOrder, knnMean] = arrays;
  let successful = false;
  try {
    const cloudControl = {
      count: n,
      k: kk,
      mask: grid.mask,
      maxRing: KNN_MAX_RING,
      cellSize: grid.cellSize,
      originX: grid.origin[0],
      originY: grid.origin[1],
      originZ: grid.origin[2],
    };
    await device.planSteps(
      chunkedPass(device, 'cloudKnn', 'knnMeanDistance', n, (cloudRange, groups) =>
        knnMeanDistance(
          { cloudControl, cloudRange, gridPoints, gridCells, bucketStart, visitOrder, knnMean },
          groups,
        ),
      ),
    );
    const sorted = await device.read(knnMean, n);
    for (let s = 0; s < n; s++) out[grid.order[s]!] = sorted[s]!;
    successful = true;
  } finally {
    for (const a of arrays) successful ? device.recycle(a) : device.drop(a);
  }
  return out;
}

/** The owner of each point's voxel on `device`; any member of the voxel may own it. */
export async function voxelRepresentativesGpu(
  device: ShadeDevice,
  positions: Float32Array,
  frame: VoxelFrame,
): Promise<Uint32Array> {
  const n = positions.length / 3;
  let table = 16;
  while (table < 2 * n) table *= 2;
  const arrays = [
    device.allocate(3 * n, positions),
    device.allocateU32(table, new Uint32Array(table).fill(0xffffffff)),
    device.allocateU32(n),
  ] as const;
  const [voxelPoints, voxelTable, voxelRep] = arrays;
  let successful = false;
  try {
    const voxelControl = {
      count: n,
      mask: table - 1,
      originX: frame.origin[0],
      originY: frame.origin[1],
      originZ: frame.origin[2],
      inverse: frame.inverse,
    };
    await device.planSteps(
      chunkedPass(device, 'cloudVoxel', 'voxelClaim', n, (voxelRange, groups) =>
        voxelClaim({ voxelControl, voxelRange, voxelPoints, voxelTable, voxelRep }, groups),
      ),
    );
    const rep = await device.read(voxelRep, n);
    successful = true;
    return Uint32Array.from(rep);
  } finally {
    for (const a of arrays) successful ? device.recycle(a) : device.drop(a);
  }
}

/**
 * The two primitives on `device`. A cloud whose arrays exceed the device's storage binding limit
 * runs on the CPU tier instead, so the cleanup never fails for size alone.
 */
export function gpuCleanPrimitives(device: ShadeDevice): CleanPrimitives {
  const fits = (positions: Float32Array, wordsPerPoint: number) =>
    (positions.length / 3) * wordsPerPoint * 4 <= device.bindingBytes;
  return {
    tier: device.backend,
    async meanDistances(positions, k) {
      return fits(positions, 3)
        ? meanNeighbourDistancesGpu(device, positions, k)
        : cpuPrimitives.meanDistances(positions, k);
    },
    async voxelRepresentatives(positions, frame) {
      return fits(positions, 6)
        ? voxelRepresentativesGpu(device, positions, frame)
        : cpuPrimitives.voxelRepresentatives(positions, frame);
    },
  };
}

/** Statistical outlier removal with the kNN distances on `device`. */
export async function removeStatisticalOutliersGpu(
  device: ShadeDevice,
  cloud: CleanCloud,
  options: OutlierOptions = {},
): Promise<OutlierResult> {
  const { k, stdRatio } = checkedOutlierOptions(options);
  const distances = await gpuCleanPrimitives(device).meanDistances(
    cloud.positions,
    Math.min(k, Math.max(cloud.count - 1, 1)),
  );
  return applyOutlierMask(cloud, distances, stdRatio);
}

/** Voxel downsampling with the binning on `device`. */
export async function voxelDownsampleGpu(
  device: ShadeDevice,
  cloud: CleanCloud,
  options: VoxelOptions = {},
): Promise<VoxelResult> {
  const voxelSize =
    options.voxelSize ??
    defaultVoxelSize(cloud.positions, options.spacingFactor ?? VOXEL_DEFAULTS.spacingFactor);
  if (cloud.count === 0) return { cloud, voxelSize };
  const frame = voxelFrame(cloud.positions, voxelSize);
  const rep = await gpuCleanPrimitives(device).voxelRepresentatives(cloud.positions, frame);
  return { cloud: reduceVoxels(cloud, rep), voxelSize };
}

/**
 * The cleanup of a fused dense cloud: outliers, then voxels, then DBSCAN clusters, each stage on
 * the output of the one before (section 3.12). The report holds the point count after each stage
 * and the time of each.
 */
export function cleanDenseCloudGpu(
  device: ShadeDevice,
  cloud: CleanCloud,
  options: DenseCleanOptions = {},
): Promise<DenseCleanResult> {
  return cleanCloudWith(gpuCleanPrimitives(device), cloud, options);
}
