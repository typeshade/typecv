import type { Resident } from 'typeshade/runtime';
import { planFlagScan, scanBlockCount } from './scan.ts';
import { ShadeDevice, type ShadePlanReport } from './device.ts';
import {
  flagFloatMatches,
  scatterFloatMatches,
  nearestFloatGuided,
  nearestFloatGuidedTiles,
  nearestFloatScalar,
  nearestFloatTiles,
  nearestFloatTilesWide,
} from './kernels/float-matching.shade.ts';

/**
 * Borrowed row-major descriptors. The caller declares public shape metadata,
 * keeps the buffer alive through the returned promise, and uses the same
 * TypeShade runtime/compiler as this ShadeDevice. Opaque storage is never inspected.
 * Source and target must use distinct Resident handles, even for identical data.
 */
export interface GpuResidentFloatDescriptors {
  readonly descriptors: Resident<Float32Array>;
  readonly count: number;
  readonly dimensions: number;
  /** Exact live array length; must equal count * dimensions. */
  readonly elements: number;
}
export interface GpuFloatMatchOptions {
  /** Strict Euclidean nearest / second-nearest ratio, defaults to 0.8. */
  readonly ratio?: number;
  /** Inclusive Euclidean L2 bound, defaults to 1e19. Descriptors need not be normalized. */
  readonly maxDistance?: number;
  /** Require mutual nearest and ratio tests in both directions; defaults to true. */
  readonly mutual?: boolean;
  /**
   * Guided matching: only candidates within `maxLinePx` of the query's epipolar line count, in
   * both directions, so the ratio test compares the matches the two-view geometry allows.
   * `fundamental` is row-major F with x₂ᵀ F x₁ = 0 for pixel points; `firstPoints` and
   * `secondPoints` hold x, y per descriptor in the same pixel coordinates.
   */
  readonly epipolar?: GpuEpipolarGuide;
}
export interface GpuEpipolarGuide {
  readonly fundamental: ArrayLike<number>;
  readonly firstPoints: ArrayLike<number>;
  readonly secondPoints: ArrayLike<number>;
  /** Point-to-line distance bound in pixels, defaults to 4. */
  readonly maxLinePx?: number;
  /**
   * The WebGPU kernel: the one-query-per-lane `nearestFloatGuided` (`'scalar'`, the default) or
   * `nearestFloatGuidedTiles` (`'tiled'`). Both give the same bits. The CPU tier always runs the
   * scalar kernel, because it has no barriers. The default is scalar because the tiled kernel was
   * 3.7 to 4.9 times slower on SwiftShader (`docs/lab-run-records.md`, "Guided matching kernels").
   */
  readonly kernel?: 'tiled' | 'scalar';
}
export interface GpuFloatMatches {
  /** Stable source-index order; consecutive source and target index pairs. */
  readonly matches: Uint32Array;
  /** Euclidean L2 distances, computed in f32 and returned as f64. */
  readonly distances: Float64Array;
  readonly report: ShadePlanReport;
  readonly execution: {
    readonly applicationUploadBytes: number;
    readonly applicationReadbackBytes: number;
    /** Full public Resident.read snapshots, including unused output capacity. */
    readonly residentSnapshotBytes: number;
  };
}

/**
 * Exhaustive tiled f32 L2 matching without an N x M matrix or descriptor reads.
 * Exact equal-distance ties choose the lower target index; duplicate nearest
 * descriptors fail the strict ratio. Floating-point rounding can change a
 * threshold boundary or a near tie relative to a double-precision CPU oracle.
 */
export function matchFloatDescriptorsGpu(
  device: ShadeDevice,
  first: GpuResidentFloatDescriptors,
  second: GpuResidentFloatDescriptors,
  options: GpuFloatMatchOptions = {},
): Promise<GpuFloatMatches> {
  const ratio = options.ratio ?? 0.8,
    maxDistance = options.maxDistance ?? 1e19,
    mutual = options.mutual ?? true,
    epipolar = options.epipolar,
    maxLinePx = epipolar?.maxLinePx ?? 4;
  for (const input of [first, second]) {
    if (
      !Number.isSafeInteger(input.count) ||
      input.count < 0 ||
      !Number.isSafeInteger(input.dimensions) ||
      input.dimensions < 1 ||
      !Number.isSafeInteger(input.elements) ||
      input.elements !== input.count * input.dimensions ||
      !input.descriptors ||
      typeof input.descriptors.read !== 'function' ||
      typeof input.descriptors.write !== 'function'
    )
      throw new RangeError('Invalid resident float descriptor shape');
  }
  if (
    first.dimensions !== second.dimensions ||
    !Number.isFinite(ratio) ||
    ratio <= 0 ||
    ratio >= 1 ||
    Math.fround(ratio * ratio) <= 0 ||
    Math.fround(ratio * ratio) >= 1 ||
    !Number.isFinite(maxDistance) ||
    maxDistance < 0 ||
    maxDistance > 1e19 ||
    typeof mutual !== 'boolean' ||
    (epipolar &&
      (epipolar.fundamental?.length !== 9 ||
        !Array.from(epipolar.fundamental).every(Number.isFinite) ||
        epipolar.firstPoints?.length !== 2 * first.count ||
        epipolar.secondPoints?.length !== 2 * second.count ||
        !Array.from(epipolar.firstPoints).every(Number.isFinite) ||
        !Array.from(epipolar.secondPoints).every(Number.isFinite) ||
        !(maxLinePx > 0 && maxLinePx <= 1e4)))
  )
    throw new RangeError('Invalid float descriptor match options');
  // This includes both borrowed inputs plus every matcher-owned allocation.
  const largest = Math.max(first.count, second.count),
    workingBytes =
      4 * (first.elements + second.elements) +
      20 * (first.count + second.count) +
      12 * first.count +
      8 +
      (epipolar ? 8 * (first.count + second.count) : 0);
  if (
    !Number.isSafeInteger(workingBytes) ||
    workingBytes > device.workingBytes ||
    Math.max(first.elements, second.elements, 3 * largest) * 4 > device.bindingBytes ||
    Math.ceil(largest / 8) > device.maxKernelElements / 64 ||
    first.elements > 0xffffffff ||
    second.elements > 0xffffffff
  )
    throw new RangeError('Float descriptor matching exceeds working/storage/dispatch budget');
  if (first.descriptors === second.descriptors)
    throw new TypeError('Float matching requires distinct source and target Resident buffers');
  if (!first.count || !second.count)
    return Promise.resolve({
      matches: new Uint32Array(0),
      distances: new Float64Array(0),
      report: { tier: device.backend, dispatches: 0, submissions: 0, groups: [] },
      execution: {
        applicationUploadBytes: 0,
        applicationReadbackBytes: 0,
        residentSnapshotBytes: 0,
      },
    });
  // Snapshot caller metadata before enqueuing. Resident contents remain borrowed.
  const left = { ...first },
    right = { ...second };
  // The guide's matrices for the two directions: F maps first-image points to lines in the
  // second image, Fᵀ second-image points to lines in the first.
  const guide = epipolar && {
    forward: Array.from(epipolar.fundamental, Math.fround),
    backward: [0, 3, 6, 1, 4, 7, 2, 5, 8].map((i) => Math.fround(epipolar.fundamental[i]!)),
    firstPoints: Float32Array.from(epipolar.firstPoints),
    secondPoints: Float32Array.from(epipolar.secondPoints),
  };
  return device.enqueue(async () => {
    const before = device.statistics;
    const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
    let success = false;
    const u32 = (length: number) => {
      const buffer = device.allocateU32(length);
      owned.push(buffer);
      return buffer;
    };
    const f32 = (length: number) => {
      const buffer = device.allocate(length);
      owned.push(buffer);
      return buffer;
    };
    const f32Values = (values: Float32Array) => {
      const buffer = device.allocate(
        Math.max(1, values.length),
        values.length ? values : undefined,
      );
      owned.push(buffer);
      return buffer;
    };
    try {
      const forwardIndices = u32(left.count * 3),
        backwardIndices = u32(right.count * 3),
        forwardDistances = f32(left.count * 2),
        backwardDistances = f32(right.count * 2),
        floatMatches = u32(left.count * 2),
        floatDistances = f32(left.count),
        floatMatchStatus = u32(2),
        flagged = Math.max(1, left.count, right.count),
        matchFlags = u32(flagged),
        matchOffsets = u32(2 * scanBlockCount(flagged)),
        firstPoints = guide ? f32Values(guide.firstPoints) : undefined,
        secondPoints = guide ? f32Values(guide.secondPoints) : undefined;
      const epipolarControl = (m: number[]) => ({
        m0: m[0]!,
        m1: m[1]!,
        m2: m[2]!,
        m3: m[3]!,
        m4: m[4]!,
        m5: m[5]!,
        m6: m[6]!,
        m7: m[7]!,
        m8: m[8]!,
        maxLinePx: Math.fround(maxLinePx),
      });
      const control = {
        queries: left.count,
        candidates: right.count,
        dimensions: left.dimensions,
        ratioSquared: Math.fround(ratio * ratio),
        maxDistanceSquared: Math.fround(maxDistance * maxDistance),
        mutual: mutual ? 1 : 0,
      };
      // Wide slabs keep the exact channel accumulation/tie order while reducing SIFT's
      // per-tile barriers from sixteen to two. Preserve the small-storage kernel otherwise.
      const unguidedTiles =
        left.dimensions >= 128 && device.workgroupStorageBytes >= 9472
          ? nearestFloatTilesWide
          : nearestFloatTiles;
      const report = await device.planSteps([
        [
          'nearestFloatForward',
          () => {
            const bindings = {
              floatQueries: left.descriptors,
              floatCandidates: right.descriptors,
              nearestIndices: forwardIndices,
              nearestDistances: forwardDistances,
              floatMatchControl: control,
            };
            if (guide) {
              const guided = {
                ...bindings,
                queryPoints: firstPoints!,
                candidatePoints: secondPoints!,
                epipolarGuide: epipolarControl(guide.forward),
              };
              if (device.backend === 'cpu' || epipolar!.kernel !== 'tiled')
                nearestFloatGuided(guided, Math.ceil(left.count / 64));
              else nearestFloatGuidedTiles(guided, Math.ceil(left.count / 8));
            } else if (device.backend === 'cpu')
              nearestFloatScalar(bindings, Math.ceil(left.count / 64));
            else unguidedTiles(bindings, Math.ceil(left.count / 8));
          },
        ],
        [
          'nearestFloatBackward',
          () => {
            const bindings = {
              floatQueries: right.descriptors,
              floatCandidates: left.descriptors,
              nearestIndices: backwardIndices,
              nearestDistances: backwardDistances,
              floatMatchControl: { ...control, queries: right.count, candidates: left.count },
            };
            if (guide) {
              const guided = {
                ...bindings,
                queryPoints: secondPoints!,
                candidatePoints: firstPoints!,
                epipolarGuide: epipolarControl(guide.backward),
              };
              if (device.backend === 'cpu' || epipolar!.kernel !== 'tiled')
                nearestFloatGuided(guided, Math.ceil(right.count / 64));
              else nearestFloatGuidedTiles(guided, Math.ceil(right.count / 8));
            } else if (device.backend === 'cpu')
              nearestFloatScalar(bindings, Math.ceil(right.count / 64));
            else unguidedTiles(bindings, Math.ceil(right.count / 8));
          },
        ],
        [
          'flagFloatMatches',
          () =>
            flagFloatMatches(
              {
                forwardIndices,
                backwardIndices,
                forwardDistances,
                backwardDistances,
                matchFlags,
                floatMatchControl: control,
              },
              Math.ceil(flagged / 64),
            ),
        ],
        [
          'scanFloatMatches',
          () =>
            planFlagScan(
              device,
              { flags: matchFlags, blockSums: matchOffsets, totals: floatMatchStatus },
              flagged,
            ),
        ],
        [
          'scatterFloatMatches',
          () =>
            scatterFloatMatches(
              {
                forwardIndices,
                backwardIndices,
                forwardDistances,
                backwardDistances,
                floatMatches,
                floatDistances,
                matchFlags,
                matchOffsets,
                floatMatchControl: control,
              },
              Math.ceil(Math.max(1, left.count) / 64),
            ),
        ],
      ]);
      const status = await device.read(floatMatchStatus, 2),
        count = status[0]!;
      if (status[1]) throw new RangeError('Nonfinite descriptors or f32 L2 overflow');
      if (count > left.count) throw new Error('Invalid GPU float match count');
      const [pairs, distances] = count
        ? await Promise.all([
            device.read(floatMatches, count * 2),
            device.read(floatDistances, count),
          ])
        : [new Uint32Array(0), new Float32Array(0)];
      const after = device.statistics;
      success = true;
      return {
        matches: pairs.slice(0, count * 2),
        distances: Float64Array.from(distances.subarray(0, count)),
        report,
        execution: {
          applicationUploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
          applicationReadbackBytes:
            after.applicationReadbackBytes - before.applicationReadbackBytes,
          residentSnapshotBytes: after.residentSnapshotBytes - before.residentSnapshotBytes,
        },
      };
    } finally {
      for (const buffer of owned) success ? device.recycle(buffer) : device.drop(buffer);
    }
  });
}
