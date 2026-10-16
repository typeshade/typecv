import type { Resident } from 'typeshade/runtime';
import { ShadeDevice, type ShadePlanReport } from './device.ts';
import {
  compactFeatures,
  compactMutualBrief,
  matchBrief,
  orientBrief,
  rgbaToGray,
  scoreFast9,
  selectTileCorners,
} from './features.shade.ts';

export interface GpuFeatureOptions {
  readonly threshold?: number;
  readonly maxFeatures?: number;
}
export interface GpuFeatureExecution {
  readonly applicationUploadBytes: number;
  readonly applicationReadbackBytes: number;
  /** Full Resident.read snapshots, including unused output capacity; not a bus traffic counter. */
  readonly residentSnapshotBytes?: number;
}
export interface GpuFeatures {
  readonly keypoints: Float64Array;
  readonly scores: Float32Array;
  /** Eight u32 words per keypoint; BRIEF-256 bits. */
  readonly descriptors: Uint32Array;
  readonly capacity: number;
  readonly report: ShadePlanReport;
  readonly execution?: GpuFeatureExecution;
}
export interface GpuResidentFeatures {
  /** Exact feature count, read from a four-byte GPU counter after stable compaction. */
  readonly count: number;
  /** Resident output capacity; the live features occupy [0, count). */
  readonly capacity: number;
  readonly report: ShadePlanReport;
  readonly execution: GpuFeatureExecution;
  /** Read and cache compact [x0,y0,x1,y1,…] keypoints, in matching-index order. */
  readKeypoints(): Promise<Float64Array>;
  /** Read keypoints, scores and descriptors for callers that need the legacy host form. */
  read(): Promise<GpuFeatures>;
  /** Release the resident point and descriptor buffers after earlier queued jobs finish. */
  dispose(): Promise<void>;
}
export interface GpuFeatureMatches {
  /** Two feature indices per row, suitable for PairMatches.matches. */
  readonly matches: Uint32Array;
  readonly distances: Float64Array;
  readonly report: ShadePlanReport;
  readonly execution?: GpuFeatureExecution;
}

type ResidentInput = {
  readonly count: number;
  readonly points: Resident<Float32Array>;
  readonly descriptors: Resident<Uint32Array>;
};
type ResidentState = ResidentInput & {
  readonly device: ShadeDevice;
  readonly capacity: number;
  readonly report: ShadePlanReport;
  readonly execution: GpuFeatureExecution;
  disposed: boolean;
  pointsRead?: Promise<Float32Array>;
  keypointsRead?: Promise<Float64Array>;
  hostRead?: Promise<GpuFeatures>;
};
const residentStates = new WeakMap<GpuResidentFeatures, ResidentState>();

const emptyReport = (tier: ShadePlanReport['tier']): ShadePlanReport => ({
  tier,
  dispatches: 0,
  submissions: 0,
  groups: [],
});
type DeviceStatistics = ShadeDevice['statistics'];
const executionDelta = (
  before: DeviceStatistics,
  after: DeviceStatistics,
): GpuFeatureExecution => ({
  applicationUploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
  applicationReadbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
  residentSnapshotBytes: after.residentSnapshotBytes - before.residentSnapshotBytes,
});

const validRgba = (input: Uint8Array | Uint8ClampedArray, width: number, height: number) =>
  Number.isSafeInteger(width) &&
  Number.isSafeInteger(height) &&
  width >= 48 &&
  height >= 48 &&
  width <= 4096 &&
  height <= 4096 &&
  input.length === width * height * 4;

function residentState(device: ShadeDevice, features: GpuResidentFeatures): ResidentState {
  const state = residentStates.get(features);
  if (!state || state.device !== device || state.disposed)
    throw new TypeError('GPU resident features are disposed or belong to another device');
  return state;
}

/** FAST-9 + oriented BRIEF; keep compact point and descriptor arrays resident for repeated pairs. */
export async function extractFeaturesGpuResident(
  device: ShadeDevice,
  rgbaInput: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  options: GpuFeatureOptions = {},
): Promise<GpuResidentFeatures> {
  const threshold = options.threshold ?? 0.045;
  if (
    !validRgba(rgbaInput, width, height) ||
    !Number.isFinite(threshold) ||
    threshold < 0.005 ||
    threshold > 0.4
  )
    throw new RangeError('Invalid RGBA frame dimensions or FAST threshold');
  const pixelCount = width * height;
  if (pixelCount > device.maxKernelElements || pixelCount * 20 > device.workingBytes)
    throw new RangeError('Frame exceeds TypeShade feature working or dispatch budget');
  const tilesX = Math.ceil(width / 24),
    tilesY = Math.ceil(height / 24);
  const tileCount = tilesX * tilesY;
  const maxFeatures = options.maxFeatures ?? Math.min(tileCount * 4, 2000);
  if (!Number.isSafeInteger(maxFeatures) || maxFeatures < 1 || maxFeatures > 2000)
    throw new RangeError('maxFeatures must be an integer from 1 through 2000');
  const slotsPerTile = Math.max(1, Math.min(4, Math.ceil(maxFeatures / tileCount)));
  const capacity = Math.min(tileCount * slotsPerTile, maxFeatures);
  const selectedTiles = Math.ceil(capacity / slotsPerTile);
  if (
    !Number.isSafeInteger(capacity) ||
    capacity < 1 ||
    capacity > device.maxKernelElements ||
    capacity * 96 + pixelCount * 12 > device.workingBytes
  )
    throw new RangeError('Feature count exceeds TypeShade working budget');

  // Canvas image data is byte-aligned RGBA. On little-endian hosts its u32 view is exactly the
  // packed word consumed by rgbaToGray. Snapshot the view once for caller ownership safety,
  // avoiding the per-pixel JS repack loop while retaining an intentional full-frame copy.
  let packed: Uint32Array;
  const littleEndian = new Uint8Array(Uint32Array.of(0x01020304).buffer)[0] === 4;
  if (littleEndian && rgbaInput.byteOffset % 4 === 0 && rgbaInput.byteLength % 4 === 0) {
    packed = new Uint32Array(rgbaInput.buffer, rgbaInput.byteOffset, pixelCount).slice();
  } else {
    packed = new Uint32Array(pixelCount);
    for (let i = 0; i < pixelCount; i++) {
      const at = i * 4;
      packed[i] =
        rgbaInput[at]! |
        (rgbaInput[at + 1]! << 8) |
        (rgbaInput[at + 2]! << 16) |
        (rgbaInput[at + 3]! << 24);
    }
  }

  return device.enqueue(async () => {
    const before = device.statistics;
    const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
    const retained = new Set<Resident<Float32Array> | Resident<Uint32Array>>();
    let success = false;
    const u32 = (length: number, input?: Uint32Array) => {
      const value = device.allocateU32(length, input);
      owned.push(value);
      return value;
    };
    const f32 = (length: number) => {
      const value = device.allocate(length);
      owned.push(value);
      return value;
    };
    try {
      const input = u32(pixelCount, packed),
        gray = f32(pixelCount),
        scores = f32(pixelCount),
        points = f32(capacity * 4),
        descriptors = u32(capacity * 8),
        compactPoints = f32(capacity * 4),
        compactDescriptors = u32(capacity * 8),
        countBuffer = u32(1, new Uint32Array(1));
      const featureControl = {
        width,
        height,
        tilesX,
        tilesY,
        slotsPerTile,
        selectedTiles,
        threshold,
        capacity,
      };
      const report = await device.plan(() => {
        rgbaToGray({ rgba: input, gray }, Math.ceil(pixelCount / 64));
        scoreFast9({ gray, scores, featureControl }, Math.ceil(pixelCount / 64));
        selectTileCorners(
          { scores, features: points, featureControl },
          Math.ceil(selectedTiles / 64),
        );
        orientBrief(
          { gray, features: points, descriptors, featureControl },
          Math.ceil(capacity / 64),
        );
        compactFeatures(
          {
            features: points,
            descriptors,
            packedFeatures: compactPoints,
            packedDescriptors: compactDescriptors,
            featureCount: countBuffer,
            featureControl,
          },
          1,
        );
      });
      const countData = await device.read(countBuffer, 1),
        count = countData[0]!;
      if (count > capacity) throw new Error('GPU feature compaction returned an invalid count');
      retained.add(compactPoints);
      retained.add(compactDescriptors);
      const execution = executionDelta(before, device.statistics);
      const state: ResidentState = {
        device,
        count,
        capacity,
        points: compactPoints,
        descriptors: compactDescriptors,
        report,
        execution,
        disposed: false,
      };
      const result: GpuResidentFeatures = {
        count,
        capacity,
        report,
        execution,
        readKeypoints: () => {
          const live = residentState(device, result);
          live.keypointsRead ??= readPoints(live).then((values) => {
            const keypoints = new Float64Array(live.count * 2);
            for (let i = 0; i < live.count; i++) {
              keypoints[i * 2] = values[i * 4]!;
              keypoints[i * 2 + 1] = values[i * 4 + 1]!;
            }
            return keypoints;
          });
          return live.keypointsRead;
        },
        read: () => {
          const live = residentState(device, result);
          live.hostRead ??= (async () => {
            const [values, descriptors] = await Promise.all([
              readPoints(live),
              device.enqueue(() => device.read(live.descriptors, live.count * 8)),
            ]);
            const keypoints = new Float64Array(live.count * 2),
              scores = new Float32Array(live.count);
            for (let i = 0; i < live.count; i++) {
              keypoints[i * 2] = values[i * 4]!;
              keypoints[i * 2 + 1] = values[i * 4 + 1]!;
              scores[i] = values[i * 4 + 2]!;
            }
            return {
              keypoints,
              scores,
              descriptors: descriptors.slice(0, live.count * 8),
              capacity: live.capacity,
              report: live.report,
              execution: executionDelta(before, device.statistics),
            };
          })();
          return live.hostRead;
        },
        dispose: async () => {
          const live = residentStates.get(result);
          if (!live || live.disposed) return;
          live.disposed = true;
          await device.enqueue(async () => {
            device.drop(live.points);
            device.drop(live.descriptors);
          });
        },
      };
      residentStates.set(result, state);
      success = true;
      return result;
    } finally {
      for (const buffer of owned) {
        if (retained.has(buffer)) continue;
        success ? device.recycle(buffer) : device.drop(buffer);
      }
    }
  });
}

function readPoints(state: ResidentState): Promise<Float32Array> {
  state.pointsRead ??= state.device.enqueue(() => state.device.read(state.points, state.count * 4));
  return state.pointsRead;
}

/** Compatibility API: return compact host arrays and release the temporary resident feature set. */
export async function extractFeaturesGpu(
  device: ShadeDevice,
  rgbaInput: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  options: GpuFeatureOptions = {},
): Promise<GpuFeatures> {
  const features = await extractFeaturesGpuResident(device, rgbaInput, width, height, options);
  try {
    return await features.read();
  } finally {
    await features.dispose();
  }
}

/** Bidirectional ratio-tested Hamming matches over persistent GPU feature arrays. */
export async function matchResidentFeaturesGpu(
  device: ShadeDevice,
  first: GpuResidentFeatures,
  second: GpuResidentFeatures,
  options: { readonly ratio?: number; readonly maxDistance?: number } = {},
): Promise<GpuFeatureMatches> {
  const left = residentState(device, first),
    right = residentState(device, second),
    ratio = options.ratio ?? 0.82,
    maxDistance = options.maxDistance ?? 80;
  validateMatchOptions(left.count, right.count, ratio, maxDistance, device);
  if (!left.count || !right.count)
    return {
      matches: new Uint32Array(0),
      distances: new Float64Array(0),
      report: emptyReport(device.backend),
      execution: { applicationUploadBytes: 0, applicationReadbackBytes: 0 },
    };
  return device.enqueue(() => runResidentMatch(device, left, right, ratio, maxDistance));
}

/** Bidirectional ratio-tested Hamming matches for the existing host-array API. */
export async function matchFeaturesGpu(
  device: ShadeDevice,
  first: GpuFeatures,
  second: GpuFeatures,
  options: { readonly ratio?: number; readonly maxDistance?: number } = {},
): Promise<GpuFeatureMatches> {
  const ratio = options.ratio ?? 0.82,
    maxDistance = options.maxDistance ?? 80;
  if (
    first.descriptors.length !== first.scores.length * 8 ||
    second.descriptors.length !== second.scores.length * 8 ||
    first.keypoints.length !== first.scores.length * 2 ||
    second.keypoints.length !== second.scores.length * 2
  )
    throw new RangeError('Invalid BRIEF match inputs or options');
  validateMatchOptions(first.scores.length, second.scores.length, ratio, maxDistance, device);
  if (!first.scores.length || !second.scores.length)
    return {
      matches: new Uint32Array(0),
      distances: new Float64Array(0),
      report: emptyReport(device.backend),
      execution: { applicationUploadBytes: 0, applicationReadbackBytes: 0 },
    };
  const pointArray = (features: GpuFeatures) => {
    const values = new Float32Array(features.scores.length * 4);
    for (let i = 0; i < features.scores.length; i++) {
      values[i * 4] = features.keypoints[i * 2]!;
      values[i * 4 + 1] = features.keypoints[i * 2 + 1]!;
      values[i * 4 + 2] = features.scores[i]!;
      values[i * 4 + 3] = 1;
    }
    return values;
  };
  const a = pointArray(first),
    b = pointArray(second);
  return device.enqueue(async () => {
    const before = device.statistics;
    const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
    let success = false;
    const f32 = (length: number, values: Float32Array) => {
      const value = device.allocate(length, values);
      owned.push(value);
      return value;
    };
    const u32 = (length: number, values: Uint32Array) => {
      const value = device.allocateU32(length, values);
      owned.push(value);
      return value;
    };
    try {
      const left: ResidentInput = {
        count: first.scores.length,
        points: f32(a.length, a),
        descriptors: u32(first.descriptors.length, first.descriptors),
      };
      const right: ResidentInput = {
        count: second.scores.length,
        points: f32(b.length, b),
        descriptors: u32(second.descriptors.length, second.descriptors),
      };
      const result = await runResidentMatch(device, left, right, ratio, maxDistance);
      success = true;
      return { ...result, execution: executionDelta(before, device.statistics) };
    } finally {
      for (const buffer of owned) success ? device.recycle(buffer) : device.drop(buffer);
    }
  });
}

function validateMatchOptions(
  firstCount: number,
  secondCount: number,
  ratio: number,
  maxDistance: number,
  device: ShadeDevice,
): void {
  if (
    !Number.isFinite(ratio) ||
    ratio <= 0.1 ||
    ratio >= 1 ||
    !Number.isSafeInteger(maxDistance) ||
    maxDistance < 1 ||
    maxDistance > 256
  )
    throw new RangeError('Invalid BRIEF match inputs or options');
  if (
    (firstCount + secondCount) * 60 > device.workingBytes ||
    Math.max(firstCount, secondCount) > device.maxKernelElements ||
    firstCount * secondCount > 4_000_000
  )
    throw new RangeError('Feature match exceeds TypeShade working or dispatch budget');
}

async function runResidentMatch(
  device: ShadeDevice,
  first: ResidentInput,
  second: ResidentInput,
  ratio: number,
  maxDistance: number,
): Promise<GpuFeatureMatches> {
  const before = device.statistics;
  const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
  let success = false;
  const u32 = (length: number) => {
    const value = device.allocateU32(length);
    owned.push(value);
    return value;
  };
  try {
    const forward = u32(first.count * 3),
      reverse = u32(second.count * 3),
      mutual = u32(first.count * 2),
      distances = u32(first.count),
      countBuffer = u32(1);
    const firstControl = {
      firstCount: first.count,
      secondCount: second.count,
      ratio,
      maxDistance,
    };
    const secondControl = {
      firstCount: second.count,
      secondCount: first.count,
      ratio,
      maxDistance,
    };
    const report = await device.plan(() => {
      matchBrief(
        {
          rightFeatures: second.points,
          leftDescriptors: first.descriptors,
          rightDescriptors: second.descriptors,
          matches: forward,
          matchControl: firstControl,
        },
        Math.ceil(first.count / 64),
      );
      matchBrief(
        {
          rightFeatures: first.points,
          leftDescriptors: second.descriptors,
          rightDescriptors: first.descriptors,
          matches: reverse,
          matchControl: secondControl,
        },
        Math.ceil(second.count / 64),
      );
      compactMutualBrief(
        {
          forwardMatches: forward,
          reverseMatches: reverse,
          mutualMatches: mutual,
          mutualDistances: distances,
          featureCount: countBuffer,
          matchControl: firstControl,
        },
        1,
      );
    });
    const count = (await device.read(countBuffer, 1))[0]!;
    if (count > first.count)
      throw new Error('GPU mutual-match compaction returned an invalid count');
    const [matchWords, distanceWords] = await Promise.all([
      device.read(mutual, count * 2),
      device.read(distances, count),
    ]);
    success = true;
    return {
      matches: matchWords.slice(0, count * 2),
      distances: Float64Array.from(distanceWords.subarray(0, count)),
      report,
      execution: executionDelta(before, device.statistics),
    };
  } finally {
    for (const buffer of owned) success ? device.recycle(buffer) : device.drop(buffer);
  }
}
