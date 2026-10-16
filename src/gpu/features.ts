import type { Resident } from 'typeshade/runtime';
import { ShadeDevice, type ShadePlanReport } from './device.ts';
import {
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
export interface GpuFeatures {
  readonly keypoints: Float64Array;
  readonly scores: Float32Array;
  /** Eight u32 words per keypoint; BRIEF-256 bits. */
  readonly descriptors: Uint32Array;
  readonly capacity: number;
  readonly report: ShadePlanReport;
}
export interface GpuFeatureMatches {
  /** Two feature indices per row, suitable for PairMatches.matches. */
  readonly matches: Uint32Array;
  readonly distances: Float64Array;
  readonly report: ShadePlanReport;
}

const validRgba = (input: Uint8Array | Uint8ClampedArray, width: number, height: number) =>
  Number.isSafeInteger(width) &&
  Number.isSafeInteger(height) &&
  width >= 48 &&
  height >= 48 &&
  width <= 4096 &&
  height <= 4096 &&
  input.length === width * height * 4;

/** FAST-9 spatial selection, intensity-centroid BRIEF-256 and orientation run in TypeShade. */
export async function extractFeaturesGpu(
  device: ShadeDevice,
  rgbaInput: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  options: GpuFeatureOptions = {},
): Promise<GpuFeatures> {
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
  const capacity = tileCount * slotsPerTile;
  if (
    !Number.isSafeInteger(capacity) ||
    capacity < 1 ||
    capacity > device.maxKernelElements ||
    capacity * 48 + pixelCount * 12 > device.workingBytes
  )
    throw new RangeError('Feature count exceeds TypeShade working budget');
  const packed = new Uint32Array(pixelCount);
  for (let i = 0; i < pixelCount; i++) {
    const at = i * 4;
    packed[i] =
      rgbaInput[at]! |
      (rgbaInput[at + 1]! << 8) |
      (rgbaInput[at + 2]! << 16) |
      (rgbaInput[at + 3]! << 24);
  }
  return device.enqueue(async () => {
    const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
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
        descriptors = u32(capacity * 8);
      const featureControl = { width, height, tilesX, tilesY, slotsPerTile, threshold, capacity };
      const report = await device.plan(() => {
        rgbaToGray({ rgba: input, gray }, Math.ceil(pixelCount / 64));
        scoreFast9({ gray, scores, featureControl }, Math.ceil(pixelCount / 64));
        selectTileCorners(
          { scores, features: points, featureControl },
          Math.ceil((tilesX * tilesY) / 64),
        );
        orientBrief(
          { gray, features: points, descriptors, featureControl },
          Math.ceil(capacity / 64),
        );
      });
      const pointData = await device.read(points, capacity * 4);
      const descriptorData = await device.read(descriptors, capacity * 8);
      const ids: number[] = [];
      for (let i = 0; i < capacity; i++) if (pointData[i * 4 + 3] !== 0) ids.push(i);
      const keypoints = new Float64Array(ids.length * 2),
        featureScores = new Float32Array(ids.length),
        compactDescriptors = new Uint32Array(ids.length * 8);
      ids.forEach((id, i) => {
        keypoints[i * 2] = pointData[id * 4]!;
        keypoints[i * 2 + 1] = pointData[id * 4 + 1]!;
        featureScores[i] = pointData[id * 4 + 2]!;
        compactDescriptors.set(descriptorData.subarray(id * 8, id * 8 + 8), i * 8);
      });
      success = true;
      return {
        keypoints,
        scores: featureScores,
        descriptors: compactDescriptors,
        capacity,
        report,
      };
    } finally {
      for (const buffer of owned) success ? device.recycle(buffer) : device.drop(buffer);
    }
  });
}

/** Bidirectional ratio-tested Hamming matches; distance ranking and filtering run on TypeShade. */
export async function matchFeaturesGpu(
  device: ShadeDevice,
  first: GpuFeatures,
  second: GpuFeatures,
  options: { readonly ratio?: number; readonly maxDistance?: number } = {},
): Promise<GpuFeatureMatches> {
  const ratio = options.ratio ?? 0.82,
    maxDistance = options.maxDistance ?? 80;
  if (
    !Number.isFinite(ratio) ||
    ratio <= 0.1 ||
    ratio >= 1 ||
    !Number.isSafeInteger(maxDistance) ||
    maxDistance < 1 ||
    maxDistance > 256 ||
    first.descriptors.length !== first.scores.length * 8 ||
    second.descriptors.length !== second.scores.length * 8
  )
    throw new RangeError('Invalid BRIEF match inputs or options');
  if (!first.scores.length || !second.scores.length)
    return {
      matches: new Uint32Array(0),
      distances: new Float64Array(0),
      report: { tier: device.backend, dispatches: 0, submissions: 0, groups: [] },
    };
  if (
    (first.scores.length + second.scores.length) * 60 > device.workingBytes ||
    Math.max(first.scores.length, second.scores.length) > device.maxKernelElements ||
    first.scores.length * second.scores.length > 4_000_000
  )
    throw new RangeError('Feature match exceeds TypeShade working or dispatch budget');
  const aPoints = new Float32Array(first.scores.length * 4),
    bPoints = new Float32Array(second.scores.length * 4);
  for (let i = 0; i < first.scores.length; i++) {
    aPoints[i * 4] = first.keypoints[i * 2]!;
    aPoints[i * 4 + 1] = first.keypoints[i * 2 + 1]!;
    aPoints[i * 4 + 3] = 1;
  }
  for (let i = 0; i < second.scores.length; i++) {
    bPoints[i * 4] = second.keypoints[i * 2]!;
    bPoints[i * 4 + 1] = second.keypoints[i * 2 + 1]!;
    bPoints[i * 4 + 3] = 1;
  }
  return device.enqueue(async () => {
    const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
    let success = false;
    const f32 = (length: number, values?: Float32Array) => {
      const value = device.allocate(length, values);
      owned.push(value);
      return value;
    };
    const u32 = (length: number, values?: Uint32Array) => {
      const value = device.allocateU32(length, values);
      owned.push(value);
      return value;
    };
    try {
      const leftFeatures = f32(aPoints.length, aPoints),
        rightFeatures = f32(bPoints.length, bPoints),
        leftDescriptors = u32(first.descriptors.length, first.descriptors),
        rightDescriptors = u32(second.descriptors.length, second.descriptors),
        forward = u32(first.scores.length * 3),
        reverse = u32(second.scores.length * 3);
      const firstControl = {
        firstCount: first.scores.length,
        secondCount: second.scores.length,
        ratio,
        maxDistance,
      };
      const secondControl = {
        firstCount: second.scores.length,
        secondCount: first.scores.length,
        ratio,
        maxDistance,
      };
      const report = await device.plan(() => {
        matchBrief(
          {
            rightFeatures,
            leftDescriptors,
            rightDescriptors,
            matches: forward,
            matchControl: firstControl,
          },
          Math.ceil(first.scores.length / 64),
        );
        matchBrief(
          {
            rightFeatures: leftFeatures,
            leftDescriptors: rightDescriptors,
            rightDescriptors: leftDescriptors,
            matches: reverse,
            matchControl: secondControl,
          },
          Math.ceil(second.scores.length / 64),
        );
      });
      const a = await device.read(forward, first.scores.length * 3),
        b = await device.read(reverse, second.scores.length * 3);
      const rows: number[] = [],
        distances: number[] = [];
      for (let i = 0; i < first.scores.length; i++) {
        const j = a[i * 3]!;
        if (a[i * 3 + 2] === 0 || j >= second.scores.length || b[j * 3 + 2] === 0 || b[j * 3] !== i)
          continue;
        rows.push(i, j);
        distances.push(a[i * 3 + 1]!);
      }
      success = true;
      return { matches: Uint32Array.from(rows), distances: Float64Array.from(distances), report };
    } finally {
      for (const buffer of owned) success ? device.recycle(buffer) : device.drop(buffer);
    }
  });
}
