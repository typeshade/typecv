import type { Resident } from 'typeshade/runtime';
import { ShadeDevice } from '../device/gpu/device.ts';
import {
  registerResidentFeatures,
  type GpuFeatureExecution,
  type GpuResidentFeatures,
} from './features.ts';
import { packRgba } from './sift.ts';
import { planFlagScan, scanBlockCount } from '../device/gpu/scan.ts';
import {
  orbAdvance,
  orbDescribe,
  orbEmit,
  orbFast,
  orbFlagCorners,
  orbGray,
  orbHarris,
  orbMark,
  orbPickTop,
  orbResize,
  orbScatterCorners,
  orbScatterFast,
} from './orb.shade.ts';
import {
  ORB_BORDER,
  ORB_DEFAULTS,
  ORB_FAST_BINS,
  ORB_HARRIS_BINS,
  ORB_HARRIS_K,
  ORB_PATCH,
  ORB_PATTERN,
  ORB_UMAX,
  ORB_BLUR_KERNEL,
  orbLevelLayout,
  orbResizeTable,
  type OrbParameters,
} from './orb-reference.ts';

export interface GpuOrbOptions {
  /**
   * Features kept over all levels, split over the levels as OpenCV's `ORB_create` does: a
   * geometric share, the finest level the most. Defaults to 1000 (OpenCV's own default is 500)
   * and is at most 8000 (the tiled Hamming matcher has no pair cap).
   */
  readonly maxFeatures?: number;
  /** Pyramid levels, defaults to 8 (at most 12). Levels under 63 pixels on a side hold no keypoints. */
  readonly levels?: number;
  /** Size ratio of two levels, defaults to 1.2 (above 1, at most 2). */
  readonly scaleFactor?: number;
  /** FAST-9 threshold on intensities in 0..255, defaults to 20 (1 to 254). */
  readonly fastThreshold?: number;
}
/**
 * ORB keypoints and 256-bit descriptors on the device. It is a `GpuResidentFeatures`: the
 * corner front end's matcher, `matchResidentFeaturesGpu`, matches it, and the mapper takes the
 * keypoints and matches as it takes SIFT's.
 */
export interface GpuResidentOrb extends GpuResidentFeatures {
  /** Corners that passed the FAST test and the suppression, over all levels, before ranking. */
  readonly requested: number;
  readonly levels: number;
  /** Eight f32 per keypoint: x, y, size (31 times the level scale), angle (radians), Harris response, level, 0, 0. */
  readDetails(): Promise<Float32Array>;
}

const validRgba = (rgba: ArrayLike<number>, width: number, height: number) =>
  Number.isSafeInteger(width) &&
  Number.isSafeInteger(height) &&
  width >= 64 &&
  height >= 64 &&
  width <= 4096 &&
  height <= 4096 &&
  rgba.length === width * height * 4;

/** The largest corner list per level; corners beyond it are cut in raster order and counted in `requested`. */
const LIST_CAPACITY = 16384;

/**
 * ORB features (Rublee et al. 2011) on the device, after OpenCV's `ORB_create`: FAST-9 corners
 * with a 3 x 3 suppression on an eight-level pyramid, kept by Harris score, oriented by the
 * intensity centroid and described by the learned 256-bit rBRIEF pattern. On the CPU tier the
 * result equals `extractOrbCpu`.
 */
export function extractOrbGpuResident(
  device: ShadeDevice,
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  options: GpuOrbOptions = {},
): Promise<GpuResidentOrb> {
  const parameters: OrbParameters = {
    maxFeatures: options.maxFeatures ?? ORB_DEFAULTS.maxFeatures,
    levels: options.levels ?? ORB_DEFAULTS.levels,
    scaleFactor: options.scaleFactor ?? ORB_DEFAULTS.scaleFactor,
    fastThreshold: options.fastThreshold ?? ORB_DEFAULTS.fastThreshold,
  };
  if (!validRgba(rgba, width, height)) throw new RangeError('Invalid RGBA frame for ORB');
  if (
    !Number.isSafeInteger(parameters.maxFeatures) ||
    parameters.maxFeatures < 1 ||
    parameters.maxFeatures > 8000 ||
    !Number.isSafeInteger(parameters.levels) ||
    parameters.levels < 1 ||
    parameters.levels > 12 ||
    !(parameters.scaleFactor > 1 && parameters.scaleFactor <= 2) ||
    !Number.isSafeInteger(parameters.fastThreshold) ||
    parameters.fastThreshold < 1 ||
    parameters.fastThreshold > 254
  )
    throw new RangeError('Invalid ORB options');
  const layout = orbLevelLayout(width, height, parameters),
    pixelCount = width * height,
    pyramidPixels = layout.reduce((sum, level) => sum + level.width * level.height, 0),
    // A level has keypoints only where a 31-pixel border leaves some pixels.
    active = layout.filter(
      (level) => level.quota > 0 && level.width > 2 * ORB_BORDER && level.height > 2 * ORB_BORDER,
    ),
    capacity = layout.reduce((sum, level) => sum + level.quota, 0),
    listCapacity = Math.max(
      1,
      Math.min(LIST_CAPACITY, (width - 2 * ORB_BORDER) * (height - 2 * ORB_BORDER)),
    ),
    // Input, pyramid, scores and per-level flags; lists; outputs.
    estimatedPeakBytes = Math.ceil(
      4 * (pixelCount + 3 * pyramidPixels) +
        4 * (listCapacity * 12 + capacity * 24) +
        4 * 2 * ORB_HARRIS_BINS +
        65536,
    );
  if (pyramidPixels * 4 > device.bindingBytes)
    throw new RangeError('ORB frame exceeds the TypeShade storage binding budget');
  if (pixelCount > device.maxKernelElements || listCapacity > device.maxKernelElements)
    throw new RangeError('ORB frame exceeds the TypeShade dispatch budget');
  if (estimatedPeakBytes > device.workingBytes)
    throw new RangeError('ORB frame exceeds the TypeShade working budget');
  const packed = packRgba(rgba, pixelCount),
    tables: number[] = [],
    tableOffsets = layout.map((level, index) => {
      if (index === 0) return [0, 0] as const;
      const previous = layout[index - 1]!,
        x = orbResizeTable(previous.width, level.width),
        y = orbResizeTable(previous.height, level.height),
        at = [tables.length, tables.length + x.length] as const;
      tables.push(...x, ...y);
      return at;
    });
  const math = {
    p1: Math.fround(Math.fround(0.9997878412794807) * Math.fround(180 / Math.PI)),
    p3: Math.fround(Math.fround(-0.3258083974640975) * Math.fround(180 / Math.PI)),
    p5: Math.fround(Math.fround(0.1555786518463281) * Math.fround(180 / Math.PI)),
    p7: Math.fround(Math.fround(-0.04432655554792128) * Math.fround(180 / Math.PI)),
    eps: Math.fround(1e-5),
    radians: Math.fround(Math.PI / 180),
    harrisK: Math.fround(ORB_HARRIS_K),
    harrisScale: (() => {
      const f = Math.fround,
        scale = f(1 / (4 * 7 * 255));
      return f(f(f(scale * scale) * scale) * scale);
    })(),
    k0: ORB_BLUR_KERNEL[0]!,
    k1: ORB_BLUR_KERNEL[1]!,
    k2: ORB_BLUR_KERNEL[2]!,
    k3: ORB_BLUR_KERNEL[3]!,
  };
  return device.enqueue(async () => {
    device.reserveWorkingBytes(estimatedPeakBytes);
    const before = device.statistics;
    const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
    const retained = new Set<Resident<Float32Array> | Resident<Uint32Array>>();
    let success = false;
    const u32 = (length: number, input?: Uint32Array) => {
      const value = device.allocateU32(Math.max(1, length), input);
      owned.push(value);
      return value;
    };
    const f32 = (length: number, input?: Float32Array) => {
      const value = device.allocate(Math.max(1, length), input);
      owned.push(value);
      return value;
    };
    try {
      const input = u32(pixelCount, packed),
        pyr = u32(pyramidPixels),
        scores = u32(pyramidPixels),
        resizeTable = u32(tables.length, Uint32Array.from(tables)),
        patternTable = f32(1024, Float32Array.from(ORB_PATTERN)),
        umaxTable = u32(16, Uint32Array.from(ORB_UMAX).subarray(0, 16)),
        totalsC = u32(2),
        totalsA = u32(2),
        totalsB = u32(2),
        cornerList = u32(listCapacity * 3),
        fastList = u32(listCapacity * 3),
        harrisValue = f32(listCapacity),
        harrisKey = u32(listCapacity),
        flagsA = u32(listCapacity),
        flagsB = u32(listCapacity),
        offsetsA = u32(2 * scanBlockCount(listCapacity)),
        offsetsB = u32(2 * scanBlockCount(listCapacity)),
        histogram = u32(Math.max(ORB_FAST_BINS, ORB_HARRIS_BINS)),
        outCount = u32(2, new Uint32Array(2)),
        bounds = u32(2),
        picked = f32(capacity * 4),
        points = f32(capacity * 4),
        details = f32(capacity * 8),
        descriptors = u32(capacity * 8),
        cornerFlags = active.map((level) => u32(level.width * level.height)),
        cornerOffsets = active.map((level) => u32(2 * scanBlockCount(level.width * level.height)));
      const control = (level: (typeof layout)[number]) => {
        const source = layout[Math.max(0, level.index - 1)]!;
        return {
          width: level.width,
          height: level.height,
          offset: level.offset,
          srcWidth: source.width,
          srcHeight: source.height,
          srcOffset: source.offset,
          xTable: tableOffsets[level.index]![0],
          yTable: tableOffsets[level.index]![1],
          threshold: parameters.fastThreshold,
          capacity: listCapacity,
          level: level.index,
          scaleX: level.scaleX,
          scaleY: level.scaleY,
          size: Math.fround(ORB_PATCH * level.scale),
          outCapacity: capacity,
        };
      };
      const report = await device.plan(() => {
        const base = layout[0]!;
        orbGray({ rgba: input, pyr, lv: control(base) }, Math.ceil(pixelCount / 64));
        for (let l = 1; l < layout.length; l++) {
          const level = layout[l]!;
          orbResize(
            { pyr, resizeTable, lv: control(level) },
            Math.ceil((level.width * level.height) / 64),
          );
        }
        active.forEach((level, slot) => {
          const lv = control(level),
            pixels = level.width * level.height,
            groups = Math.ceil(pixels / 64),
            listGroups = Math.ceil(listCapacity / 64),
            flags = cornerFlags[slot]!,
            offsets = cornerOffsets[slot]!;
          orbFast({ pyr, scores, lv }, groups);
          orbFlagCorners({ scores, cornerFlags: flags, lv }, groups);
          planFlagScan(device, { flags, blockSums: offsets, totals: totalsC }, pixels);
          orbScatterCorners(
            { scores, cornerFlags: flags, cornerOffsets: offsets, cornerList, lv },
            groups,
          );
          // Twice the quota by FAST score.
          orbPickTop(
            {
              pickCount: totalsC,
              pickHist: histogram,
              pickFlags: flagsA,
              cornerList,
              harrisKey,
              pick: {
                stage: 0,
                quota: 2 * level.quota,
                bins: ORB_FAST_BINS,
                capacity: listCapacity,
              },
            },
            1,
          );
          planFlagScan(
            device,
            { flags: flagsA, blockSums: offsetsA, totals: totalsA },
            listCapacity,
          );
          orbScatterFast(
            { pickFlags: flagsA, pickOffsets: offsetsA, cornerList, fastList, lv },
            listGroups,
          );
          orbHarris(
            { pyr, fastList, pickTotals: totalsA, harrisKey, harrisValue, orbMath: math, lv },
            listGroups,
          );
          // The quota by Harris score.
          orbPickTop(
            {
              pickCount: totalsA,
              pickHist: histogram,
              pickFlags: flagsB,
              cornerList,
              harrisKey,
              pick: {
                stage: 1,
                quota: level.quota,
                bins: ORB_HARRIS_BINS,
                capacity: listCapacity,
              },
            },
            1,
          );
          planFlagScan(
            device,
            { flags: flagsB, blockSums: offsetsB, totals: totalsB },
            listCapacity,
          );
          orbMark({ markCounter: outCount, markBounds: bounds, mark: { slot: 0 } }, 1);
          orbEmit(
            {
              pickFlags: flagsB,
              pickOffsets: offsetsB,
              fastList,
              harrisValue,
              outCount,
              picked,
              lv,
            },
            listGroups,
          );
          orbAdvance(
            {
              advanceCounter: outCount,
              advanceTotals: totalsB,
              advanceRequested: totalsC,
              advance: { capacity },
            },
            1,
          );
          orbMark({ markCounter: outCount, markBounds: bounds, mark: { slot: 1 } }, 1);
          orbDescribe(
            {
              pyr,
              picked,
              bounds,
              patternTable,
              umaxTable,
              points,
              details,
              descriptors,
              orbMath: math,
              lv,
            },
            Math.ceil(capacity / 64),
          );
        });
      });
      const counts = await device.read(outCount, 2),
        count = counts[0]!,
        requested = counts[1]!;
      if (count > capacity) throw new Error('GPU ORB compaction returned an invalid count');
      retained.add(points);
      retained.add(descriptors);
      retained.add(details);
      const after = device.statistics,
        execution: GpuFeatureExecution = {
          applicationUploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
          applicationReadbackBytes:
            after.applicationReadbackBytes - before.applicationReadbackBytes,
          residentSnapshotBytes: after.residentSnapshotBytes - before.residentSnapshotBytes,
        },
        resident = registerResidentFeatures(
          device,
          count,
          capacity,
          points,
          descriptors,
          report,
          execution,
          before,
        );
      let detailsRead: Promise<Float32Array> | undefined;
      const baseDispose = resident.dispose,
        result = Object.assign(resident, {
          requested,
          levels: layout.length,
          readDetails: () => {
            detailsRead ??= device
              .enqueue(() => device.read(details, count * 8))
              .then((values) => values.slice(0, count * 8));
            return detailsRead;
          },
          dispose: async () => {
            await baseDispose();
            await device.enqueue(async () => device.drop(details));
          },
        }) as GpuResidentOrb;
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
