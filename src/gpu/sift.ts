import type { Resident } from 'typeshade/runtime';
import { ShadeDevice, type ShadePlanReport } from '../device/gpu/device.ts';
import { claheScratchLength, planClahe, recordClahe, type ClaheOptions } from './clahe.ts';
import { rgbaToGray } from './features.shade.ts';
import type { GpuResidentFloatDescriptors } from './float-matching.ts';
import { kltLayout } from './klt.ts';
import { planFlagScan, scanBlockCount } from '../device/gpu/scan.ts';
import {
  siftAdvance,
  siftBlurH,
  siftBlurV,
  siftDescribe,
  siftDog,
  siftDownsample,
  siftFlagExtrema,
  siftMark,
  siftOrient,
  siftScatterExtrema,
  siftScatterOriented,
  siftUpsample,
} from './sift.shade.ts';

/** Three intervals per octave, as the reference implementation. */
const INTERVALS = 3;
const LEVELS = INTERVALS + 3;

export interface GpuSiftOptions {
  /** Octaves to search; the default stops when the smallest side reaches 16 pixels, at most 4. */
  readonly octaves?: number;
  /** Base scale of the first level, defaults to 1.6. The input is assumed to carry 0.5. */
  readonly sigma?: number;
  /** Minimum |DoG| of a refined extremum, on intensities in [0, 1]; defaults to 0.02 / 3. */
  readonly peakThreshold?: number;
  /** Maximum principal-curvature ratio of the spatial Hessian, defaults to 10. */
  readonly edgeThreshold?: number;
  /** Capacity of oriented keypoints, defaults to 4096 and at most 16384. */
  readonly maxFeatures?: number;
  /** Orientations per extremum, 1 or 2; defaults to 2. */
  readonly maxOrientations?: 1 | 2;
  /**
   * RootSIFT (Arandjelović and Zisserman, 2012): after the clipping, the descriptor is
   * L1-normalized and square-rooted, so that Euclidean distances between descriptors are
   * Hellinger distances of the originals. Defaults to true; false gives the reference SIFT
   * descriptor. Either way the descriptor is a non-negative unit vector under L2.
   */
  readonly rootSift?: boolean;
  /**
   * The first octave: 0 (the default) starts at the input size; -1 first upsamples the input
   * 2× (bilinear), as Lowe's reference and COLMAP's `first_octave = -1`, so that features
   * finer than the input's sampling are found. It costs four times the pixels of the base
   * octave in time and working memory. The default number of octaves grows by one with it.
   */
  readonly firstOctave?: -1 | 0;
  /**
   * Contrast-limited adaptive histogram equalization of the gray image before the scale space
   * (scikit-image's `equalize_adapthist`), for low-contrast frames such as bright faces. `true`
   * uses 8 × 8 tiles of 256 bins and a clip limit of 0.01; an object sets `tiles` and
   * `clipLimit`. Defaults to false, which leaves the pipeline as it was.
   */
  readonly clahe?: boolean | ClaheOptions;
  /**
   * Keep the gray image of the frame, before any CLAHE, in a buffer that is long enough for the
   * pyramid of the KLT tracker with this many levels (`docs/klt-tracking.md`, section 4.1). The
   * result's `gray` holds it, and `createKltTracker`'s `push({ sift })` takes it as level 0 of the
   * frame, so the tracker does not convert the RGBA again. Unset keeps the pipeline as it was.
   */
  readonly grayForKlt?: { readonly levels: number };
}
/** The gray frame that SIFT kept for the KLT tracker (`GpuSiftOptions.grayForKlt`). */
export interface GpuSiftGray {
  readonly levels: number;
  readonly width: number;
  readonly height: number;
  /** Level 0 is the first `width x height` values. The buffer is `kltLayout(...).total` long. */
  readonly buffer: Resident<Float32Array>;
  /**
   * Hand the buffer to the caller, which frees it. After this the result's `dispose()` leaves it
   * alone. Throws when the buffer was taken before or the result is disposed.
   */
  take(): Resident<Float32Array>;
}
export interface GpuSiftExecution {
  readonly applicationUploadBytes: number;
  readonly applicationReadbackBytes: number;
}
export interface GpuResidentSift {
  /** Oriented keypoints with a descriptor, in octave, interval, pixel order. */
  readonly count: number;
  /** Extrema the thresholds accepted before the capacity cut, over all orientations. */
  readonly requested: number;
  readonly capacity: number;
  readonly octaves: number;
  readonly report: ShadePlanReport;
  readonly execution: GpuSiftExecution;
  /** Resident 128-d f32 descriptors for `matchFloatDescriptorsGpu`. */
  readonly descriptors: GpuResidentFloatDescriptors;
  /** Image-pixel [x0, y0, x1, y1, …] with integer pixel centres, cached after the first read. */
  readKeypoints(): Promise<Float64Array>;
  /** Eight f32 per keypoint: x, y, sigma, orientation, response, octave, interval, 0. */
  readDetails(): Promise<Float32Array>;
  /** The gray frame kept for the tracker, when `grayForKlt` was set. */
  readonly gray?: GpuSiftGray;
  dispose(): Promise<void>;
}

const validRgba = (rgba: ArrayLike<number>, width: number, height: number) =>
  Number.isSafeInteger(width) &&
  Number.isSafeInteger(height) &&
  width >= 32 &&
  height >= 32 &&
  width <= 4096 &&
  height <= 4096 &&
  rgba.length === width * height * 4;

/** @internal RGBA bytes as the packed u32 words that `rgbaToGray` reads. */
export function packRgba(rgba: Uint8Array | Uint8ClampedArray, pixelCount: number): Uint32Array {
  const littleEndian = new Uint8Array(Uint32Array.of(0x01020304).buffer)[0] === 4;
  if (littleEndian && rgba.byteOffset % 4 === 0 && rgba.byteLength % 4 === 0)
    return new Uint32Array(rgba.buffer, rgba.byteOffset, pixelCount).slice();
  const packed = new Uint32Array(pixelCount);
  for (let i = 0; i < pixelCount; i++) {
    const at = i * 4;
    packed[i] = rgba[at]! | (rgba[at + 1]! << 8) | (rgba[at + 2]! << 16) | (rgba[at + 3]! << 24);
  }
  return packed;
}

/** Scale-space SIFT keypoints and descriptors, resident on the device. */
export function extractSiftGpuResident(
  device: ShadeDevice,
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  options: GpuSiftOptions = {},
): Promise<GpuResidentSift> {
  if (!validRgba(rgba, width, height)) throw new RangeError('Invalid RGBA frame for SIFT');
  const sigma = options.sigma ?? 1.6,
    peak = options.peakThreshold ?? 0.02 / 3,
    edge = options.edgeThreshold ?? 10,
    capacity = options.maxFeatures ?? 4096,
    maxOrientations = options.maxOrientations ?? 2,
    rootSift = options.rootSift ?? true,
    firstOctave = options.firstOctave ?? 0,
    up = firstOctave === -1 ? 2 : 1,
    baseWidth = width * up,
    baseHeight = height * up,
    maxOctaves = Math.max(1, Math.floor(Math.log2(Math.min(baseWidth, baseHeight))) - 4),
    octaves = Math.min(options.octaves ?? 4 + (up - 1), maxOctaves),
    claheOption = options.clahe ?? false,
    grayLevels = options.grayForKlt?.levels;
  if (
    grayLevels !== undefined &&
    !(Number.isInteger(grayLevels) && grayLevels >= 1 && grayLevels <= 8)
  )
    throw new RangeError('Invalid SIFT options');
  if (typeof claheOption !== 'boolean' && (typeof claheOption !== 'object' || claheOption === null))
    throw new RangeError('Invalid SIFT options');
  if (
    !(sigma >= 0.8 && sigma <= 4) ||
    !(peak > 0 && peak < 1) ||
    !(edge >= 1 && edge <= 100) ||
    !Number.isSafeInteger(capacity) ||
    capacity < 1 ||
    capacity > 16384 ||
    (maxOrientations !== 1 && maxOrientations !== 2) ||
    typeof rootSift !== 'boolean' ||
    (firstOctave !== 0 && firstOctave !== -1) ||
    !Number.isSafeInteger(octaves) ||
    octaves < 1
  )
    throw new RangeError('Invalid SIFT options');
  const clahePlan = claheOption === false ? undefined : planClahe(claheOption, width, height);
  // `pixelCount` is the input's; `basePixels` the base octave's (four times it at octave -1).
  const pixelCount = width * height,
    // The raw gray is long enough for the tracker's pyramid when it is kept (section 4.1).
    grayLength = grayLevels === undefined ? pixelCount : kltLayout(width, height, grayLevels).total,
    basePixels = baseWidth * baseHeight,
    k = 2 ** (1 / INTERVALS),
    // Images: six levels, one blur scratch and five differences at the base size, plus flags.
    imageBytes = basePixels * 4 * (LEVELS + 1 + INTERVALS + 2),
    flagBytes = basePixels * 4 * 1.34,
    keyBytes = capacity * 4 * (8 + 8 + 128 + 2 * 2 + 1),
    estimatedPeakBytes = Math.ceil(
      imageBytes +
        flagBytes +
        keyBytes +
        (grayLength + (up - 1) * basePixels) * 4 +
        (clahePlan ? (pixelCount + 2 * claheScratchLength(clahePlan)) * 4 : 0),
    );
  if (basePixels * 4 > device.bindingBytes || capacity * 128 * 4 > device.bindingBytes)
    throw new RangeError('SIFT frame or capacity exceeds the TypeShade storage binding budget');
  if (basePixels > device.maxKernelElements || 2 * capacity > device.maxKernelElements)
    throw new RangeError('SIFT frame exceeds the TypeShade dispatch budget');
  if (estimatedPeakBytes > device.workingBytes)
    throw new RangeError('SIFT frame exceeds the TypeShade working budget');
  const packed = packRgba(rgba, pixelCount);
  return device.enqueue(async () => {
    device.reserveWorkingBytes(estimatedPeakBytes);
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
        rawGray = f32(grayLength),
        gray = clahePlan ? f32(pixelCount) : rawGray,
        claheHist = clahePlan ? u32(claheScratchLength(clahePlan)) : undefined,
        claheMap = clahePlan ? f32(claheScratchLength(clahePlan)) : undefined,
        scratch = f32(basePixels),
        gaussians = f32(basePixels * LEVELS),
        dogs = Array.from({ length: LEVELS - 1 }, () => f32(basePixels)),
        upsampled = up > 1 ? f32(basePixels) : undefined,
        totals = u32(2),
        keyCount = u32(2, new Uint32Array(2)),
        orientedCount = u32(2, new Uint32Array(2)),
        bounds = u32(2),
        orientedBounds = u32(2),
        candidates = f32(capacity * 8),
        oriented = f32(capacity * 8),
        descriptors = f32(capacity * 128),
        orientFlags = u32(capacity * 2),
        orientOffsets = u32(2 * scanBlockCount(capacity * 2));
      // Per octave, flags sized exactly to the octave so that the scan sees no stale element.
      const sizes: { width: number; height: number }[] = [];
      for (
        let o = 0, w = baseWidth, h = baseHeight;
        o < octaves;
        o++, w = Math.ceil(w / 2), h = Math.ceil(h / 2)
      )
        sizes.push({ width: w, height: h });
      const candFlags = sizes.map((size) => u32(size.width * size.height)),
        candOffsets = sizes.map((size) => u32(2 * scanBlockCount(size.width * size.height)));
      const report = await device.plan(() => {
        rgbaToGray(
          { rgba: input, gray: rawGray, pixelWindow: { offset: 0, count: pixelCount } },
          Math.ceil(pixelCount / 64),
        );
        if (clahePlan) recordClahe(clahePlan, rawGray, gray, claheHist!, claheMap!);
        for (let o = 0; o < octaves; o++) {
          const { width: w, height: h } = sizes[o]!,
            pixels = w * h,
            groups = Math.ceil(pixels / 64),
            scale = 2 ** o / up,
            image = { width: w, height: h };
          // Level i of this octave lives at element offset i * basePixels of `gaussians`.
          const blur = (
            from: Resident<Float32Array>,
            inOffset: number,
            outOffset: number,
            s: number,
          ) => {
            const radius = Math.min(16, Math.max(1, Math.ceil(3 * s)));
            siftBlurH(
              {
                blurIn: from,
                blurOut: scratch,
                level: { ...image, sigma: s, radius, inOffset, outOffset: 0 },
              },
              groups,
            );
            siftBlurV(
              {
                blurIn: scratch,
                blurOut: gaussians,
                level: { ...image, sigma: s, radius, inOffset: 0, outOffset },
              },
              groups,
            );
          };
          if (o === 0 && upsampled) {
            // Octave -1: the input's nominal blur of 0.5 pixels is 1 pixel after upsampling.
            siftUpsample(
              {
                levelIn: gray,
                baseOut: upsampled,
                resample: {
                  inWidth: width,
                  inHeight: height,
                  outWidth: w,
                  outHeight: h,
                  inOffset: 0,
                },
              },
              groups,
            );
            blur(upsampled, 0, 0, Math.sqrt(Math.max(sigma * sigma - 1, 0.01)));
          } else if (o === 0) {
            // The input carries a nominal blur of 0.5 pixels.
            blur(gray, 0, 0, Math.sqrt(Math.max(sigma * sigma - 0.25, 0.01)));
          } else {
            const previous = sizes[o - 1]!;
            siftDownsample(
              {
                levelIn: gaussians,
                baseOut: scratch,
                resample: {
                  inWidth: previous.width,
                  inHeight: previous.height,
                  outWidth: w,
                  outHeight: h,
                  inOffset: INTERVALS * basePixels,
                },
              },
              groups,
            );
            // Copy the downsampled base into level 0 through a zero-radius blur.
            siftBlurV(
              {
                blurIn: scratch,
                blurOut: gaussians,
                level: { ...image, sigma: 1, radius: 0, inOffset: 0, outOffset: 0 },
              },
              groups,
            );
          }
          for (let i = 1; i < LEVELS; i++) {
            const previousSigma = sigma * k ** (i - 1),
              nextSigma = sigma * k ** i;
            blur(
              gaussians,
              (i - 1) * basePixels,
              i * basePixels,
              Math.sqrt(nextSigma ** 2 - previousSigma ** 2),
            );
          }
          for (let i = 0; i < LEVELS - 1; i++)
            siftDog(
              {
                gaussianLevels: gaussians,
                dogOut: dogs[i]!,
                image: { ...image, offsetA: i * basePixels, offsetB: (i + 1) * basePixels },
              },
              groups,
            );
          siftMark({ markCounter: keyCount, markBounds: bounds, mark: { slot: 0 } }, 1);
          for (let s = 1; s <= INTERVALS; s++) {
            const control = {
              width: w,
              height: h,
              interval: s,
              octave: o,
              scale,
              sigma0: sigma,
              k,
              peak,
              edge: ((edge + 1) * (edge + 1)) / edge,
              capacity,
            };
            const stack = { dogBelow: dogs[s - 1]!, dogMid: dogs[s]!, dogAbove: dogs[s + 1]! };
            siftFlagExtrema({ ...stack, candFlags: candFlags[o]!, extrema: control }, groups);
            planFlagScan(
              device,
              { flags: candFlags[o]!, blockSums: candOffsets[o]!, totals },
              pixels,
            );
            siftScatterExtrema(
              {
                ...stack,
                candFlags: candFlags[o]!,
                candOffsets: candOffsets[o]!,
                keyCount,
                candidates,
                extrema: control,
              },
              groups,
            );
            siftAdvance(
              { advanceCounter: keyCount, advanceTotals: totals, advance: { capacity } },
              1,
            );
          }
          siftMark({ markCounter: keyCount, markBounds: bounds, mark: { slot: 1 } }, 1);
          const keys = {
            width: w,
            height: h,
            scale,
            sigma0: sigma,
            k,
            intervals: INTERVALS,
            capacity,
            maxOrientations,
            stride: basePixels,
            rootSift: rootSift ? 1 : 0,
          };
          const levelBindings = { gaussians };
          siftMark(
            { markCounter: orientedCount, markBounds: orientedBounds, mark: { slot: 0 } },
            1,
          );
          siftOrient(
            {
              ...levelBindings,
              candidateList: candidates,
              bounds,
              orientFlags,
              orientValues: scratch,
              keys,
            },
            Math.ceil(capacity / 64),
          );
          planFlagScan(
            device,
            { flags: orientFlags, blockSums: orientOffsets, totals },
            capacity * 2,
          );
          siftScatterOriented(
            {
              candidateList: candidates,
              orientFlags,
              orientValues: scratch,
              orientOffsets,
              orientedCount,
              oriented,
              keys,
            },
            Math.ceil((capacity * 2) / 64),
          );
          siftAdvance(
            { advanceCounter: orientedCount, advanceTotals: totals, advance: { capacity } },
            1,
          );
          siftMark(
            { markCounter: orientedCount, markBounds: orientedBounds, mark: { slot: 1 } },
            1,
          );
          siftDescribe(
            {
              ...levelBindings,
              orientedList: oriented,
              bounds: orientedBounds,
              descriptors,
              keys,
            },
            Math.ceil(capacity / 64),
          );
        }
      });
      const counts = await device.read(orientedCount, 2),
        count = counts[0]!,
        requested = counts[1]!;
      if (count > capacity) throw new Error('GPU SIFT compaction returned an invalid count');
      retained.add(oriented);
      retained.add(descriptors);
      let grayTaken = false;
      if (grayLevels !== undefined) retained.add(rawGray);
      const after = device.statistics;
      const execution: GpuSiftExecution = {
        applicationUploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
        applicationReadbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
      };
      let detailsRead: Promise<Float32Array> | undefined,
        keypointsRead: Promise<Float64Array> | undefined,
        disposed = false;
      const readDetails = () => {
        if (disposed) throw new TypeError('GPU SIFT features are disposed');
        detailsRead ??= device.enqueue(() => device.read(oriented, count * 8));
        return detailsRead;
      };
      const result: GpuResidentSift = {
        count,
        requested,
        capacity,
        octaves,
        report,
        execution,
        descriptors: { descriptors, count, dimensions: 128, elements: count * 128 },
        readKeypoints: () => {
          keypointsRead ??= readDetails().then((details) => {
            const keypoints = new Float64Array(count * 2);
            for (let i = 0; i < count; i++) {
              keypoints[i * 2] = details[i * 8]!;
              keypoints[i * 2 + 1] = details[i * 8 + 1]!;
            }
            return keypoints;
          });
          return keypointsRead;
        },
        readDetails,
        ...(grayLevels === undefined
          ? {}
          : {
              gray: {
                levels: grayLevels,
                width,
                height,
                buffer: rawGray,
                take: () => {
                  if (disposed || grayTaken) throw new TypeError('The SIFT gray is not available');
                  grayTaken = true;
                  return rawGray;
                },
              },
            }),
        dispose: async () => {
          if (disposed) return;
          disposed = true;
          await device.enqueue(async () => {
            device.drop(oriented);
            device.drop(descriptors);
            if (grayLevels !== undefined && !grayTaken) device.drop(rawGray);
          });
        },
      };
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
