/**
 * The photometric depth stage of the dense MVS on one reference view (`docs/dense-mvs.md`,
 * sections 3.4 to 3.7, 4.1 and 5.1, step 2 of section 8): `mvsUndistort` writes the undistorted
 * gray image of every packed view, `mvsInitialize` and `mvsSeed` give the reference its first
 * hypotheses, and `mvsPropagate` runs the red-black iterations of the photometric pass. The
 * kernels are in `mvs.shade.ts`. The host arrays come from `planMvsViews` (`mvs-views.ts`).
 *
 * This module is internal until step 5, which adds `reconstructDenseGpu` and its export.
 * `src/gpu/index.ts` does not export it. The tests reach it through `internal.ts`.
 *
 * Details that the document leaves open, with the choice made here:
 * - All packed views have one size W × H, so that a per-view array is v × P + i (4.1). A plan
 *   with two sizes is refused.
 * - A view without distortion (k1 = k2 = 0) runs `mvsUndistort` in its copy mode, which writes
 *   the gray image unchanged. Resampling at the identity map in f32 can land a hair past the
 *   last row or column and mark it "no data".
 * - A reference sample outside the image or without data is left out of the window. A window
 *   with fewer than a quarter of its samples (9 of 36), or whose own pixel has no data, costs 2.
 * - A source with a window variance below 1e-6 costs 2, as the reference does (3.7 names only
 *   the reference).
 * - The bilateral weight of a sample at offset (dx, dy) with reference value a is
 *   exp(−(dx² + dy²) / (2 σs²) − (a − a_p)² / (2 σc²)), with a_p the value of the pixel itself,
 *   σs = `windowRadius` and σc = 0.2 (COLMAP's `sigma_spatial` default and `sigma_color`).
 * - A seed's normal faces the camera along the pixel's ray (−ray / |ray|).
 * - A propagation candidate whose moved depth falls outside [dmin, dmax], or whose plane the
 *   pixel's ray meets from behind, is not evaluated.
 * - The perturbed candidate perturbs the best hypothesis after the propagation and the random
 *   candidate. Iterations are numbered 1..T, so the first perturbation is half the inverse-depth
 *   range and 15°. The normal turns by a uniform angle in [0, 30° × 2^−t] towards a uniform
 *   random tangent. A turned normal that faces away from the camera keeps the old one.
 * - Every evaluated normal is the decoded octahedral word that is stored, so a stored cost is
 *   the cost of the stored hypothesis.
 * - Ties keep the earlier hypothesis: the current one, then the candidates in the order of 3.6.
 * - Draws: iteration 0 (initialization) uses draws 0 to 2 (depth, normal). Iteration t uses
 *   draws 0 to 2 for the random candidate and 3 to 6 for the perturbation (depth, tangent,
 *   angle). The hash input is (seed, reference, pixel index, 16 t + draw).
 * - Chunks (5.1, step 3): `mvsInitialize` and `mvsPropagate` run as chunks under the keys of
 *   the same names, one unit a pixel and a source, each with its warm step. The device sizes
 *   them from the time budget. `chunkPixels` caps a chunk at that many pixels (initialization)
 *   or color indices (propagation). The CPU tier runs each pass as one chunk.
 */
import type { Resident } from 'typeshade/runtime';
import { radialLimitSquared } from '../geometry/camera.ts';
import { ShadeDevice, type ShadePlanReport } from './device.ts';
import { rgbaToGray } from './features.shade.ts';
import { MVS_CAMERA_FLOATS, MVS_SEED_FLOATS, MVS_SOURCE_SLOTS } from './mvs-views.ts';
import type { MvsScene, MvsViewPlan } from './mvs-views.ts';
import { mvsInitialize, mvsPropagate, mvsSeed, mvsUndistort } from './mvs.shade.ts';
import { packRgba } from './sift.ts';

/** The RGBA bytes of one view at its analysis size. */
export interface MvsImage {
  readonly rgba: Uint8Array | Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
}

export interface MvsPhotometricOptions {
  /** The least source costs averaged per hypothesis (3.7). Default 3. */
  readonly bestSources?: number;
  /** Window radius in pixels (3.7). Default 5. */
  readonly windowRadius?: number;
  /** Window sample step in pixels (3.7). Default 2. */
  readonly windowStep?: number;
  /** Red-black iterations of the photometric pass (3.6). Default 8. */
  readonly photometricIterations?: number;
  /** COLMAP's bilateral weights (3.7). Default false. */
  readonly bilateral?: boolean;
  /** The seed of the counter-based random numbers (3.5). Default 0. */
  readonly seed?: number;
  /** Write the plan's seeds over the random initialization (3.5). Default true. */
  readonly seeds?: boolean;
  /** Force chunks of at most this many pixels or color indices. Default: one chunk per pass. */
  readonly chunkPixels?: number;
  /** Read the reference's arrays after the initialization and after each iteration. */
  readonly snapshots?: boolean;
  /** Read the undistorted gray images of all packed views. */
  readonly readGray?: boolean;
}

export interface MvsDepthSnapshot {
  /** Optical-axis depth in normalized units, P values. */
  readonly depth: Float32Array;
  /** Octahedral normals in the reference's camera frame (`decodeMvsNormal`), P values. */
  readonly normal: Uint32Array;
  /** The aggregated cost of each pixel's hypothesis in [0, 2], P values. */
  readonly cost: Float32Array;
}

export interface MvsPhotometricResult extends MvsDepthSnapshot {
  readonly width: number;
  readonly height: number;
  /** The reference's packed index. */
  readonly reference: number;
  /** After the initialization (index 0) and after each iteration t (index t). */
  readonly snapshots?: readonly MvsDepthSnapshot[];
  /** `mvsGray`, N × P values, when `readGray` is set. */
  readonly gray?: Float32Array;
  readonly report: ShadePlanReport;
}

const DEFAULTS = {
  bestSources: 3,
  windowRadius: 5,
  windowStep: 2,
  photometricIterations: 8,
  bilateral: false,
  seed: 0,
  seeds: true,
} as const;
/** COLMAP's `sigma_color` default (section 3.7). */
const SIGMA_COLOR = 0.2;
/** The least triangulation angle of a source at the point, degrees (3.7). */
const MIN_ANGLE_DEG = 1;
/** The largest perturbation of a normal, degrees, before the 2^−t factor (3.6). */
const PERTURB_ANGLE_DEG = 30;
/** The most window samples in one axis: radius 5 at step 1. */
const MAX_AXIS_SAMPLES = 11;
const LANES = 64;
/** The chunk keys of the kernels this module runs (section 5.1). */
export type MvsChunkKey = 'mvsInitialize' | 'mvsPropagate';

function checkInteger(value: number, low: number, high: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < low || value > high)
    throw new RangeError(`${name} must be an integer in [${low}, ${high}]`);
}

/** The unit normal of an octahedral word of `mvsNormal`, as `decodeNormal` in the kernels. */
export function decodeMvsNormal(word: number): [number, number, number] {
  const snorm = (bits: number) => Math.max(((bits << 16) >> 16) / 32767, -1);
  const px = snorm(word & 0xffff),
    py = snorm(word >>> 16);
  const z = 1 - Math.abs(px) - Math.abs(py),
    t = Math.max(-z, 0);
  const x = px + (px >= 0 ? -t : t),
    y = py + (py >= 0 ? -t : t);
  const length = Math.hypot(x, y, z);
  return [x / length, y / length, z / length];
}

/**
 * Runs the photometric pass of section 3.6 on the reference with packed index `reference`:
 * the undistortion of every packed view, the initialization, the seeds and
 * `photometricIterations` red-black iterations. `scene` is the scene `plan` was made from (its
 * k1 and k2 drive the undistortion). `images` maps a view id to its RGBA.
 */
export function mvsPhotometricGpu(
  device: ShadeDevice,
  scene: MvsScene,
  plan: MvsViewPlan,
  images: ReadonlyMap<number, MvsImage>,
  reference: number,
  options: MvsPhotometricOptions = {},
): Promise<MvsPhotometricResult> {
  const o = { ...DEFAULTS, ...options };
  const N = plan.packed.length;
  checkInteger(reference, 0, N - 1, 'reference');
  checkInteger(o.bestSources, 1, MVS_SOURCE_SLOTS - 1, 'bestSources');
  checkInteger(o.windowRadius, 1, 5, 'windowRadius');
  checkInteger(o.windowStep, 1, o.windowRadius, 'windowStep');
  checkInteger(o.photometricIterations, 0, 64, 'photometricIterations');
  checkInteger(o.seed, 0, 0xffffffff, 'seed');
  if (o.chunkPixels !== undefined) checkInteger(o.chunkPixels, 1, 2 ** 31, 'chunkPixels');
  const axisSamples = Math.floor((2 * o.windowRadius) / o.windowStep) + 1;
  if (axisSamples > MAX_AXIS_SAMPLES) throw new RangeError('The window has too many samples');
  const width = plan.cameras[16]!,
    height = plan.cameras[17]!;
  for (let v = 0; v < N; v++) {
    const base = v * MVS_CAMERA_FLOATS;
    if (plan.cameras[base + 16] !== width || plan.cameras[base + 17] !== height)
      throw new RangeError('Every packed view needs one image size');
  }
  if (!(plan.sources[reference * MVS_SOURCE_SLOTS]! > 0))
    throw new RangeError('The reference has no source view');
  const P = width * height;
  const packedImages = plan.packed.map((r) => {
    const view = scene.views[r]!,
      image = images.get(view.id);
    if (!image || image.width !== width || image.height !== height || image.rgba.length !== 4 * P)
      throw new RangeError(`View ${view.id} needs a ${width} × ${height} RGBA image`);
    const k1 = view.camera.k1 ?? 0,
      k2 = view.camera.k2 ?? 0;
    return { rgba: packRgba(image.rgba, P), k1, k2, camera: view.camera };
  });
  const seedFirst = plan.seedOffsets[reference]!,
    seedEnd = o.seeds ? plan.seedOffsets[reference + 1]! : seedFirst;
  const mvsControl = {
    width,
    height,
    reference,
    color: 0,
    iteration: 0,
    seed: o.seed,
    bestSources: o.bestSources,
    windowRadius: o.windowRadius,
    windowStep: o.windowStep,
    axisSamples,
    minSamples: Math.ceil((axisSamples * axisSamples) / 4),
    bilateral: o.bilateral ? 1 : 0,
    spatialFalloff: 1 / (2 * o.windowRadius * o.windowRadius),
    colorFalloff: 1 / (2 * SIGMA_COLOR * SIGMA_COLOR),
    minAngleCos: Math.cos((MIN_ANGLE_DEG * Math.PI) / 180),
    perturbDepth: 1,
    perturbAngle: 0,
  };
  return device.enqueue(async () => {
    const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
    let successful = false;
    const f32 = (n: number, values?: Float32Array) => {
      const r = device.allocate(n, values);
      owned.push(r);
      return r;
    };
    const u32 = (n: number, values?: Uint32Array) => {
      const r = device.allocateU32(n, values);
      owned.push(r);
      return r;
    };
    try {
      // 4.2: the depth stage's (16 N + 4) bytes per pixel, less `mvsGeoDepth` (step 4), plus one
      // view's RGBA upload and its distorted gray image.
      device.reserveWorkingBytes((12 * N + 4) * P + 8 * P);
      const mvsCameras = f32(plan.cameras.length, plan.cameras),
        mvsSources = u32(plan.sources.length, plan.sources),
        mvsSeeds = f32(
          Math.max(MVS_SEED_FLOATS, plan.seeds.length),
          plan.seeds.length ? plan.seeds : new Float32Array(MVS_SEED_FLOATS),
        ),
        mvsGray = f32(N * P),
        mvsDepth = f32(N * P),
        mvsNormal = u32(N * P),
        mvsCost = f32(P),
        gray = f32(P);
      // 3.4: one view at a time, a plan each, through its RGBA upload and its distorted gray.
      for (let v = 0; v < N; v++) {
        const view = packedImages[v]!,
          rgba = device.allocateU32(P, view.rgba);
        try {
          await device.plan(() => {
            rgbaToGray({ rgba, gray, pixelWindow: { offset: 0, count: P } }, Math.ceil(P / LANES));
            mvsUndistort(
              {
                undistortInput: gray,
                undistortOutput: mvsGray,
                undistortControl: {
                  width,
                  height,
                  view: v,
                  distorted: view.k1 !== 0 || view.k2 !== 0 ? 1 : 0,
                  fx: view.camera.fx,
                  fy: view.camera.fy,
                  cx: view.camera.cx,
                  cy: view.camera.cy,
                  k1: view.k1,
                  k2: view.k2,
                  radialLimit: Math.min(1e30, radialLimitSquared(view.k1, view.k2)),
                },
              },
              Math.ceil(P / LANES),
            );
          });
          device.recycle(rgba);
        } catch (error) {
          device.drop(rgba);
          throw error;
        }
      }
      const snapshots: MvsDepthSnapshot[] = [];
      const snapshot = async () => {
        const depth = await device.read(mvsDepth, N * P),
          normal = await device.read(mvsNormal, N * P),
          cost = await device.read(mvsCost, P);
        snapshots.push({
          depth: depth.slice(reference * P, (reference + 1) * P),
          normal: normal.slice(reference * P, (reference + 1) * P),
          cost: cost.slice(),
        });
      };
      // 5.1: a chunk is a contiguous range of pixels (initialization) or color indices
      // (propagation), a plan of its own that the device fences and times under the key of its
      // kernel. One unit is one pixel and one source, so a chunk of n elements holds n × K units.
      // The generator sizes each chunk after the previous one ran, from the rate it measured. A
      // chunk before the last fills its workgroups. `chunkPixels` caps the size. The CPU tier
      // runs each pass as one chunk (`chunkSize` returns `max` there).
      const K = plan.sources[reference * MVS_SOURCE_SLOTS]!;
      function* ranges(key: MvsChunkKey, total: number) {
        for (let first = 0; first < total;) {
          const remaining = total - first;
          let n = Math.min(
            remaining,
            Math.floor(device.chunkSize(key, remaining * K, Math.min(remaining, LANES) * K) / K),
          );
          if (n < remaining) n = Math.max(LANES, n - (n % LANES));
          n = Math.min(n, remaining, o.chunkPixels ?? Infinity);
          yield { first, end: first + n };
          first += n;
        }
      }
      const shared = { mvsCameras, mvsSources, mvsGray, mvsDepth, mvsNormal, mvsCost };
      const T = o.photometricIterations,
        colorIndices = Math.ceil(width / 2) * height,
        empty = { first: 0, end: 0 };
      // The warm step of each key (`GpuChunk.warm`): its entry over zero workgroups, so that the
      // pipeline is made before the first timed chunk. No invocation runs.
      const warmInitialize = {
          id: 'mvs',
          run: () => mvsInitialize({ ...shared, mvsControl, mvsRange: empty }, 0),
        },
        warmPropagate = {
          id: 'mvs',
          run: () => mvsPropagate({ ...shared, mvsControl, mvsRange: empty }, 0),
        };
      function* steps() {
        // 3.5: the random hypotheses, then the seeds over them.
        for (const mvsRange of ranges('mvsInitialize', P)) {
          const n = mvsRange.end - mvsRange.first;
          yield [
            'mvsInitialize',
            () => mvsInitialize({ ...shared, mvsControl, mvsRange }, Math.ceil(n / LANES)),
            undefined,
            { key: 'mvsInitialize', units: n * K, warm: warmInitialize },
          ] as const;
        }
        if (seedEnd > seedFirst)
          yield [
            'mvsSeed',
            () =>
              mvsSeed(
                { ...shared, mvsSeeds, mvsControl, mvsRange: { first: seedFirst, end: seedEnd } },
                Math.ceil((seedEnd - seedFirst) / LANES),
              ),
            o.snapshots ? snapshot : undefined,
          ] as const;
        else if (o.snapshots) yield ['snapshot', () => {}, snapshot] as const;
        // 3.6: T iterations of a red pass and a black pass.
        for (let t = 1; t <= T; t++)
          for (let color = 0; color < 2; color++) {
            const control = {
              ...mvsControl,
              color,
              iteration: t,
              perturbDepth: 2 ** -t,
              perturbAngle: ((PERTURB_ANGLE_DEG * Math.PI) / 180) * 2 ** -t,
            };
            for (const mvsRange of ranges('mvsPropagate', colorIndices)) {
              const n = mvsRange.end - mvsRange.first,
                last = color === 1 && mvsRange.end === colorIndices;
              yield [
                'mvsPropagate',
                () =>
                  mvsPropagate({ ...shared, mvsControl: control, mvsRange }, Math.ceil(n / LANES)),
                o.snapshots && last ? snapshot : undefined,
                { key: 'mvsPropagate', units: n * K, warm: warmPropagate },
              ] as const;
            }
          }
      }
      const report = await device.planSteps(steps());
      const depth = await device.read(mvsDepth, N * P),
        normal = await device.read(mvsNormal, N * P),
        cost = await device.read(mvsCost, P),
        grayImages = o.readGray ? (await device.read(mvsGray, N * P)).slice() : undefined;
      successful = true;
      return {
        width,
        height,
        reference,
        depth: depth.slice(reference * P, (reference + 1) * P),
        normal: normal.slice(reference * P, (reference + 1) * P),
        cost: cost.slice(),
        ...(o.snapshots ? { snapshots } : {}),
        ...(grayImages ? { gray: grayImages } : {}),
        report,
      };
    } finally {
      for (const b of owned) successful ? device.recycle(b) : device.drop(b);
    }
  });
}
