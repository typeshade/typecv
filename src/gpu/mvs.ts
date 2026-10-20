/**
 * The depth stage of the dense MVS (`docs/dense-mvs.md`, sections 3.4 to 3.9, 4.1 and 5.1, steps
 * 2 to 4 of section 8): `mvsUndistort` writes the undistorted gray image of every packed view,
 * `mvsInitialize` and `mvsSeed` give a reference its first hypotheses, and `mvsPropagate` runs
 * the red-black iterations of the photometric pass. `mvsPhotometricGpu` runs them on one
 * reference. `mvsDepthMapsGpu` runs them on every reference, then the geometric pass
 * (`mvsGeometricInitialize`, `mvsPropagate` with `geometric` 1) and `mvsFilter` on each. The
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
 * - The geometric pass (3.8, step 4) keeps the reference's hypotheses in `mvsGeoDepth` and
 *   `mvsNormal`, and reads the sources' photometric depths in `mvsDepth`. It never reads another
 *   view's `mvsGeoDepth` or `mvsNormal`, so the order of the references changes no result.
 *   `mvsGeometricInitialize` copies the photometric depth and computes the cost with the
 *   geometric term. The iterations hash as T + 1..T + G and perturb from 2^−1 again.
 * - ψ_s back-projects the source's depth at the nearest pixel of q along the ray of q itself,
 *   not of the pixel center, as COLMAP does. A q outside the source, a source pixel without
 *   depth or a point behind a camera gives ψmax.
 * - `mvsFilter` recomputes the photometric source costs of the final hypothesis (the stored
 *   cost holds the geometric term), and its ψ_s reads the sources' photometric depths, as the
 *   geometric pass does. It writes depth 0 over a rejected pixel of `mvsGeoDepth`.
 * - `mvsDepthMapsGpu` schedules the geometric pass under `mvsGeometricInitialize` and
 *   `mvsPropagateGeometric`, and the filter under `mvsFilter`, by the rule of step 3.
 */
import type { Resident } from 'typeshade/runtime';
import { radialLimitSquared } from '../geometry/camera.ts';
import { ShadeDevice, type GpuChunk, type ShadePlanReport } from './device.ts';
import { rgbaToGray } from './features.shade.ts';
import { MVS_CAMERA_FLOATS, MVS_SEED_FLOATS, MVS_SOURCE_SLOTS } from './mvs-views.ts';
import type { MvsScene, MvsViewPlan } from './mvs-views.ts';
import {
  mvsFilter,
  mvsGeometricInitialize,
  mvsInitialize,
  mvsPropagate,
  mvsSeed,
  mvsUndistort,
} from './mvs.shade.ts';
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
/** λ and ψmax of the geometric cost, COLMAP's `geom_consistency_regularizer` and
 *  `geom_consistency_max_cost` (section 3.8). */
const GEOMETRIC_WEIGHT = 0.3;
const GEOMETRIC_MAX_COST = 3;
/** The chunk keys of the kernels this module runs (section 5.1). */
export type MvsChunkKey =
  | 'mvsInitialize'
  | 'mvsPropagate'
  | 'mvsGeometricInitialize'
  | 'mvsPropagateGeometric'
  | 'mvsFilter';

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

interface MvsPacked {
  readonly rgba: Uint32Array;
  readonly k1: number;
  readonly k2: number;
  readonly camera: MvsScene['views'][number]['camera'];
}

/** The checked options, the image size and the packed RGBA of the views, and the base control. */
function prepareMvs(
  scene: MvsScene,
  plan: MvsViewPlan,
  images: ReadonlyMap<number, MvsImage>,
  options: MvsPhotometricOptions,
) {
  const o = { ...DEFAULTS, ...options };
  const N = plan.packed.length;
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
  const P = width * height;
  const packedImages: MvsPacked[] = plan.packed.map((r) => {
    const view = scene.views[r]!,
      image = images.get(view.id);
    if (!image || image.width !== width || image.height !== height || image.rgba.length !== 4 * P)
      throw new RangeError(`View ${view.id} needs a ${width} × ${height} RGBA image`);
    const k1 = view.camera.k1 ?? 0,
      k2 = view.camera.k2 ?? 0;
    return { rgba: packRgba(image.rgba, P), k1, k2, camera: view.camera };
  });
  const mvsControl = {
    width,
    height,
    reference: 0,
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
    geometric: 0,
    geometricWeight: GEOMETRIC_WEIGHT,
    geometricMaxCost: GEOMETRIC_MAX_COST,
  };
  return { o, N, P, width, height, packedImages, mvsControl };
}

/** Section 3.4: one view at a time, a plan each, through its RGBA upload and its distorted gray. */
async function undistortViews(
  device: ShadeDevice,
  packedImages: readonly MvsPacked[],
  gray: Resident<Float32Array>,
  mvsGray: Resident<Float32Array>,
  width: number,
  height: number,
): Promise<void> {
  const P = width * height;
  for (let v = 0; v < packedImages.length; v++) {
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
  const { o, N, P, width, height, packedImages, mvsControl } = prepareMvs(
    scene,
    plan,
    images,
    options,
  );
  checkInteger(reference, 0, N - 1, 'reference');
  if (!(plan.sources[reference * MVS_SOURCE_SLOTS]! > 0))
    throw new RangeError('The reference has no source view');
  const seedFirst = plan.seedOffsets[reference]!,
    seedEnd = o.seeds ? plan.seedOffsets[reference + 1]! : seedFirst;
  mvsControl.reference = reference;
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
      // 4.2: the depth stage's (16 N + 4) bytes per pixel, plus one view's RGBA upload and its
      // distorted gray image.
      device.reserveWorkingBytes((16 * N + 4) * P + 8 * P);
      const mvsCameras = f32(plan.cameras.length, plan.cameras),
        mvsSources = u32(plan.sources.length, plan.sources),
        mvsSeeds = f32(
          Math.max(MVS_SEED_FLOATS, plan.seeds.length),
          plan.seeds.length ? plan.seeds : new Float32Array(MVS_SEED_FLOATS),
        ),
        mvsGray = f32(N * P),
        mvsDepth = f32(N * P),
        mvsGeoDepth = f32(N * P),
        mvsNormal = u32(N * P),
        mvsCost = f32(P),
        gray = f32(P);
      await undistortViews(device, packedImages, gray, mvsGray, width, height);
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
          run: () => mvsPropagate({ ...shared, mvsGeoDepth, mvsControl, mvsRange: empty }, 0),
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
                  mvsPropagate(
                    { ...shared, mvsGeoDepth, mvsControl: control, mvsRange },
                    Math.ceil(n / LANES),
                  ),
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

export interface MvsDepthMapsOptions extends Omit<MvsPhotometricOptions, 'snapshots' | 'readGray'> {
  /** Red-black iterations of the geometric pass (3.8). Default 4. */
  readonly geometricIterations?: number;
  /** Run `mvsFilter` after the geometric pass (3.9). Default true. */
  readonly filter?: boolean;
  /**
   * The packed indices of the references in the order their photometric passes run, each once.
   * Default 0..N − 1. The order changes no result (3.8).
   */
  readonly order?: readonly number[];
  /**
   * The packed indices that run the geometric pass and the filter, in this order, each at most
   * once. Default `order`. A view left out keeps depth 0 and its photometric normal.
   */
  readonly geometricOrder?: readonly number[];
  /** Read each reference's photometric arrays and its geometric arrays before the filter. */
  readonly readStages?: boolean;
}

export interface MvsDepthMap extends MvsDepthSnapshot {
  /** The packed index. */
  readonly reference: number;
  /** After the photometric pass, when `readStages` is set. */
  readonly photometric?: MvsDepthSnapshot;
  /** After the geometric pass and before the filter, when `readStages` is set. */
  readonly geometric?: MvsDepthSnapshot;
}

export interface MvsDepthMapsResult {
  readonly width: number;
  readonly height: number;
  /**
   * One map per packed view, by packed index: the filtered depth (0 for "no depth"), the
   * geometric pass's normal and the cost of its hypothesis with the geometric term.
   */
  readonly views: readonly MvsDepthMap[];
  readonly report: ShadePlanReport;
}

/** The thresholds of `mvsFilter`, COLMAP's filter defaults (section 3.9). */
const FILTER = { minNcc: 0.1, minConsistent: 2, maxError: 1, minAngleDeg: 3 } as const;
const GEOMETRIC_DEFAULTS = { geometricIterations: 4, filter: true } as const;

/**
 * The depth stage of sections 3.4 to 3.9 for every packed view: the undistortion of every view,
 * the photometric pass of every reference in `order`, then for every reference in `order` the
 * geometric pass and the filter. The geometric pass and the filter read the sources'
 * photometric depths in `mvsDepth` and write only the reference's slots of `mvsGeoDepth`,
 * `mvsNormal` and `mvsCost`, so the order of the references changes no result.
 */
export function mvsDepthMapsGpu(
  device: ShadeDevice,
  scene: MvsScene,
  plan: MvsViewPlan,
  images: ReadonlyMap<number, MvsImage>,
  options: MvsDepthMapsOptions = {},
): Promise<MvsDepthMapsResult> {
  const { o, N, P, width, height, packedImages, mvsControl } = prepareMvs(
    scene,
    plan,
    images,
    options,
  );
  const g = { ...GEOMETRIC_DEFAULTS, ...options };
  checkInteger(g.geometricIterations, 0, 64, 'geometricIterations');
  const order = options.order ?? Array.from({ length: N }, (_, v) => v);
  if (order.length !== N || new Set(order).size !== N || order.some((v) => !(v >= 0 && v < N)))
    throw new RangeError('order must hold every packed index once');
  const geometricOrder = options.geometricOrder ?? order;
  if (
    new Set(geometricOrder).size !== geometricOrder.length ||
    geometricOrder.some((v) => !(Number.isInteger(v) && v >= 0 && v < N))
  )
    throw new RangeError('geometricOrder must hold packed indices, each at most once');
  for (let v = 0; v < N; v++)
    if (!(plan.sources[v * MVS_SOURCE_SLOTS]! > 0))
      throw new RangeError(`Packed view ${v} has no source view`);
  const filterControl = {
    maxCost: 1 - FILTER.minNcc,
    minConsistent: FILTER.minConsistent,
    maxError: FILTER.maxError,
    minAngleCos: Math.cos((FILTER.minAngleDeg * Math.PI) / 180),
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
      // 4.2: the depth stage's (16 N + 4) bytes per pixel, plus one view's RGBA upload and its
      // distorted gray image.
      device.reserveWorkingBytes((16 * N + 4) * P + 8 * P);
      const mvsCameras = f32(plan.cameras.length, plan.cameras),
        mvsSources = u32(plan.sources.length, plan.sources),
        mvsSeeds = f32(
          Math.max(MVS_SEED_FLOATS, plan.seeds.length),
          plan.seeds.length ? plan.seeds : new Float32Array(MVS_SEED_FLOATS),
        ),
        mvsGray = f32(N * P),
        mvsDepth = f32(N * P),
        mvsGeoDepth = f32(N * P),
        mvsNormal = u32(N * P),
        mvsCost = f32(P),
        gray = f32(P);
      await undistortViews(device, packedImages, gray, mvsGray, width, height);
      const shared = { mvsCameras, mvsSources, mvsGray, mvsDepth, mvsGeoDepth, mvsNormal, mvsCost };
      // The filter binds no `mvsCost`.
      const { mvsCost: _cost, ...filterArrays } = shared;
      const photometric: MvsDepthSnapshot[] = [],
        geometric: MvsDepthSnapshot[] = [],
        costs: Float32Array[] = [];
      const read = async (depthArray: typeof mvsDepth, r: number, into: MvsDepthSnapshot[]) => {
        const depth = await device.read(depthArray, N * P),
          normal = await device.read(mvsNormal, N * P),
          cost = await device.read(mvsCost, P);
        into[r] = {
          depth: depth.slice(r * P, (r + 1) * P),
          normal: normal.slice(r * P, (r + 1) * P),
          cost: cost.slice(),
        };
      };
      const readCost = async (r: number) => {
        costs[r] = (await device.read(mvsCost, P)).slice();
      };
      const schedule = { device, shared, chunkPixels: o.chunkPixels };
      const steps = function* () {
        for (const r of order)
          yield* withAfter(
            photometricSteps(
              schedule,
              { ...mvsControl, reference: r },
              plan.sources[r * MVS_SOURCE_SLOTS]!,
              mvsSeeds,
              plan.seedOffsets[r]!,
              o.seeds ? plan.seedOffsets[r + 1]! : plan.seedOffsets[r]!,
              o.photometricIterations,
            ),
            options.readStages ? () => read(mvsDepth, r, photometric) : undefined,
          );
        for (const r of geometricOrder) {
          const control = { ...mvsControl, reference: r, geometric: 1 },
            K = plan.sources[r * MVS_SOURCE_SLOTS]!;
          const pass = withAfter(
            geometricSteps(schedule, control, K, o.photometricIterations, g.geometricIterations),
            options.readStages ? () => read(mvsGeoDepth, r, geometric) : undefined,
          );
          const filter = (mvsRange: MvsRange, groups: number) =>
            mvsFilter({ ...filterArrays, mvsControl: control, filterControl, mvsRange }, groups);
          // The cost after the filter, which leaves it as the geometric pass wrote it.
          if (g.filter) {
            yield* pass;
            yield* withAfter(chunkSteps(schedule, 'mvsFilter', 'mvsFilter', P, K, filter), () =>
              readCost(r),
            );
          } else yield* withAfter(pass, () => readCost(r));
        }
      };
      const report = await device.planSteps(steps());
      const depth = await device.read(mvsGeoDepth, N * P),
        normal = await device.read(mvsNormal, N * P);
      const views = Array.from({ length: N }, (_, r) => ({
        reference: r,
        depth: depth.slice(r * P, (r + 1) * P),
        normal: normal.slice(r * P, (r + 1) * P),
        cost: costs[r] ?? new Float32Array(P),
        ...(options.readStages ? { photometric: photometric[r]! } : {}),
        ...(options.readStages && geometric[r] ? { geometric: geometric[r] } : {}),
      }));
      successful = true;
      return { width, height, views, report };
    } finally {
      for (const b of owned) successful ? device.recycle(b) : device.drop(b);
    }
  });
}

type MvsControlRecord = ReturnType<typeof prepareMvs>['mvsControl'];
interface MvsRange {
  readonly first: number;
  readonly end: number;
}
interface MvsSharedArrays {
  readonly mvsCameras: Resident<Float32Array>;
  readonly mvsSources: Resident<Uint32Array>;
  readonly mvsGray: Resident<Float32Array>;
  readonly mvsDepth: Resident<Float32Array>;
  readonly mvsGeoDepth: Resident<Float32Array>;
  readonly mvsNormal: Resident<Uint32Array>;
  readonly mvsCost: Resident<Float32Array>;
}
interface MvsSchedule {
  readonly device: ShadeDevice;
  readonly shared: MvsSharedArrays;
  /** Caps a chunk at this many elements. */
  readonly chunkPixels: number | undefined;
}

type MvsStep = readonly [
  name: string,
  run: () => void,
  after?: () => Promise<void>,
  chunk?: GpuChunk,
];
/**
 * The steps with `after` joined to the after hook of the last one, so that it reads what the
 * last step wrote. A step of its own with no dispatch would be a plan of no work.
 */
function* withAfter(
  steps: Iterable<MvsStep>,
  after: (() => Promise<void>) | undefined,
): Generator<MvsStep> {
  let previous: MvsStep | undefined;
  for (const step of steps) {
    if (previous) yield previous;
    previous = step;
  }
  if (!previous) return;
  if (!after) return yield previous;
  const [name, run, before, chunk] = previous;
  yield [
    name,
    run,
    async () => {
      await before?.();
      await after();
    },
    chunk,
  ];
}

/**
 * The chunks of one pass over `total` elements under `key` (5.1), as plan steps, by the rule of
 * `mvsPhotometricGpu`: one unit is one element and one of the K sources, the device sizes each
 * chunk from the rate it measured, a chunk before the last fills its workgroups, and
 * `chunkPixels` caps the size. The warm step runs `dispatch` over zero workgroups. The CPU tier
 * runs the pass as one chunk.
 */
function* chunkSteps(
  schedule: MvsSchedule,
  key: MvsChunkKey,
  name: string,
  total: number,
  K: number,
  dispatch: (mvsRange: MvsRange, groups: number) => void,
) {
  const { device, chunkPixels } = schedule;
  const warm = { id: 'mvs', run: () => dispatch({ first: 0, end: 0 }, 0) };
  for (let first = 0; first < total;) {
    const remaining = total - first;
    let n = Math.min(
      remaining,
      Math.floor(device.chunkSize(key, remaining * K, Math.min(remaining, LANES) * K) / K),
    );
    if (n < remaining) n = Math.max(LANES, n - (n % LANES));
    n = Math.min(n, remaining, chunkPixels ?? Infinity);
    const mvsRange = { first, end: first + n };
    yield [
      name,
      () => dispatch(mvsRange, Math.ceil(n / LANES)),
      undefined,
      { key, units: n * K, warm },
    ] as const;
    first += n;
  }
}

/**
 * The photometric pass of one reference as plan steps (3.5 and 3.6): the initialization, the
 * seeds `seedFirst..seedEnd` and T red-black iterations, under the keys `mvsInitialize` and
 * `mvsPropagate`.
 */
function* photometricSteps(
  schedule: MvsSchedule,
  mvsControl: MvsControlRecord,
  K: number,
  mvsSeeds: Resident<Float32Array>,
  seedFirst: number,
  seedEnd: number,
  T: number,
) {
  const P = mvsControl.width * mvsControl.height;
  // The initialization and the seeds bind no `mvsGeoDepth`.
  const { mvsGeoDepth: _geometric, ...photometric } = schedule.shared;
  yield* chunkSteps(schedule, 'mvsInitialize', 'mvsInitialize', P, K, (mvsRange, groups) =>
    mvsInitialize({ ...photometric, mvsControl, mvsRange }, groups),
  );
  if (seedEnd > seedFirst)
    yield [
      'mvsSeed',
      () =>
        mvsSeed(
          { ...photometric, mvsSeeds, mvsControl, mvsRange: { first: seedFirst, end: seedEnd } },
          Math.ceil((seedEnd - seedFirst) / LANES),
        ),
    ] as const;
  yield* propagationSteps(schedule, 'mvsPropagate', mvsControl, K, 1, T, 0);
}

/**
 * The geometric pass of one reference as plan steps (3.8): `mvsGeometricInitialize`, then G
 * red-black iterations with the geometric term, under the keys `mvsGeometricInitialize` and
 * `mvsPropagateGeometric`. The iterations hash as T + 1..T + G, so their draws differ from the
 * photometric pass's, and their perturbation starts again at 2^−1.
 */
function* geometricSteps(
  schedule: MvsSchedule,
  mvsControl: MvsControlRecord,
  K: number,
  T: number,
  G: number,
) {
  const P = mvsControl.width * mvsControl.height;
  yield* chunkSteps(
    schedule,
    'mvsGeometricInitialize',
    'mvsGeometricInitialize',
    P,
    K,
    (mvsRange, groups) =>
      mvsGeometricInitialize({ ...schedule.shared, mvsControl, mvsRange }, groups),
  );
  yield* propagationSteps(schedule, 'mvsPropagateGeometric', mvsControl, K, T + 1, G, T);
}

/**
 * Iterations `first..first + count − 1` of red-black propagation under `key`. Iteration t
 * hashes as t and perturbs by 2^−(t − perturbOffset) (3.6).
 */
function* propagationSteps(
  schedule: MvsSchedule,
  key: MvsChunkKey,
  mvsControl: MvsControlRecord,
  K: number,
  first: number,
  count: number,
  perturbOffset: number,
) {
  const colorIndices = Math.ceil(mvsControl.width / 2) * mvsControl.height;
  for (let t = first; t < first + count; t++) {
    const scale = 2 ** -(t - perturbOffset);
    for (let color = 0; color < 2; color++) {
      const control = {
        ...mvsControl,
        color,
        iteration: t,
        perturbDepth: scale,
        perturbAngle: ((PERTURB_ANGLE_DEG * Math.PI) / 180) * scale,
      };
      yield* chunkSteps(schedule, key, 'mvsPropagate', colorIndices, K, (mvsRange, groups) =>
        mvsPropagate({ ...schedule.shared, mvsControl: control, mvsRange }, groups),
      );
    }
  }
}
