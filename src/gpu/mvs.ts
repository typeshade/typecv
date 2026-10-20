/**
 * The dense MVS (`docs/dense-mvs.md`, sections 3.4 to 3.11, 4.1 and 5.1, steps 2 to 5 of section
 * 8): `mvsUndistort` writes the undistorted gray image of every packed view, `mvsInitialize` and
 * `mvsSeed` give a reference its first hypotheses, and `mvsPropagate` runs the red-black
 * iterations of the photometric pass. `mvsPhotometricGpu` runs them on one reference.
 * `mvsDepthMapsGpu` runs them on every reference, then the geometric pass
 * (`mvsGeometricInitialize`, `mvsPropagate` with `geometric` 1) and `mvsFilter` on each.
 * `reconstructDenseGpu` plans the views, runs the depth stage and fuses the filtered depth maps
 * (`mvsFuseFlags`, the shared flag scan and `mvsScatterPoints`). The kernels are in
 * `mvs.shade.ts`. The host arrays come from `planMvsViews` (`mvs-views.ts`).
 *
 * `src/gpu/index.ts` exports `reconstructDenseGpu` and its types (5.4). The other functions are
 * internal, and the tests reach them through `internal.ts`.
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
 * - The depth stage allocates `mvsGray` and undistorts every view before it allocates the other
 *   arrays, and it recycles the distorted gray image first, so that its peak is the
 *   (16 N + 4) bytes per pixel of 4.2 and the small arrays (step 5).
 * - Fusion (3.10, step 5): "the point of j at q" is j's filtered depth at the nearest pixel of q,
 *   back-projected along the ray of that pixel's center, and its normal is j's normal there. The
 *   point and the normal of a fused point are the means over the reference and its consistent
 *   views, in the normalized world frame. A view is consistent by the thresholds of COLMAP's
 *   fusion (0.01, 2 px, 10°). "A lower index" is the packed index.
 * - `mvsScatterPoints` computes the fused point of its pixel again. The scan rewrites the flags in
 *   place, and a point kept per pixel would need a second P-sized point array (4.2). It runs as
 *   chunks under the key `mvsFuse`, as `mvsFuseFlags` does, since its work per unit is the same.
 * - Colors (3.11): `mvsRgba` holds the reference's packed RGBA as uploaded, and `mvsScatterPoints`
 *   samples it at the pixel's position in the distorted image, the map of `mvsUndistort`, each
 *   channel bilinear and rounded. So the fusion stage holds one RGBA array, `mvsRgba`, and no
 *   upload beside it.
 * - The depth previews of 5.3 follow the depth stage of every view, from one read of
 *   `mvsGeoDepth`, because a read takes the whole array. A preview pixel is the mean of the
 *   depths above 0 in its 4 × 4 block.
 */
import type { Resident } from 'typeshade/runtime';
import { radialLimitSquared } from '../geometry/camera.ts';
import { ShadeDevice, type GpuChunk, type ShadePlanReport } from './device.ts';
import { rgbaToGray } from './features.shade.ts';
import {
  MVS_CAMERA_FLOATS,
  MVS_SEED_FLOATS,
  MVS_SOURCE_SLOTS,
  denormalizeMvsPoints,
  planMvsViews,
} from './mvs-views.ts';
import type { MvsScene, MvsViewOptions, MvsViewPlan, MvsViewStatus } from './mvs-views.ts';
import {
  mvsFilter,
  mvsFuseFlags,
  mvsGeometricInitialize,
  mvsInitialize,
  mvsPropagate,
  mvsScatterPoints,
  mvsSeed,
  mvsUndistort,
} from './mvs.shade.ts';
import { planFlagScan, scanBlockCount } from './scan.ts';
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
  | 'mvsFilter'
  | 'mvsFuse';

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

/** The arrays a job allocates, recycled after a success and dropped after a failure. */
function ownArrays(device: ShadeDevice) {
  const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
  return {
    f32(n: number, values?: Float32Array) {
      const r = device.allocate(n, values);
      owned.push(r);
      return r;
    },
    u32(n: number, values?: Uint32Array) {
      const r = device.allocateU32(n, values);
      owned.push(r);
      return r;
    },
    release(successful: boolean) {
      for (const b of owned) successful ? device.recycle(b) : device.drop(b);
    },
  };
}
type MvsOwner = ReturnType<typeof ownArrays>;

interface MvsDepthArrays extends MvsSharedArrays {
  readonly mvsSeeds: Resident<Float32Array>;
}

/**
 * The arrays of the depth stage (4.1), with `mvsGray` written by the undistortion of every packed
 * view (3.4). The undistortion runs before `mvsDepth`, `mvsGeoDepth`, `mvsNormal` and `mvsCost`
 * exist, and it recycles its distorted gray image, so the stage's peak is the
 * (16 N + 4) bytes per pixel of 4.2 and the small arrays.
 */
async function allocateDepthStage(
  device: ShadeDevice,
  own: MvsOwner,
  plan: MvsViewPlan,
  packedImages: readonly MvsPacked[],
  width: number,
  height: number,
): Promise<MvsDepthArrays> {
  const N = plan.packed.length,
    P = width * height;
  device.reserveWorkingBytes((16 * N + 4) * P);
  const mvsCameras = own.f32(plan.cameras.length, plan.cameras),
    mvsSources = own.u32(plan.sources.length, plan.sources),
    mvsSeeds = own.f32(
      Math.max(MVS_SEED_FLOATS, plan.seeds.length),
      plan.seeds.length ? plan.seeds : new Float32Array(MVS_SEED_FLOATS),
    ),
    mvsGray = own.f32(N * P),
    gray = own.f32(P);
  await undistortViews(device, packedImages, gray, mvsGray, width, height);
  device.recycle(gray);
  return {
    mvsCameras,
    mvsSources,
    mvsSeeds,
    mvsGray,
    mvsDepth: own.f32(N * P),
    mvsGeoDepth: own.f32(N * P),
    mvsNormal: own.u32(N * P),
    mvsCost: own.f32(P),
  };
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
  return device.enqueue(async () => {
    const own = ownArrays(device);
    let successful = false;
    try {
      const { mvsSeeds, ...shared } = await allocateDepthStage(
        device,
        own,
        plan,
        packedImages,
        width,
        height,
      );
      const snapshots: MvsDepthSnapshot[] = [];
      const snapshot = async () => {
        const depth = await device.read(shared.mvsDepth, N * P),
          normal = await device.read(shared.mvsNormal, N * P),
          cost = await device.read(shared.mvsCost, P);
        snapshots.push({
          depth: depth.slice(reference * P, (reference + 1) * P),
          normal: normal.slice(reference * P, (reference + 1) * P),
          cost: cost.slice(),
        });
      };
      // 5.1 and step 3: the chunks of `photometricSteps`, with a snapshot after the seeds and
      // after each iteration when `snapshots` is set.
      const report = await device.planSteps(
        photometricSteps(
          { device, shared, chunkPixels: o.chunkPixels },
          { ...mvsControl, reference },
          plan.sources[reference * MVS_SOURCE_SLOTS]!,
          mvsSeeds,
          seedFirst,
          seedEnd,
          o.photometricIterations,
          o.snapshots ? snapshot : undefined,
        ),
      );
      const depth = await device.read(shared.mvsDepth, N * P),
        normal = await device.read(shared.mvsNormal, N * P),
        cost = await device.read(shared.mvsCost, P),
        grayImages = o.readGray ? (await device.read(shared.mvsGray, N * P)).slice() : undefined;
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
      own.release(successful);
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
    const own = ownArrays(device);
    let successful = false;
    try {
      const { mvsSeeds, ...shared } = await allocateDepthStage(
        device,
        own,
        plan,
        packedImages,
        width,
        height,
      );
      const { mvsDepth, mvsGeoDepth, mvsNormal, mvsCost } = shared;
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
      own.release(successful);
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
 * `chunkPixels` caps the size. The warm step runs `dispatch` over zero workgroups, once for each
 * `warmId` of a key. The CPU tier runs the pass as one chunk.
 */
function* chunkSteps(
  schedule: MvsSchedule,
  key: MvsChunkKey,
  name: string,
  total: number,
  K: number,
  dispatch: (mvsRange: MvsRange, groups: number) => void,
  warmId = 'mvs',
) {
  const { device, chunkPixels } = schedule;
  const warm = { id: warmId, run: () => dispatch({ first: 0, end: 0 }, 0) };
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
 * `mvsPropagate`. `after(t)` runs after the seeds (t = 0) and after iteration t.
 */
function* photometricSteps(
  schedule: MvsSchedule,
  mvsControl: MvsControlRecord,
  K: number,
  mvsSeeds: Resident<Float32Array>,
  seedFirst: number,
  seedEnd: number,
  T: number,
  after?: (t: number) => Promise<void>,
) {
  const P = mvsControl.width * mvsControl.height;
  // The initialization and the seeds bind no `mvsGeoDepth`.
  const { mvsGeoDepth: _geometric, ...photometric } = schedule.shared;
  function* start() {
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
  }
  yield* withAfter(start(), after && (() => after(0)));
  yield* propagationSteps(schedule, 'mvsPropagate', mvsControl, K, 1, T, 0, after);
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
 * hashes as t and perturbs by 2^−(t − perturbOffset) (3.6). `after(t)` runs after iteration t.
 */
function* propagationSteps(
  schedule: MvsSchedule,
  key: MvsChunkKey,
  mvsControl: MvsControlRecord,
  K: number,
  first: number,
  count: number,
  perturbOffset: number,
  after?: (t: number) => Promise<void>,
) {
  const colorIndices = Math.ceil(mvsControl.width / 2) * mvsControl.height;
  for (let t = first; t < first + count; t++) {
    const scale = 2 ** -(t - perturbOffset);
    function* iteration() {
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
    yield* withAfter(iteration(), after && (() => after(t)));
  }
}

/** The fusion thresholds, COLMAP's `stereo_fusion` defaults (section 3.10). */
const FUSION = { maxDepthError: 0.01, maxReprojection: 2, maxNormalDeg: 10 } as const;
const DENSE_DEFAULTS = { minConsistentViews: 3 } as const;
/** A depth preview has a quarter of the image's size in each axis (5.3). */
const PREVIEW_STEP = 4;

/** Points of the dense cloud, in the model's frame. */
export interface GpuDensePoints {
  readonly count: number;
  /** x, y, z per point. */
  readonly positions: Float32Array;
  /** r, g, b per point, the reference pixel's color after undistortion (3.11). */
  readonly colors: Uint8Array;
  /**
   * The unit world normal per point as two snorm16 values of an octahedral map (4.1): the low
   * and the high half of `pack2x16snorm`.
   */
  readonly normals: Int16Array;
}

/** What `onView` receives (5.3). Its arrays are its own: a worker can transfer them. */
export type GpuDenseViewEvent =
  | {
      readonly type: 'depth';
      readonly viewId: number;
      /** The preview's size, ⌈W / 4⌉ × ⌈H / 4⌉. */
      readonly width: number;
      readonly height: number;
      /**
       * The filtered depth at a quarter of the size in each axis, in the model's units: the mean
       * of the depths of each 4 × 4 block, 0 for a block without depth.
       */
      readonly depth: Float32Array;
    }
  | ({ readonly type: 'points'; readonly viewId: number } & GpuDensePoints);

export interface GpuDenseOptions extends MvsViewOptions {
  /** The least source costs averaged per hypothesis (3.7). Default 3. */
  readonly bestSources?: number;
  /** Window radius in pixels (3.7). Default 5. */
  readonly windowRadius?: number;
  /** Window sample step in pixels (3.7). Default 2. */
  readonly windowStep?: number;
  /** Red-black iterations of the photometric pass (3.6). Default 8. */
  readonly photometricIterations?: number;
  /** Red-black iterations of the geometric pass (3.8). Default 4. */
  readonly geometricIterations?: number;
  /** COLMAP's bilateral weights (3.7). Default false. */
  readonly bilateral?: boolean;
  /** The seed of the counter-based random numbers (3.5). Default 0. */
  readonly seed?: number;
  /** The least number of consistent views of a fused point, the reference included (3.10). Default 3. */
  readonly minConsistentViews?: number;
  /**
   * The depth previews, after the depth stage of every view, and the points of each reference
   * after its fusion (5.3). The run waits for a returned promise.
   */
  readonly onView?: (event: GpuDenseViewEvent) => void | Promise<void>;
}

/** `done`: the view has a depth map and was fused. The other two are the reasons it was not. */
export type GpuDenseViewStatus = 'done' | Exclude<MvsViewStatus, 'ready'>;

export interface GpuDenseView {
  readonly id: number;
  readonly status: GpuDenseViewStatus;
  /** The ids of the source views, best first (3.2). */
  readonly sources: readonly number[];
  /** The pixels with a filtered depth. */
  readonly depthPixels: number;
  /** The points this view emitted as the reference: `points.positions[3 · firstPoint ..]`. */
  readonly firstPoint: number;
  readonly points: number;
}

export interface GpuDenseResult {
  /** The points of every reference, in the order of the packed views. */
  readonly points: GpuDensePoints;
  /** One entry per input view, in input order. */
  readonly views: readonly GpuDenseView[];
  readonly report: ShadePlanReport;
}

/** @internal The options of `reconstructDense`, the tests' entry. */
export interface MvsDenseOptions extends GpuDenseOptions {
  /** Force chunks of at most this many pixels or color indices. Default: one chunk per pass. */
  readonly chunkPixels?: number;
}

/**
 * The dense MVS of sections 3.1 to 3.11 after the sparse SfM: the view selection, the depth stage
 * of every view (the photometric pass, the geometric pass and the filter) and the fusion of the
 * filtered depth maps into a colored point cloud with normals. `scene` is the final sparse model
 * (`pose` world to camera, `camera` with k1 and k2). A view without `width` and `height` takes
 * those of its image. `images` maps a view id to its RGBA at the analysis size, one size for
 * every view with a depth map. The points are in the model's frame.
 */
export function reconstructDenseGpu(
  device: ShadeDevice,
  scene: MvsScene,
  images: ReadonlyMap<number, MvsImage>,
  options: GpuDenseOptions = {},
): Promise<GpuDenseResult> {
  return reconstructDense(device, scene, images, options);
}

/** @internal `reconstructDenseGpu` with the tests' options. */
export async function reconstructDense(
  device: ShadeDevice,
  scene: MvsScene,
  images: ReadonlyMap<number, MvsImage>,
  options: MvsDenseOptions = {},
): Promise<GpuDenseResult> {
  const d = { ...DENSE_DEFAULTS, ...GEOMETRIC_DEFAULTS, ...options };
  checkInteger(d.minConsistentViews, 1, 64, 'minConsistentViews');
  checkInteger(d.geometricIterations, 0, 64, 'geometricIterations');
  const sized: MvsScene = {
    ...scene,
    views: scene.views.map((view) => {
      const image = images.get(view.id);
      return view.width === undefined && view.height === undefined && image
        ? { ...view, width: image.width, height: image.height }
        : view;
    }),
  };
  const plan = planMvsViews(sized, options);
  const N = plan.packed.length;
  const viewIds = plan.packed.map((r) => sized.views[r]!.id);
  const result = (
    pointsByView: readonly GpuDensePoints[],
    depthPixels: readonly number[],
    report: ShadePlanReport,
  ): GpuDenseResult => {
    const firstPoints: number[] = [];
    let total = 0;
    for (let v = 0; v < N; v++) {
      firstPoints.push(total);
      total += pointsByView[v]!.count;
    }
    const points = {
      count: total,
      positions: new Float32Array(3 * total),
      colors: new Uint8Array(3 * total),
      normals: new Int16Array(2 * total),
    };
    pointsByView.forEach((part, v) => {
      points.positions.set(part.positions, 3 * firstPoints[v]!);
      points.colors.set(part.colors, 3 * firstPoints[v]!);
      points.normals.set(part.normals, 2 * firstPoints[v]!);
    });
    const views = plan.views.map((entry) => ({
      id: entry.id,
      status: entry.status === 'ready' ? ('done' as const) : entry.status,
      sources: entry.sources.map((v) => viewIds[v]!),
      depthPixels: entry.index >= 0 ? depthPixels[entry.index]! : 0,
      firstPoint: entry.index >= 0 ? firstPoints[entry.index]! : total,
      points: entry.index >= 0 ? pointsByView[entry.index]!.count : 0,
    }));
    return { points, views, report };
  };
  if (N === 0)
    return result([], [], { tier: device.backend, dispatches: 0, submissions: 0, groups: [] });
  const { o, P, width, height, packedImages, mvsControl } = prepareMvs(
    sized,
    plan,
    images,
    options,
  );
  const { scale } = plan.normalization;
  const filterControl = {
    maxCost: 1 - FILTER.minNcc,
    minConsistent: FILTER.minConsistent,
    maxError: FILTER.maxError,
    minAngleCos: Math.cos((FILTER.minAngleDeg * Math.PI) / 180),
  };
  return device.enqueue(async () => {
    const own = ownArrays(device);
    let successful = false;
    try {
      const { mvsSeeds, ...shared } = await allocateDepthStage(
        device,
        own,
        plan,
        packedImages,
        width,
        height,
      );
      const schedule = { device, shared, chunkPixels: options.chunkPixels };
      const { mvsCost: _cost, ...filterArrays } = shared;
      const depthPixels: number[] = [];
      // 5.3: the depth previews of every view, from one read of the filtered depths.
      const afterDepth = async () => {
        const depth = await device.read(shared.mvsGeoDepth, N * P);
        for (let v = 0; v < N; v++) {
          const map = depth.subarray(v * P, (v + 1) * P);
          depthPixels[v] = map.reduce((n, z) => n + (z > 0 ? 1 : 0), 0);
          if (options.onView)
            await options.onView({ type: 'depth', viewId: viewIds[v]!, ...depthPreview(map) });
        }
      };
      const depthPreview = (map: Float32Array) => {
        const pw = Math.ceil(width / PREVIEW_STEP),
          ph = Math.ceil(height / PREVIEW_STEP),
          preview = new Float32Array(pw * ph);
        for (let by = 0; by < ph; by++)
          for (let bx = 0; bx < pw; bx++) {
            let sum = 0,
              n = 0;
            for (let y = by * PREVIEW_STEP; y < Math.min(height, (by + 1) * PREVIEW_STEP); y++)
              for (let x = bx * PREVIEW_STEP; x < Math.min(width, (bx + 1) * PREVIEW_STEP); x++) {
                const z = map[y * width + x]!;
                if (z > 0) {
                  sum += z;
                  n++;
                }
              }
            preview[by * pw + bx] = n ? (sum / n) * scale : 0;
          }
        return { width: pw, height: ph, depth: preview };
      };
      const pointsByView: GpuDensePoints[] = [];
      function* depthStage() {
        for (let r = 0; r < N; r++)
          yield* photometricSteps(
            schedule,
            { ...mvsControl, reference: r },
            plan.sources[r * MVS_SOURCE_SLOTS]!,
            mvsSeeds,
            plan.seedOffsets[r]!,
            plan.seedOffsets[r + 1]!,
            o.photometricIterations,
          );
        for (let r = 0; r < N; r++) {
          const control = { ...mvsControl, reference: r, geometric: 1 },
            K = plan.sources[r * MVS_SOURCE_SLOTS]!;
          yield* geometricSteps(
            schedule,
            control,
            K,
            o.photometricIterations,
            d.geometricIterations,
          );
          yield* chunkSteps(schedule, 'mvsFilter', 'mvsFilter', P, K, (mvsRange, groups) =>
            mvsFilter({ ...filterArrays, mvsControl: control, filterControl, mvsRange }, groups),
          );
        }
      }
      function* steps() {
        yield* withAfter(depthStage(), afterDepth);
        // 4.2 and 4.3: the fusion stage keeps `mvsGeoDepth`, `mvsNormal` and `mvsCameras`. The
        // hook above has read the last filter's output, so no queued call reads these arrays.
        for (const b of [
          shared.mvsGray,
          shared.mvsDepth,
          shared.mvsCost,
          mvsSeeds,
          shared.mvsSources,
        ])
          device.recycle(b);
        device.reserveWorkingBytes(28 * P + 8 * scanBlockCount(P) + 8);
        const mvsRgba = own.u32(P),
          mvsFlags = own.u32(P),
          mvsFlagOffsets = own.u32(2 * scanBlockCount(P)),
          totals = own.u32(2),
          mvsPoints = own.f32(3 * P),
          mvsPointData = own.u32(2 * P);
        const fused = {
          mvsCameras: shared.mvsCameras,
          mvsGeoDepth: shared.mvsGeoDepth,
          mvsNormal: shared.mvsNormal,
          mvsFlags,
        };
        for (let r = 0; r < N; r++) {
          // 3.11: the reference's RGBA as uploaded. The previous reference's read has finished,
          // so the write follows every call that read the array.
          mvsRgba.write(packedImages[r]!.rgba);
          const view = packedImages[r]!;
          const control = { ...mvsControl, reference: r };
          const fuseControl = {
            views: N,
            minViews: d.minConsistentViews,
            maxDepthError: FUSION.maxDepthError,
            maxReprojection: FUSION.maxReprojection,
            minNormalCos: Math.cos((FUSION.maxNormalDeg * Math.PI) / 180),
            distorted: view.k1 !== 0 || view.k2 !== 0 ? 1 : 0,
            k1: view.k1,
            k2: view.k2,
            radialLimit: Math.min(1e30, radialLimitSquared(view.k1, view.k2)),
          };
          // 5.1: one unit is one pixel and one other view, for both kernels.
          yield* chunkSteps(
            schedule,
            'mvsFuse',
            'mvsFuseFlags',
            P,
            N - 1,
            (mvsRange, groups) =>
              mvsFuseFlags({ ...fused, mvsControl: control, fuseControl, mvsRange }, groups),
            'mvsFuseFlags',
          );
          yield [
            'mvsFlagScan',
            () => planFlagScan(device, { flags: mvsFlags, blockSums: mvsFlagOffsets, totals }, P),
          ] as const;
          const scatter = chunkSteps(
            schedule,
            'mvsFuse',
            'mvsScatterPoints',
            P,
            N - 1,
            (mvsRange, groups) =>
              mvsScatterPoints(
                {
                  ...fused,
                  mvsRgba,
                  mvsFlagOffsets,
                  mvsPoints,
                  mvsPointData,
                  mvsControl: control,
                  fuseControl,
                  mvsRange,
                },
                groups,
              ),
            'mvsScatterPoints',
          );
          yield* withAfter(scatter, async () => {
            const count = (await device.read(totals, 2))[0]!;
            if (count > P) throw new Error('The dense fusion returned an invalid count');
            const positions = await device.read(mvsPoints, 3 * count),
              data = await device.read(mvsPointData, 2 * count);
            const part = {
              count,
              positions: Float32Array.from(
                denormalizeMvsPoints(positions.subarray(0, 3 * count), plan.normalization),
              ),
              colors: new Uint8Array(3 * count),
              normals: new Int16Array(2 * count),
            };
            for (let k = 0; k < count; k++) {
              const normal = data[2 * k]!,
                color = data[2 * k + 1]!;
              part.normals[2 * k] = (normal << 16) >> 16;
              part.normals[2 * k + 1] = normal >> 16;
              for (let c = 0; c < 3; c++) part.colors[3 * k + c] = (color >>> (8 * c)) & 255;
            }
            pointsByView[r] = part;
            if (options.onView)
              await options.onView({
                type: 'points',
                viewId: viewIds[r]!,
                count,
                positions: part.positions.slice(),
                colors: part.colors.slice(),
                normals: part.normals.slice(),
              });
          });
        }
      }
      const report = await device.planSteps(steps());
      successful = true;
      return result(pointsByView, depthPixels, report);
    } finally {
      own.release(successful);
    }
  });
}
