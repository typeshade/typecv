/**
 * The dense MVS (`docs/dense-mvs.md`, sections 3.4 to 3.11, 4.1 and 5.1, steps 2 to 5 of section
 * 8, and Amendment 3): `mvsUndistort` writes the undistorted gray image of a view to its slot,
 * `mvsInitialize` and `mvsSeed` give a reference its first hypotheses, and `mvsPropagate` runs the
 * red-black iterations of the photometric pass. `mvsPhotometricGpu` runs them on one reference.
 * `mvsDepthMapsGpu` runs them on every reference, then the geometric pass
 * (`mvsGeometricInitialize`, `mvsPropagate` with `geometric` 1) and `mvsFilter` on each.
 * `reconstructDenseGpu` plans the views, runs the depth stage and fuses the filtered depth maps
 * (`mvsFuseFlags`, the shared flag scan and `mvsScatterPoints`). The kernels are in
 * `mvs.shade.ts`. The host arrays come from `planMvsViews` (`mvs-views.ts`), and the residency
 * from `planMvsResidency` (`mvs-residency.ts`).
 *
 * `src/gpu/index.ts` exports `reconstructDenseGpu` and its types (5.4). The other functions are
 * internal, and the tests reach them through `internal.ts`.
 *
 * Residency (Amendment 3): the device holds a window, never every view. A depth pass keeps the
 * reference's own arrays (`mvsGeoDepth`, `mvsNormal`, `mvsCost`, and in the photometric pass
 * `mvsDepth`), P values each, and view slots of `mvsGray` (and of `mvsDepth` in the geometric
 * pass) for the reference and its sources. The slots are a cache: a view already in a slot is
 * not uploaded again. A finished output goes back to the host: the photometric depth and normal
 * after the photometric pass, the filtered depth and the geometric normal after the filter. A
 * pass whose window does not fit the budget streams its sources through in groups
 * (`mvsStreamCosts`, `mvsStreamSelect`), and a fusion whose window does not fit streams its
 * fusion views through in groups with per-pixel sums (`mvsFuseAccumulate`, `mvsFuseEmit`,
 * `mvsFuseScatterSums`). Both give the window mode's results bit for bit on the CPU tier.
 *
 * Details that the document leaves open, with the choice made here:
 * - All packed views have one size W × H, so that a slot is s × P + i (4.1). A plan with two
 *   sizes is refused.
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
 *   angle). The hash input is (seed, packed index of the reference, pixel index, 16 t + draw).
 * - Chunks (5.1, step 3): every pass runs as chunks under its key, one unit an element and a
 *   source, each with its warm step. The device sizes them from the time budget. `chunkPixels`
 *   caps a chunk at that many elements. The CPU tier runs each pass as one chunk.
 * - The geometric pass (3.8, step 4) keeps the reference's hypotheses in `mvsGeoDepth` and
 *   `mvsNormal`, and reads the sources' photometric depths in the slots of `mvsDepth`. It never
 *   reads another view's geometric output, so the order of the references changes no result.
 *   `mvsGeometricInitialize` copies the photometric depth and computes the cost with the
 *   geometric term. The iterations hash as T + 1..T + G and perturb from 2^−1 again.
 * - ψ_s interpolates the source's inverse depth on the fractional ray of q when all four
 *   samples are positive and span at most 2%. At discontinuities and borders it uses nearest
 *   depth. A q outside the source, a missing nearest depth or a point behind a camera gives ψmax.
 * - `mvsFilter` recomputes the photometric source costs of the final hypothesis (the stored
 *   cost holds the geometric term), and its ψ_s reads the sources' photometric depths, as the
 *   geometric pass does. It writes depth 0 over a rejected pixel of `mvsGeoDepth`.
 * - Fusion (3.10, step 5): "the point of j at q" is j's filtered depth at the nearest pixel of q,
 *   back-projected along the ray of that pixel's center, and its normal is j's normal there. The
 *   point and the normal of a fused point are the means over the reference and its consistent
 *   fusion views, in the normalized world frame, summed in ascending packed index. A view is
 *   consistent by the thresholds of COLMAP's fusion (0.01, 2 px, 10°). "A lower index" is the
 *   packed index.
 * - `mvsScatterPoints` computes the fused point of its pixel again. The scan rewrites the flags in
 *   place, and a point kept per pixel would need a second P-sized point array (4.2). It runs as
 *   chunks under the key `mvsFuse`, as `mvsFuseFlags` does, since its work per unit is the same.
 * - Colors (3.11): `mvsRgba` holds the reference's packed RGBA as uploaded, and the scatter
 *   samples it at the pixel's position in the distorted image, the map of `mvsUndistort`, each
 *   channel bilinear and rounded.
 * - The depth previews of 5.3 follow the depth stage of every view, from the host's filtered
 *   depths. A preview pixel is the mean of the depths above 0 in its 4 × 4 block.
 * - Progress and cancel (5.3, step 6): `onProgress` runs after each reference's photometric pass,
 *   its geometric pass and filter, and its fusion. `signal` is read before each plan, so a chunk
 *   in flight completes. A cancelled run recycles its arrays, as a successful one does, and
 *   rejects with the signal's reason.
 */
import type { Resident } from 'typeshade/runtime';
import { readTogether } from './readback.ts';
import { radialLimitSquared } from '../geometry/camera.ts';
import {
  snapshotObservationPaths,
  type ObservationPathMask,
} from '../reconstruction/observation-paths.ts';
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
  MVS_PARTIAL_HYPOTHESES,
  mvsDepthArrays,
  mvsFusionArrays,
  planMvsResidency,
  type MvsDepthResidency,
  type MvsFusionResidency,
  type MvsResidency,
} from './mvs-residency.ts';
import {
  mvsApplyObservationMask,
  mvsCopyF32,
  mvsCopyU32,
  mvsFilter,
  mvsFilterOracle,
  mvsFuseAccumulate,
  mvsFuseEmit,
  mvsFuseFlags,
  mvsFuseScatterSums,
  mvsGeometricInitialize,
  mvsGeometricInitializeOracle,
  mvsInitialize,
  mvsInitializeOracle,
  mvsPropagate,
  mvsPropagateOracle,
  mvsScatterPoints,
  mvsSeed,
  mvsSeedOracle,
  mvsStreamCosts,
  mvsStreamCostsOracle,
  mvsStreamSelect,
  mvsUndistort,
} from './mvs.shade.ts';
import { planFlagScan, scanBlockCount } from './scan.ts';
import { packRgba } from './sift.ts';
import { readOnlyRgbaWords } from './rgba-read-view.ts';
import { fuseTsdfGpu, type TsdfProgress, type TsdfReport } from './tsdf.ts';
import {
  completeOwnedPlanarDepths,
  type OwnedPlanarInput,
  type PlanarDepthReport,
} from './planar-depth.ts';
import {
  checkedTsdfOptions,
  TSDF_POINT_WORDS,
  type TsdfOptions,
  type TsdfPoints,
} from '../reconstruction/tsdf.ts';
import {
  checkedLabelOptions,
  labelDenseCloud,
  type DenseLabelOptions,
  type DenseLabelReport,
  type LabelLandmark,
} from '../reconstruction/dense-label.ts';

export type { MvsResidency } from './mvs-residency.ts';
export type { TsdfReport } from './tsdf.ts';
export type { TsdfOptions } from '../reconstruction/tsdf.ts';
export type { DenseLabelOptions, DenseLabelReport } from '../reconstruction/dense-label.ts';

/** The RGBA bytes of one view at its analysis size. */
export interface MvsImage {
  readonly rgba: Uint8Array | Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
  /** Explicit optical paths on the undistorted analysis lattice. Only direct/static pixels
   * contribute to ordinary surface geometry. RGB is preserved; this does not detect reflections.
   * Use cameraMaskFromPixelPaths with the same mask for upstream camera/static-map selection.
   */
  readonly observationMask?: ObservationPathMask;
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
  /** Centered, odd patch lattice with edge-aware intensity weights (sigma 0.1). Default false. */
  readonly boundaryAware?: boolean;
  /**
   * Minimum NCC-cost improvement over a valid rotation-only warp for each supporting source.
   * Depth without two non-ambiguous supports is withheld. Default 0 disables this test;
   * this is an observability check, not semantic sky detection.
   */
  readonly minFiniteGain?: number;
  /** The seed of the counter-based random numbers (3.5). Default 0. */
  readonly seed?: number;
  /**
   * The least confidence of a kept depth (section 10.3): the product of the photometric term, the
   * fraction of consistent sources and the view-normal term. A depth below it becomes "no depth"
   * in the filter. Default 0, which keeps the rules of 3.9 alone.
   */
  readonly minConfidence?: number;
  /** Write the plan's seeds over the random initialization (3.5). Default true. */
  readonly seeds?: boolean;
  /** Force chunks of at most this many pixels or color indices. Default: one chunk per pass. */
  readonly chunkPixels?: number;
  /**
   * The most bytes the run keeps on the device (Amendment 3). Default: the device's
   * `workingBytes` less its active bytes when the run starts. A smaller value forces the
   * streamed paths.
   */
  readonly residentBytes?: number;
  /**
   * @internal Force streamed mode with at most this many sources or fusion views per group, a
   * test of the grouped paths (`planMvsResidency`).
   */
  readonly streamGroup?: number;
  /** Read the reference's arrays after the initialization and after each iteration. */
  readonly snapshots?: boolean;
  /** Read the undistorted gray images of all packed views. */
  readonly readGray?: boolean;
  /** @internal Compare the WebGPU workgroup pose cache with the scalar calculation. */
  readonly relativePoseCache?: boolean;
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
  /** The undistorted gray images, N × P values, when `readGray` is set. */
  readonly gray?: Float32Array;
  readonly residency: MvsResidency;
  readonly report: ShadePlanReport;
}

const DEFAULTS = {
  bestSources: 3,
  windowRadius: 5,
  windowStep: 2,
  photometricIterations: 8,
  bilateral: false,
  boundaryAware: false,
  minFiniteGain: 0,
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
/** The chunk keys of the kernels this module runs (section 5.1 and Amendment 3). */
export type MvsChunkKey =
  | 'mvsInitialize'
  | 'mvsPropagate'
  | 'mvsGeometricInitialize'
  | 'mvsPropagateGeometric'
  | 'mvsFilter'
  | 'mvsFuse'
  | 'mvsStreamCosts'
  | 'mvsStreamCostsGeometric'
  | 'mvsStreamSelect'
  | 'mvsFuseScatter';

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
  readonly image: MvsImage;
  readonly k1: number;
  readonly k2: number;
  readonly camera: MvsScene['views'][number]['camera'];
}

/** The checked options, the image size and the views' images, and the base control. */
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
  if (o.minConfidence !== undefined && !(o.minConfidence >= 0 && o.minConfidence <= 1))
    throw new RangeError('minConfidence must be in [0, 1]');
  if (o.relativePoseCache !== undefined && typeof o.relativePoseCache !== 'boolean')
    throw new RangeError('relativePoseCache must be a boolean');
  if (typeof o.boundaryAware !== 'boolean') throw new RangeError('boundaryAware must be a boolean');
  if (!(Number.isFinite(o.minFiniteGain) && o.minFiniteGain >= 0 && o.minFiniteGain <= 2))
    throw new RangeError('minFiniteGain must be in [0, 2]');
  if (o.chunkPixels !== undefined) checkInteger(o.chunkPixels, 1, 2 ** 31, 'chunkPixels');
  if (o.residentBytes !== undefined)
    checkInteger(o.residentBytes, 0, Number.MAX_SAFE_INTEGER, 'residentBytes');
  const axisSamples = o.boundaryAware
    ? 2 * Math.floor(o.windowRadius / o.windowStep) + 1
    : Math.floor((2 * o.windowRadius) / o.windowStep) + 1;
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
    const observationMask = image.observationMask
      ? snapshotObservationPaths(image.observationMask)
      : undefined;
    if (observationMask && (observationMask.width !== width || observationMask.height !== height))
      throw new RangeError('Observation-path mask must match the analysis lattice');
    return {
      image: observationMask ? { ...image, observationMask } : image,
      k1: view.camera.k1 ?? 0,
      k2: view.camera.k2 ?? 0,
      camera: view.camera,
    };
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
    bilateral: o.bilateral || o.boundaryAware ? 1 : 0,
    boundaryAware: o.boundaryAware ? 1 : 0,
    minFiniteGain: o.minFiniteGain,
    spatialFalloff: 1 / (2 * o.windowRadius * o.windowRadius),
    colorFalloff: 1 / (2 * (o.boundaryAware ? 0.1 : SIGMA_COLOR) ** 2),
    minAngleCos: Math.cos((MIN_ANGLE_DEG * Math.PI) / 180),
    perturbDepth: 1,
    perturbAngle: 0,
    geometric: 0,
    geometricWeight: GEOMETRIC_WEIGHT,
    geometricMaxCost: GEOMETRIC_MAX_COST,
    view: 0,
    state: 0,
    relativePoseCache: 0,
  };
  return { o, N, P, width, height, packedImages, mvsControl };
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
      owned.length = 0;
    },
  };
}
type MvsOwner = ReturnType<typeof ownArrays>;

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
 * The chunks of one pass over the elements `start..start + total` under `key` (5.1), as plan
 * steps: one unit is one element and one of the K sources, the device sizes each chunk from the
 * rate it measured, a chunk before the last fills its workgroups, and `chunkPixels` caps the
 * size. The warm step runs `dispatch` over zero workgroups, once for each `warmId` of a key. The
 * CPU tier runs the pass as one chunk.
 */
function* chunkSteps(
  schedule: Pick<MvsSchedule, 'device' | 'chunkPixels'>,
  key: MvsChunkKey,
  name: string,
  total: number,
  K: number,
  dispatch: (mvsRange: MvsRange, groups: number) => void,
  warmId = 'mvs',
  start = 0,
) {
  const { device, chunkPixels } = schedule;
  const units = Math.max(1, K);
  const warm = { id: warmId, run: () => dispatch({ first: start, end: start }, 0) };
  for (let first = 0; first < total;) {
    const remaining = total - first;
    let n = Math.min(
      remaining,
      Math.floor(
        device.chunkSize(key, remaining * units, Math.min(remaining, LANES) * units) / units,
      ),
    );
    if (n < remaining) n = Math.max(LANES, n - (n % LANES));
    n = Math.min(n, remaining, chunkPixels ?? Infinity);
    const mvsRange = { first: start + first, end: start + first + n };
    yield [
      name,
      () => dispatch(mvsRange, Math.ceil(n / LANES)),
      undefined,
      { key, units: n * units, warm },
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
      (schedule.device.backend === 'cpu' ? mvsInitializeOracle : mvsInitialize)(
        { ...photometric, mvsControl, mvsRange },
        groups,
      ),
    );
    if (seedEnd > seedFirst)
      yield [
        'mvsSeed',
        () =>
          (schedule.device.backend === 'cpu' ? mvsSeedOracle : mvsSeed)(
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
      (schedule.device.backend === 'cpu' ? mvsGeometricInitializeOracle : mvsGeometricInitialize)(
        { ...schedule.shared, mvsControl, mvsRange },
        groups,
      ),
  );
  yield* propagationSteps(schedule, 'mvsPropagateGeometric', mvsControl, K, T + 1, G, T);
}

/** The control of color pass `color` of iteration t, which perturbs by 2^−(t − perturbOffset). */
function iterationControl(
  mvsControl: MvsControlRecord,
  t: number,
  color: number,
  perturbOffset: number,
): MvsControlRecord {
  const scale = 2 ** -(t - perturbOffset);
  return {
    ...mvsControl,
    color,
    iteration: t,
    perturbDepth: scale,
    perturbAngle: ((PERTURB_ANGLE_DEG * Math.PI) / 180) * scale,
  };
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
    function* iteration() {
      for (let color = 0; color < 2; color++) {
        const control = iterationControl(mvsControl, t, color, perturbOffset);
        yield* chunkSteps(schedule, key, 'mvsPropagate', colorIndices, K, (mvsRange, groups) =>
          (schedule.device.backend === 'cpu' ? mvsPropagateOracle : mvsPropagate)(
            { ...schedule.shared, mvsControl: control, mvsRange },
            groups,
          ),
        );
      }
    }
    yield* withAfter(iteration(), after && (() => after(t)));
  }
}

/** The thresholds of `mvsFilter`, COLMAP's filter defaults (section 3.9). */
const FILTER = { minNcc: 0.1, minConsistent: 2, maxError: 1, minAngleDeg: 3 } as const;
const GEOMETRIC_DEFAULTS = { geometricIterations: 4, filter: true } as const;
/** The uniform of `mvsFilter`. `minConfidence` is the option of section 10.3 (0 keeps the rules of 3.9 alone). */
function filterControlOf(minConfidence: number | undefined, writeConfidence = false) {
  return {
    maxCost: 1 - FILTER.minNcc,
    minConsistent: FILTER.minConsistent,
    maxError: FILTER.maxError,
    minAngleCos: Math.cos((FILTER.minAngleDeg * Math.PI) / 180),
    minConfidence: minConfidence ?? 0,
    writeConfidence: writeConfidence ? 1 : 0,
  };
}

/**
 * A cache of view slots (Amendment 3): `place` keeps the views already in a slot where they are
 * and puts the others in the least recently used slots that hold no view of the request.
 */
class SlotCache {
  private readonly held: Int32Array;
  private readonly used: Float64Array;
  private clock = 0;
  constructor(readonly slots: number) {
    this.held = new Int32Array(slots).fill(-1);
    this.used = new Float64Array(slots);
  }
  place(views: readonly number[]): { slotOf: Map<number, number>; loads: [number, number][] } {
    if (views.length > this.slots) throw new RangeError('A window has more views than slots');
    const slotOf = new Map<number, number>();
    for (const v of views) {
      const s = this.held.indexOf(v);
      if (s >= 0) slotOf.set(v, s);
    }
    const taken = new Set(slotOf.values());
    const loads: [number, number][] = [];
    for (const v of views) {
      if (slotOf.has(v)) continue;
      let best = -1;
      for (let s = 0; s < this.slots; s++)
        if (!taken.has(s) && (best < 0 || this.used[s]! < this.used[best]!)) best = s;
      this.held[best] = v;
      slotOf.set(v, best);
      taken.add(best);
      loads.push([v, best]);
    }
    for (const s of slotOf.values()) this.used[s] = ++this.clock;
    return { slotOf, loads };
  }
}

/** One job's device, plan and report: every plan it runs goes through `steps`. */
interface MvsJob {
  readonly device: ShadeDevice;
  readonly plan: MvsViewPlan;
  readonly packedImages: readonly MvsPacked[];
  readonly P: number;
  readonly width: number;
  readonly height: number;
  readonly mvsControl: MvsControlRecord;
  readonly o: ReturnType<typeof prepareMvs>['o'];
  readonly chunkPixels: number | undefined;
  readonly signal: AbortSignal | undefined;
  /** The filter writes each pixel's confidence to `mvsCost`, and the depth stage reads it back
   *  (sections 10.3 and 10.4). */
  readonly confidence: boolean;
  readonly reports: ShadePlanReport[];
  readonly owners: MvsOwner[];
}

async function steps(job: MvsJob, list: Iterable<MvsStep>): Promise<void> {
  job.signal?.throwIfAborted();
  job.reports.push(await job.device.planSteps(cancellable(list, job.signal)));
}

/** The plans of a job as one report: their groups in order and the rates of their chunk keys. */
function jobReport(job: MvsJob): ShadePlanReport {
  const chunks = Object.assign({}, ...job.reports.map((r) => r.chunks ?? {}));
  return {
    tier: job.device.backend,
    dispatches: job.reports.reduce((n, r) => n + r.dispatches, 0),
    submissions: job.reports.reduce((n, r) => n + r.submissions, 0),
    groups: job.reports.flatMap((r) => r.groups),
    ...(Object.keys(chunks).length ? { chunks } : {}),
  };
}

/** The sources of packed view r, best first (3.2). */
function sourcesOf(plan: MvsViewPlan, r: number): number[] {
  const base = r * MVS_SOURCE_SLOTS;
  return Array.from(plan.sources.subarray(base + 1, base + 1 + plan.sources[base]!));
}

/** `mvsCameras` of a window: slot s holds the record of the view in it (element 23 its index). */
function slotCameras(plan: MvsViewPlan, slots: number, slotOf: ReadonlyMap<number, number>) {
  const cameras = new Float32Array(slots * MVS_CAMERA_FLOATS);
  for (const [v, s] of slotOf)
    cameras.set(
      plan.cameras.subarray(v * MVS_CAMERA_FLOATS, (v + 1) * MVS_CAMERA_FLOATS),
      s * MVS_CAMERA_FLOATS,
    );
  return cameras;
}

/** The residency of a job within the device's budget now (Amendment 3). */
function jobResidency(
  device: ShadeDevice,
  job: Pick<MvsJob, 'plan' | 'width' | 'height' | 'o'>,
  references: readonly number[],
  fusionSlots: number,
): MvsResidency {
  const s = device.statistics;
  const available = Math.max(0, device.workingBytes - (s.liveBytes - s.pooledBytes));
  const budgetBytes = Math.min(job.o.residentBytes ?? available, available);
  let maxSources = 1,
    seeds = 1;
  for (const r of references) {
    maxSources = Math.max(maxSources, job.plan.sources[r * MVS_SOURCE_SLOTS]!);
    seeds = Math.max(seeds, job.plan.seedOffsets[r + 1]! - job.plan.seedOffsets[r]!);
  }
  return planMvsResidency({
    width: job.width,
    height: job.height,
    maxSources,
    fusionSlots,
    seedFloats: MVS_SEED_FLOATS * seeds,
    bestSources: job.o.bestSources,
    budgetBytes,
    bindingBytes: device.bindingBytes,
    ...(job.o.streamGroup !== undefined ? { streamGroup: job.o.streamGroup } : {}),
  });
}

/** The arrays of one depth pass (`mvsDepthArrays`), allocated by `owner`. */
function allocateDepthArrays(
  job: MvsJob,
  owner: MvsOwner,
  residency: MvsDepthResidency,
  geometric: boolean,
  seedFloats: number,
) {
  const counts = mvsDepthArrays(
    { width: job.width, height: job.height, seedFloats, bestSources: job.o.bestSources },
    geometric,
    residency.slots,
    residency.elements,
  );
  return {
    mvsCameras: owner.f32(counts.mvsCameras!),
    mvsSources: owner.u32(counts.mvsSources!),
    mvsSeeds: owner.f32(counts.mvsSeeds!),
    mvsGray: owner.f32(counts.mvsGray!),
    mvsDepth: owner.f32(counts.mvsDepth!),
    mvsGeoDepth: owner.f32(counts.mvsGeoDepth!),
    mvsNormal: owner.u32(counts.mvsNormal!),
    mvsCost: owner.f32(counts.mvsCost!),
    stageU32: owner.u32(counts.stageU32!),
    stageF32: owner.f32(counts.stageF32!),
    mvsPartial: counts.mvsPartial ? owner.f32(counts.mvsPartial) : undefined,
  };
}
type MvsDepthArrays = ReturnType<typeof allocateDepthArrays>;

/** The arrays the window-mode kernels bind (4.1). */
function sharedArrays(A: MvsDepthArrays): MvsSharedArrays {
  const { mvsCameras, mvsSources, mvsGray, mvsDepth, mvsGeoDepth, mvsNormal, mvsCost } = A;
  return { mvsCameras, mvsSources, mvsGray, mvsDepth, mvsGeoDepth, mvsNormal, mvsCost };
}

/**
 * Upload packed view v to slot `slot` (3.4, Amendment 3): its RGBA to the stage, its undistorted
 * gray image to the slot of `mvsGray`, and with `depth` its photometric depth to the slot of
 * `mvsDepth`.
 */
async function loadDepthView(
  job: MvsJob,
  A: MvsDepthArrays,
  v: number,
  slot: number,
  depth: Float32Array | undefined,
): Promise<void> {
  const { P, width, height } = job;
  const view = job.packedImages[v]!;
  const groups = Math.ceil(P / LANES);
  A.stageU32.write(packRgba(view.image.rgba, P));
  await steps(job, [
    [
      'mvsUndistort',
      () => {
        rgbaToGray(
          { rgba: A.stageU32, gray: A.stageF32, pixelWindow: { offset: 0, count: P } },
          groups,
        );
        mvsUndistort(
          {
            undistortInput: A.stageF32,
            undistortOutput: A.mvsGray,
            undistortControl: {
              width,
              height,
              view: slot,
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
          groups,
        );
      },
    ],
  ]);
  if (view.image.observationMask) {
    A.stageF32.write(Float32Array.from(view.image.observationMask.values));
    await steps(job, [
      [
        'mvsApplyObservationMask',
        () =>
          mvsApplyObservationMask(
            {
              undistortInput: A.stageF32,
              undistortOutput: A.mvsGray,
              undistortControl: {
                width,
                height,
                view: slot,
                distorted: 0,
                fx: view.camera.fx,
                fy: view.camera.fy,
                cx: view.camera.cx,
                cy: view.camera.cy,
                k1: view.k1,
                k2: view.k2,
                radialLimit: 1e30,
              },
            },
            groups,
          ),
      ],
    ]);
  }
  if (!depth) return;
  A.stageF32.write(depth);
  await steps(job, [
    [
      'mvsCopyF32',
      () =>
        mvsCopyF32(
          {
            copyInputF32: A.stageF32,
            copyOutputF32: A.mvsDepth,
            copyControl: { count: P, offset: slot * P },
          },
          groups,
        ),
    ],
  ]);
}

/**
 * Make `views` resident in the slots of `cache`, upload the new ones, and write the window's
 * camera records and the source record of `sources` (the views after the first, by slot).
 */
async function placeDepthWindow(
  job: MvsJob,
  A: MvsDepthArrays,
  cache: SlotCache,
  views: readonly number[],
  depths: readonly (Float32Array | undefined)[] | undefined,
): Promise<Map<number, number>> {
  const { slotOf, loads } = cache.place(views);
  for (const [v, slot] of loads) await loadDepthView(job, A, v, slot, depths?.[v]);
  A.mvsCameras.write(slotCameras(job.plan, cache.slots, slotOf));
  const record = new Uint32Array(MVS_SOURCE_SLOTS);
  record[0] = views.length - 1;
  views.slice(1).forEach((s, k) => (record[1 + k] = slotOf.get(s)!));
  A.mvsSources.write(record);
  return slotOf;
}

/** The streamed phases of Amendment 3 (`MvsStreamControl.phase` in `mvs.shade.ts`). */
const PHASE = { initialize: 0, seeds: 1, candidates: 2, perturbed: 3, geometric: 4, filter: 5 };

/**
 * One streamed phase of reference r over `total` elements: for each chunk that `mvsPartial`
 * holds, the costs of every group of sources (each group made resident in turn), then the
 * selection. `order` alternates the groups, so the last group of one phase is the first of the
 * next and stays resident.
 */
async function streamPhase(
  job: MvsJob,
  A: MvsDepthArrays,
  cache: SlotCache,
  r: number,
  groups: readonly number[][],
  depths: readonly (Float32Array | undefined)[] | undefined,
  phase: number,
  total: number,
  control: MvsControlRecord,
  flip: { forward: boolean },
): Promise<void> {
  const L = job.o.bestSources;
  const capacity = A.mvsPartial ? residencyElements(A) * MVS_PARTIAL_HYPOTHESES * L : 0;
  const stride =
    phase === PHASE.candidates ? MVS_PARTIAL_HYPOTHESES * L : phase === PHASE.filter ? L + 1 : L;
  const hypotheses = phase === PHASE.candidates ? MVS_PARTIAL_HYPOTHESES : 1;
  let per = Math.floor(capacity / stride);
  if (per < total) per -= per % LANES;
  const K = groups.reduce((n, g) => n + g.length, 0);
  const geometric = control.geometric !== 0;
  const schedule = { device: job.device, chunkPixels: job.chunkPixels };
  const costKey: MvsChunkKey = geometric ? 'mvsStreamCostsGeometric' : 'mvsStreamCosts';
  for (let first = 0; first < total; first += per) {
    const count = Math.min(per, total - first);
    const order = flip.forward ? groups : [...groups].reverse();
    flip.forward = !flip.forward;
    for (let g = 0; g < order.length; g++) {
      const group = order[g]!;
      const slotOf = await placeDepthWindow(job, A, cache, [r, ...group], depths);
      const mvsControl = { ...control, reference: slotOf.get(r)! };
      const streamControl = { phase, start: g === 0 ? 1 : 0, first, stride, costs: L, sources: K };
      await steps(
        job,
        chunkSteps(
          schedule,
          costKey,
          'mvsStreamCosts',
          count,
          hypotheses * group.length,
          (mvsRange, workgroups) =>
            (job.device.backend === 'cpu' ? mvsStreamCostsOracle : mvsStreamCosts)(
              {
                mvsControl,
                mvsRange,
                streamControl,
                filterControl: filterControlOf(job.o.minConfidence, job.confidence),
                mvsSeeds: A.mvsSeeds,
                mvsCameras: A.mvsCameras,
                mvsSources: A.mvsSources,
                mvsGray: A.mvsGray,
                mvsDepth: A.mvsDepth,
                mvsGeoDepth: A.mvsGeoDepth,
                mvsNormal: A.mvsNormal,
                mvsPartial: A.mvsPartial!,
              },
              workgroups,
            ),
          costKey,
          first,
        ),
      );
    }
    // The reference is resident in its slot after every group.
    const reference = cache.place([r]).slotOf.get(r)!;
    const mvsControl = { ...control, reference };
    const streamControl = { phase, start: 0, first, stride, costs: L, sources: K };
    await steps(
      job,
      chunkSteps(
        schedule,
        'mvsStreamSelect',
        'mvsStreamSelect',
        count,
        1,
        (mvsRange, workgroups) =>
          mvsStreamSelect(
            {
              mvsControl,
              mvsRange,
              streamControl,
              filterControl: filterControlOf(job.o.minConfidence, job.confidence),
              mvsSeeds: A.mvsSeeds,
              mvsCameras: A.mvsCameras,
              mvsGray: A.mvsGray,
              mvsDepth: A.mvsDepth,
              mvsGeoDepth: A.mvsGeoDepth,
              mvsNormal: A.mvsNormal,
              mvsCost: A.mvsCost,
              mvsPartial: A.mvsPartial!,
            },
            workgroups,
          ),
        'mvsStreamSelect',
        first,
      ),
    );
  }
}
/** The elements `mvsPartial` holds: its length over 9 hypotheses × `bestSources`. */
const partialElements = new WeakMap<object, number>();
function residencyElements(A: MvsDepthArrays): number {
  return partialElements.get(A) ?? 0;
}

/** The groups of at most `size` sources, in source order. */
function sourceGroups(sources: readonly number[], size: number): number[][] {
  const groups: number[][] = [];
  for (let k = 0; k < sources.length; k += size) groups.push(sources.slice(k, k + size));
  return groups;
}

/** What one reference's depth pass reads back to the host. */
interface MvsPassOutput {
  readonly depth: Float32Array;
  readonly normal: Uint32Array;
  readonly cost?: Float32Array;
}

/** Keep excluded paths out of exported maps, geometric evidence and later completion/fusion. */
function excludeObservationPaths(
  job: MvsJob,
  reference: number,
  depth: Float32Array,
  normal?: Uint32Array,
  auxiliary?: Float32Array,
  emptyAuxiliary = 2,
): void {
  const mask = job.packedImages[reference]!.image.observationMask;
  if (!mask) return;
  for (let i = 0; i < job.P; i++)
    if (mask.values[i] !== 0) {
      depth[i] = 0;
      if (normal) normal[i] = 0;
      if (auxiliary) auxiliary[i] = emptyAuxiliary;
    }
}

/** The reference's arrays now: its depth (`mvsDepth` or `mvsGeoDepth`), normal and cost. */
async function readState(
  job: MvsJob,
  A: MvsDepthArrays,
  geometric: boolean,
  withCost: boolean,
  reference: number,
): Promise<MvsPassOutput> {
  const { device, P } = job;
  const [depth, normal, cost] = await readTogether([
    device.read(geometric ? A.mvsGeoDepth : A.mvsDepth, P).then((a) => a.slice(0, P)),
    device.read(A.mvsNormal, P).then((a) => a.slice(0, P)),
    withCost ? device.read(A.mvsCost, P).then((a) => a.slice(0, P)) : Promise.resolve(undefined),
  ]);
  excludeObservationPaths(job, reference, depth, normal, cost, geometric && job.confidence ? 0 : 2);
  return { depth, normal, ...(cost ? { cost } : {}) };
}

/**
 * The photometric pass of reference r (3.5 and 3.6) in the residency's mode. `after(t)` runs
 * after the seeds (t = 0) and after each iteration t.
 */
async function photometricPass(
  job: MvsJob,
  A: MvsDepthArrays,
  residency: MvsDepthResidency,
  cache: SlotCache,
  r: number,
  after?: (t: number) => Promise<void>,
): Promise<void> {
  const { plan, o, mvsControl } = job;
  const sources = sourcesOf(plan, r);
  const seedFirst = plan.seedOffsets[r]!,
    seedEnd = o.seeds ? plan.seedOffsets[r + 1]! : seedFirst,
    seedCount = seedEnd - seedFirst;
  const seeds = new Float32Array(seedCapacity(A));
  seeds.set(plan.seeds.subarray(MVS_SEED_FLOATS * seedFirst, MVS_SEED_FLOATS * seedEnd));
  A.mvsSeeds.write(seeds);
  const T = o.photometricIterations;
  if (residency.mode === 'window') {
    const slotOf = await placeDepthWindow(job, A, cache, [r, ...sources], undefined);
    const control = { ...mvsControl, reference: slotOf.get(r)!, view: r, state: 0 };
    await steps(
      job,
      photometricSteps(
        { device: job.device, shared: sharedArrays(A), chunkPixels: job.chunkPixels },
        control,
        sources.length,
        A.mvsSeeds,
        0,
        seedCount,
        T,
        after,
      ),
    );
    return;
  }
  const groups = sourceGroups(sources, residency.slots - 1);
  const control = { ...mvsControl, view: r, state: 0 };
  const flip = { forward: true };
  const P = job.width * job.height;
  await streamPhase(job, A, cache, r, groups, undefined, PHASE.initialize, P, control, flip);
  if (seedCount > 0)
    await streamPhase(job, A, cache, r, groups, undefined, PHASE.seeds, seedCount, control, flip);
  await after?.(0);
  await streamIterations(job, A, cache, r, groups, undefined, control, 1, T, 0, after);
}

/** The seed capacity of `mvsSeeds` in f32. */
const seedCapacities = new WeakMap<object, number>();
function seedCapacity(A: MvsDepthArrays): number {
  return seedCapacities.get(A) ?? 0;
}

/** Streamed red-black iterations `first..first + count − 1` (3.6), as `propagationSteps`. */
async function streamIterations(
  job: MvsJob,
  A: MvsDepthArrays,
  cache: SlotCache,
  r: number,
  groups: readonly number[][],
  depths: readonly (Float32Array | undefined)[] | undefined,
  control: MvsControlRecord,
  first: number,
  count: number,
  perturbOffset: number,
  after?: (t: number) => Promise<void>,
): Promise<void> {
  const colorIndices = Math.ceil(job.width / 2) * job.height;
  const flip = { forward: true };
  for (let t = first; t < first + count; t++) {
    for (let color = 0; color < 2; color++) {
      const pass = iterationControl(control, t, color, perturbOffset);
      for (const phase of [PHASE.candidates, PHASE.perturbed])
        await streamPhase(job, A, cache, r, groups, depths, phase, colorIndices, pass, flip);
    }
    await after?.(t);
  }
}

/**
 * The geometric pass (3.8) and, with `filter`, the filter (3.9) of reference r in the residency's
 * mode. `mvsNormal` holds the reference's photometric normal, and the slots of `mvsDepth` the
 * photometric depths of the reference and its sources. `beforeFilter` runs after the geometric
 * pass.
 */
async function geometricPass(
  job: MvsJob,
  A: MvsDepthArrays,
  residency: MvsDepthResidency,
  cache: SlotCache,
  r: number,
  photometric: readonly (Float32Array | undefined)[],
  normal: Uint32Array,
  G: number,
  filter: boolean,
  beforeFilter?: () => Promise<void>,
): Promise<void> {
  const { plan, o, mvsControl, P } = job;
  const sources = sourcesOf(plan, r);
  const T = o.photometricIterations;
  A.mvsNormal.write(normal);
  if (residency.mode === 'window') {
    const slotOf = await placeDepthWindow(job, A, cache, [r, ...sources], photometric);
    const control = { ...mvsControl, reference: slotOf.get(r)!, view: r, state: 0, geometric: 1 };
    const schedule = { device: job.device, shared: sharedArrays(A), chunkPixels: job.chunkPixels };
    await steps(job, geometricSteps(schedule, control, sources.length, T, G));
    await beforeFilter?.();
    if (!filter) return;
    // The filter binds `mvsCost` for the confidence of section 10.4 (`writeConfidence`).
    const arrays = A;
    await steps(
      job,
      chunkSteps(schedule, 'mvsFilter', 'mvsFilter', P, sources.length, (mvsRange, groups) =>
        (job.device.backend === 'cpu' ? mvsFilterOracle : mvsFilter)(
          {
            mvsCameras: arrays.mvsCameras,
            mvsSources: arrays.mvsSources,
            mvsGray: arrays.mvsGray,
            mvsDepth: arrays.mvsDepth,
            mvsGeoDepth: arrays.mvsGeoDepth,
            mvsNormal: arrays.mvsNormal,
            mvsCost: arrays.mvsCost,
            mvsControl: control,
            filterControl: filterControlOf(job.o.minConfidence, job.confidence),
            mvsRange,
          },
          groups,
        ),
      ),
    );
    return;
  }
  const groups = sourceGroups(sources, residency.slots - 1);
  const control = { ...mvsControl, view: r, state: 0, geometric: 1 };
  const flip = { forward: true };
  await streamPhase(job, A, cache, r, groups, photometric, PHASE.geometric, P, control, flip);
  await streamIterations(job, A, cache, r, groups, photometric, control, T + 1, G, T);
  await beforeFilter?.();
  if (filter)
    await streamPhase(job, A, cache, r, groups, photometric, PHASE.filter, P, control, flip);
}

/** The depth stage's outputs on the host, by packed index (Amendment 3, item 7). */
interface MvsHostMaps {
  readonly photometricDepth: (Float32Array | undefined)[];
  readonly photometricNormal: (Uint32Array | undefined)[];
  readonly depth: (Float32Array | undefined)[];
  readonly normal: (Uint32Array | undefined)[];
  readonly cost: (Float32Array | undefined)[];
  /** The confidence of each filtered depth (10.3), 0 without depth, when the job asks for it. */
  readonly confidence: (Float32Array | undefined)[];
  readonly photometricStage: (MvsDepthSnapshot | undefined)[];
  readonly geometricStage: (MvsDepthSnapshot | undefined)[];
}

interface MvsDepthStageOptions {
  readonly order: readonly number[];
  readonly geometricOrder: readonly number[];
  readonly geometricIterations: number;
  readonly filter: boolean;
  /** Keep each reference's cost after the filter, and the stage snapshots. */
  readonly costs: boolean;
  readonly readStages: boolean;
  /** Keep the photometric normal of a view that runs no geometric pass. */
  readonly keepPhotometric: boolean;
  readonly onPhotometric?: (r: number, done: number) => void;
  readonly onGeometric?: (r: number, done: number) => void;
  /**
   * Amendment 5: reference r's photometric depth on the host, after its pass, and after each of
   * `photometricPreviewIterations` (the iteration as `iteration`). A read of the array only.
   */
  readonly onPhotometricDepth?: (
    r: number,
    depth: Float32Array,
    iteration?: number,
  ) => Promise<void>;
  /** Amendment 5: iterations of the photometric pass after which `onPhotometricDepth` runs too. */
  readonly photometricPreviewIterations?: readonly number[];
  /** Amendment 5: reference r's filtered depth on the host, after its geometric pass and filter. */
  readonly onGeometricDepth?: (r: number, depth: Float32Array) => Promise<void>;
  /** Borrowed full-resolution arrays, awaited before reuse; never iteration previews. */
  readonly onDiagnostic?: (
    r: number,
    stage: GpuDenseDepthDiagnostic['stage'],
    depth: Float32Array,
    confidence?: Float32Array,
  ) => Promise<void>;
  /** Called after the last filter, while the geometric pass's arrays are still resident. */
  readonly afterDepth?: (maps: MvsHostMaps) => Promise<void>;
}

/** Allocate one depth pass's arrays for `residency`, with their capacities recorded. */
function depthArrays(
  job: MvsJob,
  residency: MvsDepthResidency,
  geometric: boolean,
  seedFloats: number,
) {
  const owner = ownArrays(job.device);
  job.owners.push(owner);
  job.device.reserveWorkingBytes(residency.bytes);
  const A = allocateDepthArrays(job, owner, residency, geometric, seedFloats);
  partialElements.set(A, residency.elements);
  seedCapacities.set(A, Math.max(3, seedFloats));
  return { A, owner };
}

/**
 * The depth stage of sections 3.4 to 3.9 (Amendment 3): the photometric pass of each reference
 * of `order`, then the geometric pass and the filter of each of `geometricOrder`. Each pass reads
 * its outputs back to the host, and a photometric output is released after its last reader.
 */
async function depthStage(
  job: MvsJob,
  residency: MvsResidency,
  options: MvsDepthStageOptions,
): Promise<MvsHostMaps> {
  const { plan } = job;
  const N = plan.packed.length;
  const maps: MvsHostMaps = {
    photometricDepth: new Array(N),
    photometricNormal: new Array(N),
    depth: new Array(N),
    normal: new Array(N),
    cost: new Array(N),
    confidence: new Array(N),
    photometricStage: new Array(N),
    geometricStage: new Array(N),
  };
  const seedFloats =
    MVS_SEED_FLOATS *
    Math.max(1, ...options.order.map((r) => plan.seedOffsets[r + 1]! - plan.seedOffsets[r]!));
  // The photometric pass of every reference.
  {
    const { A, owner } = depthArrays(job, residency.photometric, false, seedFloats);
    const cache = new SlotCache(residency.photometric.slots);
    let done = 0;
    // The last iteration's depth is the pass's own preview, so an iteration preview comes before it.
    const iterations = options.onPhotometricDepth
      ? (options.photometricPreviewIterations ?? []).filter(
          (t) => t >= 1 && t < job.o.photometricIterations,
        )
      : [];
    for (const r of options.order) {
      // Amendment 5, rule 6: the depth after a few iterations, read without changing the pass.
      const after = iterations.length
        ? async (t: number) => {
            if (!iterations.includes(t)) return;
            const depth = (await job.device.read(A.mvsDepth, job.P)).slice(0, job.P);
            excludeObservationPaths(job, r, depth);
            await options.onPhotometricDepth!(r, depth, t);
          }
        : undefined;
      await photometricPass(job, A, residency.photometric, cache, r, after);
      const out = await readState(job, A, false, options.readStages, r);
      await options.onPhotometricDepth?.(r, out.depth);
      await options.onDiagnostic?.(r, 'photometric', out.depth);
      maps.photometricDepth[r] = out.depth;
      maps.photometricNormal[r] = out.normal;
      if (options.readStages)
        maps.photometricStage[r] = { depth: out.depth, normal: out.normal, cost: out.cost! };
      options.onPhotometric?.(r, ++done);
    }
    owner.release(true);
  }
  // Each photometric depth is read by its own geometric pass and by those of its references.
  const readers = new Int32Array(N);
  for (const r of options.geometricOrder) for (const v of [r, ...sourcesOf(plan, r)]) readers[v]!++;
  const { A, owner } = depthArrays(job, residency.geometric, true, 0);
  const cache = new SlotCache(residency.geometric.slots);
  let done = 0;
  for (const r of options.geometricOrder) {
    const before =
      options.readStages || options.onDiagnostic
        ? async () => {
            if (options.readStages) {
              const out = await readState(job, A, true, true, r);
              maps.geometricStage[r] = { depth: out.depth, normal: out.normal, cost: out.cost! };
              await options.onDiagnostic?.(r, 'geometric-unfiltered', out.depth);
            } else {
              const depth = (await job.device.read(A.mvsGeoDepth, job.P)).slice(0, job.P);
              excludeObservationPaths(job, r, depth);
              await options.onDiagnostic!(r, 'geometric-unfiltered', depth);
            }
          }
        : undefined;
    await geometricPass(
      job,
      A,
      residency.geometric,
      cache,
      r,
      maps.photometricDepth,
      maps.photometricNormal[r]!,
      options.geometricIterations,
      options.filter,
      before,
    );
    // With `job.confidence`, `mvsCost` holds the confidence after the filter, not the cost.
    const out = await readState(job, A, true, options.costs || job.confidence, r);
    maps.depth[r] = out.depth;
    maps.normal[r] = out.normal;
    if (job.confidence) maps.confidence[r] = out.cost;
    else if (out.cost) maps.cost[r] = out.cost;
    await options.onGeometricDepth?.(r, out.depth);
    await options.onDiagnostic?.(
      r,
      'geometric-filtered',
      out.depth,
      job.confidence ? out.cost : undefined,
    );
    if (!options.keepPhotometric) maps.photometricNormal[r] = undefined;
    for (const v of [r, ...sourcesOf(plan, r)])
      if (--readers[v]! === 0 && !options.readStages) maps.photometricDepth[v] = undefined;
    options.onGeometric?.(r, ++done);
  }
  await options.afterDepth?.(maps);
  owner.release(true);
  return maps;
}

/** The host maps of a run without a depth stage (`MvsDenseOptions.depthMaps`). */
function givenMaps(given: MvsFilteredMaps, N: number): MvsHostMaps {
  if (given.depth.length !== N || given.normal.length !== N)
    throw new RangeError('depthMaps needs one map per packed view');
  return {
    photometricDepth: new Array(N),
    photometricNormal: new Array(N),
    depth: given.depth.slice(),
    normal: given.normal.slice(),
    cost: new Array(N),
    confidence: given.confidence ? given.confidence.slice() : new Array(N),
    photometricStage: new Array(N),
    geometricStage: new Array(N),
  };
}

/** Shared by the three entries: the job of one enqueued run. */
function createJob(
  device: ShadeDevice,
  scene: MvsScene,
  plan: MvsViewPlan,
  images: ReadonlyMap<number, MvsImage>,
  options: MvsPhotometricOptions & {
    readonly signal?: AbortSignal;
    readonly confidence?: boolean;
  },
) {
  const prepared = prepareMvs(scene, plan, images, options);
  prepared.mvsControl.relativePoseCache =
    device.backend === 'webgpu' && options.relativePoseCache !== false ? 1 : 0;
  const job: MvsJob = {
    device,
    plan,
    packedImages: prepared.packedImages,
    P: prepared.P,
    width: prepared.width,
    height: prepared.height,
    mvsControl: prepared.mvsControl,
    o: prepared.o,
    chunkPixels: options.chunkPixels,
    signal: options.signal,
    confidence: options.confidence === true,
    reports: [],
    owners: [],
  };
  return { job, N: prepared.N };
}

/** Release every array of the job: recycled after a success or a cancel, dropped otherwise. */
function releaseJob(job: MvsJob, successful: boolean): void {
  for (const owner of job.owners) owner.release(successful);
}

/**
 * Runs the photometric pass of section 3.6 on the reference with packed index `reference`:
 * the undistortion of the reference and its sources, the initialization, the seeds and
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
  const { job, N } = createJob(device, scene, plan, images, options);
  checkInteger(reference, 0, N - 1, 'reference');
  if (!(plan.sources[reference * MVS_SOURCE_SLOTS]! > 0))
    throw new RangeError('The reference has no source view');
  const { P, width, height } = job;
  return device.enqueue(async () => {
    let successful = false;
    try {
      const residency = jobResidency(device, job, [reference], 1);
      const seedFloats =
        MVS_SEED_FLOATS *
        Math.max(1, plan.seedOffsets[reference + 1]! - plan.seedOffsets[reference]!);
      const { A } = depthArrays(job, residency.photometric, false, seedFloats);
      const cache = new SlotCache(residency.photometric.slots);
      const snapshots: MvsDepthSnapshot[] = [];
      const snapshot = async () => {
        const out = await readState(job, A, false, true, reference);
        snapshots.push({ depth: out.depth, normal: out.normal, cost: out.cost! });
      };
      await photometricPass(
        job,
        A,
        residency.photometric,
        cache,
        reference,
        options.snapshots ? snapshot : undefined,
      );
      const out = await readState(job, A, false, true, reference);
      let gray: Float32Array | undefined;
      if (options.readGray) {
        // Each packed view through slot 0, read whole.
        gray = new Float32Array(N * P);
        for (let v = 0; v < N; v++) {
          await loadDepthView(job, A, v, 0, undefined);
          gray.set((await device.read(A.mvsGray, P)).subarray(0, P), v * P);
        }
      }
      successful = true;
      return {
        width,
        height,
        reference,
        depth: out.depth,
        normal: out.normal,
        cost: out.cost!,
        ...(options.snapshots ? { snapshots } : {}),
        ...(gray ? { gray } : {}),
        residency,
        report: jobReport(job),
      };
    } finally {
      releaseJob(job, successful);
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
  readonly residency: MvsResidency;
  readonly report: ShadePlanReport;
}

/**
 * The depth stage of sections 3.4 to 3.9 for every packed view: the photometric pass of every
 * reference in `order`, then for every reference in `geometricOrder` the geometric pass and the
 * filter. The geometric pass and the filter read the sources' photometric depths and write only
 * the reference's outputs, so the order of the references changes no result.
 */
export function mvsDepthMapsGpu(
  device: ShadeDevice,
  scene: MvsScene,
  plan: MvsViewPlan,
  images: ReadonlyMap<number, MvsImage>,
  options: MvsDepthMapsOptions = {},
): Promise<MvsDepthMapsResult> {
  const { job, N } = createJob(device, scene, plan, images, options);
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
  const { P, width, height } = job;
  return device.enqueue(async () => {
    let successful = false;
    try {
      const residency = jobResidency(device, job, order, 1);
      const maps = await depthStage(job, residency, {
        order,
        geometricOrder,
        geometricIterations: g.geometricIterations,
        filter: g.filter,
        costs: true,
        readStages: options.readStages ?? false,
        keepPhotometric: true,
      });
      const views = Array.from({ length: N }, (_, r) => ({
        reference: r,
        depth: maps.depth[r] ?? new Float32Array(P),
        normal: maps.normal[r] ?? maps.photometricNormal[r]!,
        cost: maps.cost[r] ?? new Float32Array(P),
        ...(options.readStages ? { photometric: maps.photometricStage[r]! } : {}),
        ...(options.readStages && maps.geometricStage[r]
          ? { geometric: maps.geometricStage[r] }
          : {}),
      }));
      successful = true;
      return { width, height, views, residency, report: jobReport(job) };
    } finally {
      releaseJob(job, successful);
    }
  });
}

/** The fusion thresholds, COLMAP's `stereo_fusion` defaults (section 3.10). */
const FUSION = { maxDepthError: 0.01, maxReprojection: 2, maxNormalDeg: 10 } as const;
const DENSE_DEFAULTS = { minConsistentViews: 3 } as const;
/** A depth preview has a quarter of the image's size in each axis (5.3). */
const PREVIEW_STEP = 4;
/** The photometric iterations after which a reference's depth is previewed (Amendment 5, rule 6). */
const PREVIEW_ITERATIONS: readonly number[] = [1, 3];

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

/** Full-resolution diagnostic map. Arrays are borrowed until the callback resolves: do not mutate or transfer them. */
export interface GpuDenseDepthDiagnostic {
  readonly viewId: number;
  readonly stage: 'photometric' | 'geometric-unfiltered' | 'geometric-filtered' | 'planar';
  readonly width: number;
  readonly height: number;
  /** Camera-axis Z in the normalized frame; zero means invalid. Multiply by depthScaleToModel. */
  readonly depth: Float32Array;
  /** Filter confidence [0,1], zero for invalid; available after filtering and planar completion. */
  readonly confidence?: Float32Array;
  readonly depthScaleToModel: number;
  /** Undistorted pinhole grid; original image pixels may require radial undistortion. */
  readonly camera: { fx: number; fy: number; cx: number; cy: number };
  readonly sourceViewIds: readonly number[];
  /** World-to-camera pose in the original model frame. */
  readonly pose: { R: readonly number[]; t: readonly number[] };
}

/** What `onView` receives (5.3). Its arrays are its own: a worker can transfer them. */
export type GpuDenseViewEvent =
  | {
      readonly type: 'depth';
      readonly viewId: number;
      /**
       * Amendment 5: `photometric` after the reference's photometric pass (not filtered), and
       * after the iterations of `iteration`; `geometric` after its geometric pass and filter.
       */
      readonly stage: 'photometric' | 'geometric';
      /** The photometric iteration of an iteration preview (Amendment 5, rule 6); absent after a pass. */
      readonly iteration?: number;
      /** The preview's size, ⌈W / 4⌉ × ⌈H / 4⌉. */
      readonly width: number;
      readonly height: number;
      /**
       * The depth at a quarter of the size in each axis, in the model's units: the mean of the
       * depths of each 4 × 4 block, 0 for a block without depth.
       */
      readonly depth: Float32Array;
    }
  | ({
      readonly type: 'points';
      /** The reference, or −1 for the whole cloud of the TSDF fusion (`fusion` `'tsdf'`). */
      readonly viewId: number;
    } & GpuDensePoints);

/** The stages of `onProgress` (5.3): the photometric pass, the geometric pass with the filter, and
 *  the fusion. */
export type GpuDenseStage = 'photometric' | 'geometric' | 'fusion';

/** What `onProgress` receives after each reference of each stage (5.3). */
export interface GpuDenseProgress {
  readonly stage: GpuDenseStage;
  /** The reference that completed the stage. */
  readonly viewId: number;
  /** The references that completed this stage, this one included, of `total`. */
  readonly done: number;
  readonly total: number;
}

/** Live TSDF work, distinct from the completed-reference counts of `onProgress`. */
export interface GpuDenseFusionProgress extends TsdfProgress {
  /** Scene view id for the current input, absent while working on the volume. */
  readonly viewId?: number;
}

export interface GpuDenseOptions extends MvsViewOptions {
  /** Centered, edge-aware patch support (`MvsPhotometricOptions.boundaryAware`). Default false. */
  readonly boundaryAware?: boolean;
  /** Finite-cost margin over a valid directional warp (`MvsPhotometricOptions.minFiniteGain`). Default 0. */
  readonly minFiniteGain?: number;
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
  /** The least confidence of a kept depth (section 10.3). Default 0: the rules of 3.9 alone. */
  readonly minConfidence?: number;
  /**
   * The fusion of the filtered depth maps: `'pixel'`, the per-pixel fusion of section 3.10, or
   * `'tsdf'`, the TSDF fusion of section 10.4 with the confidence of 10.3 as its weight. Default
   * `'pixel'`.
   */
  readonly fusion?: 'pixel' | 'tsdf';
  /** The options of the TSDF fusion (section 10.4), with `fusion` `'tsdf'`. */
  readonly tsdf?: TsdfOptions;
  /** Complete local planar holes only when two original filtered source maps verify them. TSDF only. */
  readonly planarDepth?: boolean;
  /**
   * With `fusion` `'tsdf'`: the visibility labeling of section 10.5 of the TSDF's points, with
   * these options, before the points are returned. The sparse landmarks of `scene` support the
   * points within 4 pixel footprints (10.2). Default: no labeling.
   */
  readonly labeling?: DenseLabelOptions;
  /**
   * The depth previews of each reference after its photometric pass, after iterations 1 and 3 of
   * that pass, and after its geometric pass and filter (Amendment 5); and the points of each
   * reference after its fusion (5.3). The run waits for a returned promise. The answer is the
   * same with and without it. With `fusion` `'tsdf'`, one `points` event with `viewId` −1 carries
   * the whole cloud after the fusion.
   */
  readonly onView?: (event: GpuDenseViewEvent) => void | Promise<void>;
  /** Full-resolution maps at stage boundaries, distinct from display previews. Awaited; arrays are borrowed. */
  readonly onDepthDiagnostic?: (event: GpuDenseDepthDiagnostic) => void | Promise<void>;
  /** Called after each reference of each stage (5.3), when the stage's last plan has completed. */
  readonly onProgress?: (event: GpuDenseProgress) => void;
  /** TSDF phases and submitted chunks, without additional numerical readbacks or GPU fences. */
  readonly onFusionProgress?: (event: GpuDenseFusionProgress) => void;
  /** Extracted TSDF boxes before the final sort/labeling. Own arrays; final `onView` is unchanged. */
  readonly onFusionPreview?: (points: GpuDensePoints) => void | Promise<void>;
  /**
   * Stops the run (5.3, the lab's cancel). The run reads the signal before each plan, so a chunk
   * in flight completes. It then gives its arrays back to the device and rejects with the
   * signal's reason.
   */
  readonly signal?: AbortSignal;
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
  /**
   * The arrays each stage kept on the device and its mode (Amendment 3): window or streamed,
   * within the device's budget. Undefined when no view has a depth map.
   */
  readonly residency?: MvsResidency;
  /** With `fusion` `'tsdf'`: the volume, the passes and the bytes of the TSDF fusion (10.4). */
  readonly tsdf?: TsdfReport;
  readonly planar?: PlanarDepthReport;
  /** With `labeling`: what the labeling of section 10.5 removed and kept. */
  readonly labeling?: DenseLabelReport;
  readonly report: ShadePlanReport;
}

/** @internal The filtered depth maps of the packed views, for a fusion without a depth stage. */
export interface MvsFilteredMaps {
  readonly depth: readonly Float32Array[];
  readonly normal: readonly Uint32Array[];
  /** Present when the run read the confidence (`fusion` `'tsdf'`). */
  readonly confidence?: readonly Float32Array[];
}

/** @internal The options of `reconstructDense`, the tests' entry. */
export interface MvsDenseOptions extends GpuDenseOptions {
  /** Compare the workgroup pose cache with the scalar calculation (`MvsPhotometricOptions`). */
  readonly relativePoseCache?: boolean;
  /** Force chunks of at most this many pixels or color indices. Default: one chunk per pass. */
  readonly chunkPixels?: number;
  /** The most bytes the run keeps on the device (`MvsPhotometricOptions.residentBytes`). */
  readonly residentBytes?: number;
  /** Force streamed mode (`MvsPhotometricOptions.streamGroup`). */
  readonly streamGroup?: number;
  /** Called with the filtered maps after the depth stage (the ETH3D script keeps them). */
  readonly onDepthMaps?: (maps: MvsFilteredMaps) => void | Promise<void>;
  /** Skip the depth stage and fuse these maps, by packed index. */
  readonly depthMaps?: MvsFilteredMaps;
  /** The TSDF fusion's passes of at most this many blocks (`TsdfGpuOptions.maxBlocks`). */
  readonly tsdfMaxBlocks?: number;
  /** Called with the TSDF fusion's points and their attributes, in the model's frame. */
  readonly onTsdfPoints?: (points: TsdfPoints) => void | Promise<void>;
}

/**
 * The dense MVS of sections 3.1 to 3.11 after the sparse SfM: the view selection, the depth stage
 * of every view (the photometric pass, the geometric pass and the filter) and the fusion of the
 * filtered depth maps into a colored point cloud with normals. `scene` is the final sparse model
 * (`pose` world to camera, `camera` with k1 and k2). A view without `width` and `height` takes
 * those of its image. `images` maps a view id to its RGBA at the analysis size, one size for
 * every view with a depth map. The points are in the model's frame. The run keeps a window of
 * views on the device, sized from the device's budget (Amendment 3), so the number of views is
 * not limited.
 */
export function reconstructDenseGpu(
  device: ShadeDevice,
  scene: MvsScene,
  images: ReadonlyMap<number, MvsImage>,
  options: GpuDenseOptions = {},
): Promise<GpuDenseResult> {
  return reconstructDense(device, scene, images, options);
}

/** The arrays of the fusion stage (`mvsFusionArrays`), allocated by `owner`. */
function allocateFusionArrays(job: MvsJob, owner: MvsOwner, residency: MvsFusionResidency) {
  const streamed = residency.mode === 'streamed';
  const counts = mvsFusionArrays(
    { width: job.width, height: job.height },
    residency.slots,
    residency.chunk,
    streamed,
  );
  return {
    mvsCameras: owner.f32(counts.mvsCameras!),
    mvsGeoDepth: owner.f32(counts.mvsGeoDepth!),
    mvsNormal: owner.u32(counts.mvsNormal!),
    mvsRgba: owner.u32(counts.mvsRgba!),
    stageU32: owner.u32(counts.stageU32!),
    stageF32: owner.f32(counts.stageF32!),
    mvsFuseSums: counts.mvsFuseSums ? owner.f32(counts.mvsFuseSums) : undefined,
    mvsFlags: owner.u32(counts.mvsFlags!),
    mvsFlagOffsets: owner.u32(counts.mvsFlagOffsets!),
    totals: owner.u32(counts.totals!),
    mvsPoints: owner.f32(counts.mvsPoints!),
    mvsPointData: owner.u32(counts.mvsPointData!),
  };
}
type MvsFusionArrays = ReturnType<typeof allocateFusionArrays>;

/**
 * Make `views` resident in the fusion slots (their filtered depths and normals from the host),
 * write the window's camera records and the order list of `order` (ascending packed index).
 */
async function placeFusionWindow(
  job: MvsJob,
  F: MvsFusionArrays,
  cache: SlotCache,
  views: readonly number[],
  order: readonly number[],
  maps: MvsHostMaps,
): Promise<Map<number, number>> {
  const { P } = job;
  const { slotOf, loads } = cache.place(views);
  for (const [v, slot] of loads) {
    F.stageF32.write(maps.depth[v]!);
    F.stageU32.write(maps.normal[v]!);
    await steps(job, [
      [
        'mvsCopyF32',
        () => {
          const groups = Math.ceil(P / LANES),
            copyControl = { count: P, offset: slot * P };
          mvsCopyF32(
            { copyInputF32: F.stageF32, copyOutputF32: F.mvsGeoDepth, copyControl },
            groups,
          );
          mvsCopyU32({ copyInputU32: F.stageU32, copyOutputU32: F.mvsNormal, copyControl }, groups);
        },
      ],
    ]);
  }
  // The camera records, then the order list (`fuseControl.order`, Amendment 3).
  const cameras = new Float32Array((MVS_CAMERA_FLOATS + 1) * cache.slots);
  cameras.set(slotCameras(job.plan, cache.slots, slotOf));
  order.forEach((v, k) => (cameras[MVS_CAMERA_FLOATS * cache.slots + k] = slotOf.get(v)!));
  F.mvsCameras.write(cameras);
  return slotOf;
}

/** The radius of a landmark in the labeling (10.5, 10.2): this many pixel footprints. */
const LABEL_LANDMARK_FOOTPRINTS = 4;

/**
 * The landmarks of a scene with the radius of `landmarkCoverage` (section 10.2): 4 pixel
 * footprints, depth / focal, in the nearest view that observes the landmark in front of it.
 */
export function labelLandmarks(scene: MvsScene): LabelLandmark[] {
  const views = new Map(scene.views.map((v) => [v.id, v]));
  const out: LabelLandmark[] = [];
  for (const l of scene.landmarks) {
    let radius = Infinity;
    for (const o of l.observations) {
      const v = views.get(o.viewId);
      if (!v) continue;
      const R = v.pose.R,
        t = v.pose.t;
      const z = R[6]! * l.xyz[0]! + R[7]! * l.xyz[1]! + R[8]! * l.xyz[2]! + t[2]!;
      if (!(z > 0)) continue;
      radius = Math.min(
        radius,
        (LABEL_LANDMARK_FOOTPRINTS * z) / ((v.camera.fx + v.camera.fy) / 2),
      );
    }
    if (Number.isFinite(radius)) out.push({ xyz: l.xyz, radius });
  }
  return out;
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
  if (d.fusion !== undefined && d.fusion !== 'pixel' && d.fusion !== 'tsdf')
    throw new RangeError("fusion must be 'pixel' or 'tsdf'");
  const tsdf = d.fusion === 'tsdf';
  if (d.planarDepth !== undefined && typeof d.planarDepth !== 'boolean')
    throw new RangeError('planarDepth must be boolean');
  if (d.planarDepth && !tsdf) throw new RangeError("planarDepth needs fusion 'tsdf'");
  if (tsdf) checkedTsdfOptions(d.tsdf);
  if (d.labeling !== undefined) {
    if (!tsdf) throw new RangeError("labeling needs fusion 'tsdf'");
    checkedLabelOptions(d.labeling);
  }
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
  let planarReport: PlanarDepthReport | undefined;
  const viewIds = plan.packed.map((r) => sized.views[r]!.id);
  const result = (
    pointsByView: readonly GpuDensePoints[],
    depthPixels: readonly number[],
    report: ShadePlanReport,
    residency?: MvsResidency,
    tsdfReport?: TsdfReport,
    labelReport?: DenseLabelReport,
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
    return {
      points,
      views,
      ...(residency ? { residency } : {}),
      ...(tsdfReport ? { tsdf: tsdfReport } : {}),
      ...(planarReport ? { planar: planarReport } : {}),
      ...(labelReport ? { labeling: labelReport } : {}),
      report,
    };
  };
  if (N === 0)
    return result([], [], { tier: device.backend, dispatches: 0, submissions: 0, groups: [] });
  const { job } = createJob(device, sized, plan, images, {
    ...options,
    confidence: tsdf || !!options.onDepthDiagnostic,
  });
  const { P, width, height } = job;
  const { scale } = plan.normalization;
  const order = Array.from({ length: N }, (_, v) => v);
  const windowOf = (r: number) =>
    Array.from(plan.fusion.subarray(plan.fusionOffsets[r]!, plan.fusionOffsets[r + 1]!));
  const fusionSlots = Math.max(...order.map((r) => windowOf(r).length));
  return device.enqueue(async () => {
    options.signal?.throwIfAborted();
    let successful = false;
    try {
      const residency = jobResidency(device, job, order, fusionSlots);
      const depthPixels: number[] = [];
      const diagnostic = async (
        r: number,
        stage: GpuDenseDepthDiagnostic['stage'],
        depth: Float32Array,
        confidence?: Float32Array,
      ) => {
        if (!options.onDepthDiagnostic) return;
        const c = plan.cameras.subarray(r * MVS_CAMERA_FLOATS, (r + 1) * MVS_CAMERA_FLOATS);
        const pose = sized.views[plan.packed[r]!]!.pose;
        await options.onDepthDiagnostic({
          viewId: viewIds[r]!,
          stage,
          width,
          height,
          depth,
          ...(confidence ? { confidence } : {}),
          depthScaleToModel: scale,
          camera: { fx: c[12]!, fy: c[13]!, cx: c[14]!, cy: c[15]! },
          sourceViewIds: sourcesOf(plan, r).map((s) => viewIds[s]!),
          pose: { R: Array.from(pose.R), t: Array.from(pose.t) },
        });
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
      // 5.3: one event after each reference of each stage.
      const progress = (stage: GpuDenseStage, r: number, done: number) =>
        options.onProgress?.({ stage, viewId: viewIds[r]!, done, total: N });
      const maps: MvsHostMaps = options.depthMaps
        ? givenMaps(options.depthMaps, N)
        : await depthStage(job, residency, {
            order,
            geometricOrder: order,
            geometricIterations: d.geometricIterations,
            filter: true,
            costs: false,
            readStages: false,
            keepPhotometric: false,
            ...(options.onDepthDiagnostic ? { onDiagnostic: diagnostic } : {}),
            onPhotometric: (r, done) => progress('photometric', r, done),
            onGeometric: (r, done) => progress('geometric', r, done),
            // Amendment 5: a preview of each reference's depth as soon as the host holds it.
            ...(options.onView
              ? {
                  photometricPreviewIterations: PREVIEW_ITERATIONS,
                  onPhotometricDepth: async (r: number, depth: Float32Array, iteration?: number) =>
                    options.onView!({
                      type: 'depth',
                      stage: 'photometric',
                      viewId: viewIds[r]!,
                      ...(iteration === undefined ? {} : { iteration }),
                      ...depthPreview(depth),
                    }),
                  onGeometricDepth: async (r: number, depth: Float32Array) =>
                    options.onView!({
                      type: 'depth',
                      stage: 'geometric',
                      viewId: viewIds[r]!,
                      ...depthPreview(depth),
                    }),
                }
              : {}),
          });
      if (options.depthMaps)
        for (let v = 0; v < N; v++) {
          if (!job.packedImages[v]!.image.observationMask) continue;
          maps.depth[v] = maps.depth[v]!.slice();
          maps.normal[v] = maps.normal[v]!.slice();
          if (maps.confidence[v]) maps.confidence[v] = maps.confidence[v]!.slice();
          excludeObservationPaths(job, v, maps.depth[v]!, maps.normal[v], maps.confidence[v], 0);
        }
      for (let v = 0; v < N; v++)
        depthPixels[v] = maps.depth[v]!.reduce((n, z) => n + (z > 0 ? 1 : 0), 0);
      if (!tsdf)
        await options.onDepthMaps?.({
          depth: maps.depth.slice() as Float32Array[],
          normal: maps.normal.slice() as Uint32Array[],
          ...(tsdf ? { confidence: maps.confidence.slice() as Float32Array[] } : {}),
        });
      if (tsdf) {
        // The TSDF fusion (section 10.4) of every filtered depth map, in the normalized frame.
        options.signal?.throwIfAborted();
        let input: OwnedPlanarInput = {
          width,
          height,
          views: order.map((v) => {
            const camera = plan.cameras.subarray(
              v * MVS_CAMERA_FLOATS,
              (v + 1) * MVS_CAMERA_FLOATS,
            );
            const view = job.packedImages[v]!;
            return {
              rotation: camera.subarray(0, 9),
              center: camera.subarray(20, 23),
              fx: camera[12]!,
              fy: camera[13]!,
              cx: camera[14]!,
              cy: camera[15]!,
              k1: view.k1,
              k2: view.k2,
              radialLimit: Math.min(1e30, radialLimitSquared(view.k1, view.k2)),
              depth: maps.depth[v]!,
              normal: maps.normal[v]!,
              ...(maps.confidence[v] ? { confidence: maps.confidence[v] } : {}),
              rgba: readOnlyRgbaWords(view.image.rgba, P),
            };
          }),
        };
        if (d.planarDepth) {
          // The private input slots now own these references. Retiring a frozen map must also
          // release this second container; supplied depthMaps' outer arrays were copied earlier.
          for (const v of order) {
            maps.depth[v] = undefined;
            maps.normal[v] = undefined;
            maps.confidence[v] = undefined;
          }
          const completed = await completeOwnedPlanarDepths(device, input, {
            sources: order.map((r) => sourcesOf(plan, r)),
            excludedPixels: order.map((r) => job.packedImages[r]!.image.observationMask?.values),
            ...(job.o.residentBytes !== undefined ? { residentBytes: job.o.residentBytes } : {}),
            signal: options.signal,
          });
          input = completed.input;
          planarReport = completed.report;
          job.reports.push(completed.report.report);
          for (const [index, v] of order.entries()) {
            maps.depth[v] = input.views[index]!.depth;
            maps.normal[v] = input.views[index]!.normal;
            maps.confidence[v] = input.views[index]!.confidence;
            depthPixels[v] = input.views[index]!.depth.reduce((n, z) => n + (z > 0 ? 1 : 0), 0);
            await diagnostic(v, 'planar', maps.depth[v]!, maps.confidence[v]);
          }
        }
        await options.onDepthMaps?.({
          depth: maps.depth.slice() as Float32Array[],
          normal: maps.normal.slice() as Uint32Array[],
          confidence: maps.confidence.slice() as Float32Array[],
        });
        const fused = await fuseTsdfGpu(device, input, {
          ...d.tsdf,
          ...(job.o.residentBytes !== undefined ? { residentBytes: job.o.residentBytes } : {}),
          ...(options.tsdfMaxBlocks !== undefined ? { maxBlocks: options.tsdfMaxBlocks } : {}),
          signal: options.signal,
          onProgress: options.onFusionProgress
            ? (event) =>
                options.onFusionProgress!({
                  ...event,
                  ...(event.view < 0 ? {} : { viewId: viewIds[order[event.view]!]! }),
                })
            : undefined,
          onPart: options.onFusionPreview
            ? async (part) => {
                const count = part.positions.length / 3;
                const colors = new Uint8Array(count * 3),
                  normals = new Int16Array(count * 2);
                for (let k = 0; k < count; k++) {
                  const normal = part.data[k * TSDF_POINT_WORDS]!,
                    color = part.data[k * TSDF_POINT_WORDS + 1]!;
                  normals[2 * k] = (normal << 16) >> 16;
                  normals[2 * k + 1] = normal >> 16;
                  for (let c = 0; c < 3; c++) colors[3 * k + c] = (color >>> (8 * c)) & 255;
                }
                await options.onFusionPreview!({
                  count,
                  positions: Float32Array.from(
                    denormalizeMvsPoints(part.positions, plan.normalization),
                  ),
                  colors,
                  normals,
                });
              }
            : undefined,
        });
        job.reports.push(fused.tsdf.report);
        const world = Float32Array.from(denormalizeMvsPoints(fused.positions, plan.normalization));
        await options.onTsdfPoints?.({ ...fused, positions: world });
        // The labeling of section 10.5 in the model's frame, on the host.
        let keep: Uint8Array | undefined;
        let labelReport: DenseLabelReport | undefined;
        if (d.labeling) {
          options.signal?.throwIfAborted();
          const labeled = labelDenseCloud(
            { ...fused, positions: world },
            labelLandmarks(sized),
            d.labeling,
          );
          keep = labeled.keep;
          labelReport = labeled.report;
        }
        // Every point belongs to the volume, none to one view: the views' slices are empty.
        const count = keep ? keep.reduce((n, x) => n + x, 0) : fused.count;
        const part: GpuDensePoints = {
          count,
          positions: new Float32Array(3 * count),
          colors: new Uint8Array(3 * count),
          normals: new Int16Array(2 * count),
        };
        for (let k = 0, at = 0; k < fused.count; k++) {
          if (keep && !keep[k]) continue;
          const normal = fused.normals[k]!,
            color = fused.colors[k]!;
          part.positions.set(world.subarray(3 * k, 3 * k + 3), 3 * at);
          part.normals[2 * at] = (normal << 16) >> 16;
          part.normals[2 * at + 1] = normal >> 16;
          for (let c = 0; c < 3; c++) part.colors[3 * at + c] = (color >>> (8 * c)) & 255;
          at++;
        }
        // 5.3: the volume's points in one event, after the fusion, then the progress of each view.
        if (options.onView)
          await options.onView({
            type: 'points',
            viewId: -1,
            count: part.count,
            positions: part.positions.slice(),
            colors: part.colors.slice(),
            normals: part.normals.slice(),
          });
        for (const r of order) progress('fusion', r, r + 1);
        successful = true;
        const out = result(
          order.map(() => ({
            count: 0,
            positions: new Float32Array(0),
            colors: new Uint8Array(0),
            normals: new Int16Array(0),
          })),
          depthPixels,
          jobReport(job),
          residency,
          { ...fused.tsdf, voxel: fused.tsdf.voxel * plan.normalization.scale },
          labelReport,
        );
        return { ...out, points: part };
      }
      // The fusion (3.10, Amendment 3): a filtered output is released after its last window.
      const readers = new Int32Array(N);
      for (const r of order) for (const v of windowOf(r)) readers[v]!++;
      const owner = ownArrays(device);
      job.owners.push(owner);
      device.reserveWorkingBytes(residency.fusion.bytes);
      const F = allocateFusionArrays(job, owner, residency.fusion);
      const cache = new SlotCache(residency.fusion.slots);
      const schedule = { device, chunkPixels: job.chunkPixels };
      const pointsByView: GpuDensePoints[] = [];
      for (const r of order) {
        const view = job.packedImages[r]!;
        const window = windowOf(r);
        F.mvsRgba.write(packRgba(view.image.rgba, P));
        const fuseBase = {
          order: MVS_CAMERA_FLOATS * residency.fusion.slots,
          minViews: d.minConsistentViews,
          maxDepthError: FUSION.maxDepthError,
          maxReprojection: FUSION.maxReprojection,
          minNormalCos: Math.cos((FUSION.maxNormalDeg * Math.PI) / 180),
          distorted: view.k1 !== 0 || view.k2 !== 0 ? 1 : 0,
          k1: view.k1,
          k2: view.k2,
          radialLimit: Math.min(1e30, radialLimitSquared(view.k1, view.k2)),
        };
        const parts: { positions: Float32Array; data: Uint32Array }[] = [];
        const readPoints = async () => {
          const count = (await device.read(F.totals, 2))[0]!;
          if (count > P) throw new Error('The dense fusion returned an invalid count');
          const [positions, data] = await readTogether([
            device.read(F.mvsPoints, 3 * count).then((a) => a.slice(0, 3 * count)),
            device.read(F.mvsPointData, 2 * count).then((a) => a.slice(0, 2 * count)),
          ]);
          parts.push({ positions, data });
        };
        if (residency.fusion.mode === 'window') {
          const slotOf = await placeFusionWindow(job, F, cache, window, window, maps);
          const control = { ...job.mvsControl, reference: slotOf.get(r)!, view: r };
          const fuseControl = { ...fuseBase, views: window.length, first: 0, end: P, start: 1 };
          const fused = {
            mvsCameras: F.mvsCameras,
            mvsGeoDepth: F.mvsGeoDepth,
            mvsNormal: F.mvsNormal,
            mvsFlags: F.mvsFlags,
          };
          // 5.1: one unit is one pixel and one fusion view, for both kernels.
          const K = window.length - 1;
          await steps(job, [
            ...chunkSteps(
              schedule,
              'mvsFuse',
              'mvsFuseFlags',
              P,
              K,
              (mvsRange, groups) =>
                mvsFuseFlags({ ...fused, mvsControl: control, fuseControl, mvsRange }, groups),
              'mvsFuseFlags',
            ),
            [
              'mvsFlagScan',
              () =>
                planFlagScan(
                  device,
                  { flags: F.mvsFlags, blockSums: F.mvsFlagOffsets, totals: F.totals },
                  P,
                ),
            ] as const,
            ...withAfter(
              chunkSteps(
                schedule,
                'mvsFuse',
                'mvsScatterPoints',
                P,
                K,
                (mvsRange, groups) =>
                  mvsScatterPoints(
                    {
                      ...fused,
                      mvsRgba: F.mvsRgba,
                      mvsFlagOffsets: F.mvsFlagOffsets,
                      mvsPoints: F.mvsPoints,
                      mvsPointData: F.mvsPointData,
                      mvsControl: control,
                      fuseControl,
                      mvsRange,
                    },
                    groups,
                  ),
                'mvsScatterPoints',
              ),
              readPoints,
            ),
          ]);
        } else {
          // Streamed: the reference stays resident, its fusion views pass through in groups in
          // ascending packed index, and each chunk of pixels sums them in `mvsFuseSums`.
          const others = window.filter((v) => v !== r);
          const groups = others.length ? sourceGroups(others, residency.fusion.slots - 1) : [[]];
          const C = residency.fusion.chunk;
          for (let first = 0; first < P; first += C) {
            const end = Math.min(P, first + C),
              n = end - first;
            let control = { ...job.mvsControl, view: r };
            for (let g = 0; g < groups.length; g++) {
              const group = groups[g]!;
              const slotOf = await placeFusionWindow(job, F, cache, [r, ...group], group, maps);
              control = { ...control, reference: slotOf.get(r)! };
              const fuseControl = {
                ...fuseBase,
                views: group.length,
                first,
                end,
                start: g === 0 ? 1 : 0,
              };
              const mvsControl = control;
              await steps(
                job,
                chunkSteps(
                  schedule,
                  'mvsFuse',
                  'mvsFuseAccumulate',
                  n,
                  group.length,
                  (mvsRange, workgroups) =>
                    mvsFuseAccumulate(
                      {
                        mvsControl,
                        mvsRange,
                        fuseControl,
                        mvsCameras: F.mvsCameras,
                        mvsGeoDepth: F.mvsGeoDepth,
                        mvsNormal: F.mvsNormal,
                        mvsFuseSums: F.mvsFuseSums!,
                      },
                      workgroups,
                    ),
                  'mvsFuseAccumulate',
                  first,
                ),
              );
            }
            const fuseControl = { ...fuseBase, views: 0, first, end, start: 0 };
            const mvsControl = control;
            // The scan reads every block of `mvsFlags` and `mvsFlagOffsets` (`scan.shade.ts`), so
            // the emit writes the whole array, 0 past the chunk, and the scan counts all of it.
            const flags = scanBlockCount(C) * LANES;
            await steps(job, [
              [
                'mvsFuseEmit',
                () =>
                  mvsFuseEmit(
                    {
                      mvsControl,
                      mvsRange: { first: 0, end: flags },
                      fuseControl,
                      mvsGeoDepth: F.mvsGeoDepth,
                      mvsFuseSums: F.mvsFuseSums!,
                      mvsFlags: F.mvsFlags,
                    },
                    flags / LANES,
                  ),
              ],
              [
                'mvsFlagScan',
                () =>
                  planFlagScan(
                    device,
                    { flags: F.mvsFlags, blockSums: F.mvsFlagOffsets, totals: F.totals },
                    flags,
                  ),
              ],
              ...withAfter(
                chunkSteps(
                  schedule,
                  'mvsFuseScatter',
                  'mvsFuseScatterSums',
                  n,
                  1,
                  (mvsRange, workgroups) =>
                    mvsFuseScatterSums(
                      {
                        mvsControl,
                        mvsRange,
                        fuseControl,
                        mvsCameras: F.mvsCameras,
                        mvsGeoDepth: F.mvsGeoDepth,
                        mvsRgba: F.mvsRgba,
                        mvsFuseSums: F.mvsFuseSums!,
                        mvsFlags: F.mvsFlags,
                        mvsFlagOffsets: F.mvsFlagOffsets,
                        mvsPoints: F.mvsPoints,
                        mvsPointData: F.mvsPointData,
                      },
                      workgroups,
                    ),
                  'mvsFuseScatterSums',
                  first,
                ),
                readPoints,
              ),
            ]);
          }
        }
        // The points of the reference, in pixel order, back to the model's frame.
        const count = parts.reduce((n, p) => n + p.positions.length / 3, 0);
        const positions = new Float32Array(3 * count),
          data = new Uint32Array(2 * count);
        let at = 0;
        for (const p of parts) {
          positions.set(p.positions, 3 * at);
          data.set(p.data, 2 * at);
          at += p.positions.length / 3;
        }
        const part = {
          count,
          positions: Float32Array.from(denormalizeMvsPoints(positions, plan.normalization)),
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
        progress('fusion', r, r + 1);
        for (const v of window)
          if (--readers[v]! === 0) {
            maps.depth[v] = undefined;
            maps.normal[v] = undefined;
          }
      }
      owner.release(true);
      successful = true;
      return result(pointsByView, depthPixels, jobReport(job), residency);
    } catch (error) {
      // A cancelled run stopped between plans, so its arrays hold no half-written state: the
      // device may keep them for a later array.
      if (options.signal?.aborted) successful = true;
      throw error;
    } finally {
      releaseJob(job, successful);
    }
  });
}

/** The steps, read until `signal` aborts: then the next read throws the signal's reason. */
function* cancellable<T>(steps: Iterable<T>, signal: AbortSignal | undefined): Generator<T> {
  for (const step of steps) {
    signal?.throwIfAborted();
    yield step;
  }
}
