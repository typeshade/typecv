/**
 * Host stages of the dense MVS (`docs/dense-mvs.md`, sections 3.1 to 3.3 and 4.1): the
 * normalization of the sparse model, the source views of each reference, the depth range and
 * the seeds of each reference, and their packing into `mvsCameras`, `mvsSources` and
 * `mvsSeeds`. Pure host code in f64. The kernels of later steps read the packed arrays.
 *
 * This module is internal until step 5, which adds `reconstructDenseGpu` and its export.
 * `src/gpu/index.ts` does not export it.
 *
 * Details that the document leaves open, with the choice made here:
 * - The map's radius (3.1) is the RMS distance of the landmarks from their centroid.
 * - A source must itself get a depth map, because the per-view arrays hold only the N views
 *   with a depth map and the geometric pass reads the sources' depth maps. Selection therefore
 *   repeats until no view loses its depth map: a view with too few landmarks is no source, and
 *   a view whose sources all drop out loses its own depth map.
 * - Equal scores are ordered by the lower view index.
 * - The percentiles of 3.3 are the sorted depths at index ⌊(n − 1) · 0.01⌋ and
 *   ⌈(n − 1) · 0.99⌉, the rule of `depthRange` in `website/src/lib/sfm-scene.ts`.
 * - A seed's pixel is the observation undistorted with the view's k1, k2 and rounded to the
 *   nearest integer, because the depth maps are on the undistorted image (3.4). A seed outside
 *   the image or outside the depth range is dropped. Two seeds on one pixel keep the nearer
 *   depth, so no two lanes of `mvsSeed` write one pixel. Seeds are ordered by pixel index.
 * - `seedOffsets` (N + 1 values) gives the seed range of each packed view in `seeds`.
 */
import { unprojectPixel, validateCamera } from '../geometry/camera.ts';

/** One registered view, the shape of `SceneView` in `website/src/lib/sfm-scene.ts`. */
export interface MvsSceneView {
  readonly id: number;
  readonly width?: number;
  readonly height?: number;
  /** World to camera, x_c = R x_w + t, R row-major. */
  readonly pose: { readonly R: ArrayLike<number>; readonly t: ArrayLike<number> };
  readonly camera: {
    readonly fx: number;
    readonly fy: number;
    readonly cx: number;
    readonly cy: number;
    readonly k1?: number;
    readonly k2?: number;
  };
}
/** One landmark, the shape of `ScenePoint` in `website/src/lib/sfm-scene.ts`. */
export interface MvsSceneLandmark {
  readonly xyz: ArrayLike<number>;
  readonly observations: readonly {
    readonly viewId: number;
    readonly pixel: readonly [number, number];
  }[];
}
export interface MvsScene {
  readonly views: readonly MvsSceneView[];
  readonly landmarks: readonly MvsSceneLandmark[];
}

export interface MvsViewOptions {
  /** The most source views of one reference (3.2). Default 6. At most 6, the `mvsSources` slots. */
  readonly maxSources?: number;
  /** A pair with fewer shared landmarks scores 0 (3.2). Default 16. */
  readonly minSharedLandmarks?: number;
  /** α0 in degrees: g(α) = min(1, (α / α0)²) (3.2). Default 5. */
  readonly fullAngleDeg?: number;
  /** αmin in degrees: g(α) = 0 below it (3.2). Default 1. */
  readonly minAngleDeg?: number;
  /** A view with fewer observed landmarks in front of it gets no depth map (3.3). Default 8. */
  readonly minLandmarks?: number;
  /** The near bound is this × the 1st percentile of the frustum depths (3.3). Default 0.75. */
  readonly nearStretch?: number;
  /** The far bound is this × the 99th percentile of the frustum depths (3.3). Default 8. */
  readonly farStretch?: number;
  /**
   * The most other views a reference fuses against (3.10, Amendment 3): those that share the most
   * landmarks with it, ties to the lower packed index. COLMAP's `check_num_images`. Default 50.
   */
  readonly fusionViews?: number;
}

/** World' = (world − center) / scale. Depths and points of the MVS are in world' units. */
export interface MvsNormalization {
  readonly center: Float64Array;
  readonly scale: number;
}

/** `ready`: the view gets a depth map. The other two are the reasons it does not (5.4). */
export type MvsViewStatus = 'ready' | 'noSource' | 'fewLandmarks';

export interface MvsViewPlanEntry {
  readonly id: number;
  readonly status: MvsViewStatus;
  /** The view's index in the packed arrays, or −1 when it gets no depth map. */
  readonly index: number;
  /** Packed indices of the source views, best first. */
  readonly sources: readonly number[];
  /** Normalized optical-axis depths, [0.75 × 1st percentile, 8 × 99th percentile] of the frustum depths. */
  readonly depthRange?: readonly [number, number];
  /** Observed landmarks in front of the view. */
  readonly landmarks: number;
}

export interface MvsViewPlan {
  readonly normalization: MvsNormalization;
  /** One entry per input view, in input order. */
  readonly views: readonly MvsViewPlanEntry[];
  /** Input view index of each packed view, in input order. Its length is N. */
  readonly packed: readonly number[];
  /**
   * `mvsCameras`: f32 × 24 per packed view: R (9), t (3), fx, fy, cx, cy, W, H, dmin, dmax, C (3),
   * and the packed index (Amendment 3: the fusion's "lower index" reads it from a slot's record).
   */
  readonly cameras: Float32Array;
  /** `mvsSources`: u32 × 7 per packed view: the source count, then up to 6 packed indices. */
  readonly sources: Uint32Array;
  /** `mvsSeeds`: f32 × 3 per seed: x, y (integer pixels of the undistorted image), depth. */
  readonly seeds: Float32Array;
  /** Seeds of packed view v are seeds[3 · seedOffsets[v] .. 3 · seedOffsets[v + 1]). */
  readonly seedOffsets: Uint32Array;
  /**
   * The fusion window of packed view v (3.10, Amendment 3): `fusion[fusionOffsets[v] ..
   * fusionOffsets[v + 1])`, the packed indices of v and its fusion views in ascending order.
   */
  readonly fusion: Uint32Array;
  readonly fusionOffsets: Uint32Array;
}

export const MVS_CAMERA_FLOATS = 24;
export const MVS_SOURCE_SLOTS = 7;
export const MVS_SEED_FLOATS = 3;

const DEFAULTS = {
  maxSources: 6,
  minSharedLandmarks: 16,
  fullAngleDeg: 5,
  minAngleDeg: 1,
  minLandmarks: 8,
  nearStretch: 0.75,
  farStretch: 8,
  fusionViews: 50,
} as const;

function resolveOptions(options: MvsViewOptions): Required<MvsViewOptions> {
  const resolved = { ...DEFAULTS, ...options };
  const { maxSources, minSharedLandmarks, fullAngleDeg, minAngleDeg, minLandmarks } = resolved;
  if (!(resolved.nearStretch > 0 && resolved.nearStretch <= 1))
    throw new RangeError('nearStretch must lie in (0, 1]');
  if (!(resolved.farStretch >= 1 && Number.isFinite(resolved.farStretch)))
    throw new RangeError('farStretch must be a finite number of at least 1');
  if (!Number.isSafeInteger(maxSources) || maxSources < 1 || maxSources > MVS_SOURCE_SLOTS - 1)
    throw new RangeError('maxSources must be an integer in [1, 6]');
  if (!Number.isSafeInteger(minSharedLandmarks) || minSharedLandmarks < 1)
    throw new RangeError('minSharedLandmarks must be a positive integer');
  if (!Number.isSafeInteger(minLandmarks) || minLandmarks < 1)
    throw new RangeError('minLandmarks must be a positive integer');
  if (!Number.isSafeInteger(resolved.fusionViews) || resolved.fusionViews < 1)
    throw new RangeError('fusionViews must be a positive integer');
  if (!(fullAngleDeg > 0 && fullAngleDeg < 180 && minAngleDeg >= 0 && minAngleDeg < 180))
    throw new RangeError('The angles must lie in (0, 180) degrees');
  return resolved;
}

function validateScene(scene: MvsScene): void {
  const ids = new Set<number>();
  for (const view of scene.views) {
    if (ids.has(view.id)) throw new RangeError('Duplicate view id ' + view.id);
    ids.add(view.id);
    validateCamera(view.camera);
    if (view.pose.R.length !== 9 || view.pose.t.length !== 3)
      throw new RangeError('A pose needs R (9) and t (3)');
    for (let i = 0; i < 9; i++)
      if (!Number.isFinite(view.pose.R[i])) throw new RangeError('A pose must be finite');
    for (let i = 0; i < 3; i++)
      if (!Number.isFinite(view.pose.t[i])) throw new RangeError('A pose must be finite');
  }
  for (const landmark of scene.landmarks)
    for (let i = 0; i < 3; i++)
      if (!Number.isFinite(landmark.xyz[i])) throw new RangeError('A landmark must be finite');
}

/** The camera center −Rᵀ t in the world frame. */
export function mvsCameraCenter(view: MvsSceneView): Float64Array {
  const { R, t } = view.pose;
  return Float64Array.from(
    { length: 3 },
    (_, i) => -(R[i]! * t[0]! + R[i + 3]! * t[1]! + R[i + 6]! * t[2]!),
  );
}

/** Optical-axis depth z = row 3 of (R X + t). */
export function mvsDepth(view: MvsSceneView, xyz: ArrayLike<number>): number {
  const { R, t } = view.pose;
  return R[6]! * xyz[0]! + R[7]! * xyz[1]! + R[8]! * xyz[2]! + t[2]!;
}

/** 3.1: the landmarks' centroid and RMS radius. An empty or a one-point map has scale 1. */
export function mvsNormalization(scene: MvsScene): MvsNormalization {
  const center = new Float64Array(3);
  const n = scene.landmarks.length;
  if (!n) return { center, scale: 1 };
  for (const { xyz } of scene.landmarks) for (let i = 0; i < 3; i++) center[i]! += xyz[i]!;
  for (let i = 0; i < 3; i++) center[i]! /= n;
  let sum = 0;
  for (const { xyz } of scene.landmarks)
    sum += (xyz[0]! - center[0]!) ** 2 + (xyz[1]! - center[1]!) ** 2 + (xyz[2]! - center[2]!) ** 2;
  const scale = Math.sqrt(sum / n);
  return { center, scale: scale > 0 && Number.isFinite(scale) ? scale : 1 };
}

/**
 * The scene in world' units. A landmark becomes (X − center) / scale. A pose keeps R, and t
 * becomes (R · center + t) / scale, so that the camera frame is scaled by 1 / scale and pixels
 * do not change.
 */
export function normalizeMvsScene(scene: MvsScene, normalization: MvsNormalization): MvsScene {
  const { center: c, scale: s } = normalization;
  return {
    views: scene.views.map((view) => {
      const { R, t } = view.pose;
      return {
        ...view,
        pose: {
          R: Float64Array.from({ length: 9 }, (_, i) => R[i]!),
          t: Float64Array.from(
            { length: 3 },
            (_, i) =>
              (R[3 * i]! * c[0]! + R[3 * i + 1]! * c[1]! + R[3 * i + 2]! * c[2]! + t[i]!) / s,
          ),
        },
      };
    }),
    landmarks: scene.landmarks.map((landmark) => ({
      ...landmark,
      xyz: Float64Array.from({ length: 3 }, (_, i) => (landmark.xyz[i]! - c[i]!) / s),
    })),
  };
}

/** Maps packed world' points (x, y, z per point) back to the model's frame, X = scale · X' + center. */
export function denormalizeMvsPoints(
  points: ArrayLike<number>,
  normalization: MvsNormalization,
): Float64Array {
  if (points.length % 3) throw new RangeError('Points are x, y, z triples');
  const { center: c, scale: s } = normalization;
  return Float64Array.from({ length: points.length }, (_, i) => points[i]! * s + c[i % 3]!);
}

/** g(α) of 3.2, with α in degrees. */
export function mvsAngleWeight(
  alphaDeg: number,
  fullAngleDeg: number = DEFAULTS.fullAngleDeg,
  minAngleDeg: number = DEFAULTS.minAngleDeg,
): number {
  if (!(alphaDeg >= minAngleDeg)) return 0;
  return Math.min(1, (alphaDeg / fullAngleDeg) ** 2);
}

function angleDeg(p: ArrayLike<number>, a: Float64Array, b: Float64Array): number {
  const u = [a[0]! - p[0]!, a[1]! - p[1]!, a[2]! - p[2]!],
    v = [b[0]! - p[0]!, b[1]! - p[1]!, b[2]! - p[2]!];
  const cross = Math.hypot(
    u[1]! * v[2]! - u[2]! * v[1]!,
    u[2]! * v[0]! - u[0]! * v[2]!,
    u[0]! * v[1]! - u[1]! * v[0]!,
  );
  return (Math.atan2(cross, u[0]! * v[0]! + u[1]! * v[1]! + u[2]! * v[2]!) * 180) / Math.PI;
}

/**
 * 3.2: score(r, s) for every pair of input views, a symmetric V × V matrix, row-major, with a
 * zero diagonal. A landmark counts for a pair when both views observe it, once per view even
 * when a track repeats a view. A pair with fewer than `minSharedLandmarks` shared landmarks
 * scores 0.
 */
export function mvsViewScores(scene: MvsScene, options: MvsViewOptions = {}): Float64Array {
  return viewPairs(scene, options).scores;
}

/** The scores of 3.2 and the shared landmark count of every pair of input views, V × V each. */
function viewPairs(scene: MvsScene, options: MvsViewOptions) {
  const o = resolveOptions(options);
  validateScene(scene);
  const V = scene.views.length;
  const index = new Map(scene.views.map((view, i) => [view.id, i]));
  const centers = scene.views.map(mvsCameraCenter);
  const scores = new Float64Array(V * V),
    shared = new Uint32Array(V * V);
  for (const landmark of scene.landmarks) {
    const seen = [
      ...new Set(
        landmark.observations
          .map((observation) => index.get(observation.viewId))
          .filter((i) => i !== undefined),
      ),
    ].sort((a, b) => a - b);
    for (let a = 0; a < seen.length; a++)
      for (let b = a + 1; b < seen.length; b++) {
        const r = seen[a]!,
          s = seen[b]!;
        const g = mvsAngleWeight(
          angleDeg(landmark.xyz, centers[r]!, centers[s]!),
          o.fullAngleDeg,
          o.minAngleDeg,
        );
        shared[r * V + s]!++;
        scores[r * V + s]! += g;
      }
  }
  for (let r = 0; r < V; r++)
    for (let s = r + 1; s < V; s++) {
      const value = shared[r * V + s]! >= o.minSharedLandmarks ? scores[r * V + s]! : 0;
      scores[r * V + s] = value;
      scores[s * V + r] = value;
      shared[s * V + r] = shared[r * V + s]!;
    }
  return { scores, shared };
}

/**
 * 3.3: the normalized depths of the landmarks that the view observes, in front of the view, in
 * ascending order. Pass the normalized scene for world' depths.
 */
export function mvsObservedDepths(scene: MvsScene, viewId: number): Float64Array {
  const view = scene.views.find((candidate) => candidate.id === viewId);
  if (!view) throw new RangeError('Unknown view id ' + viewId);
  const depths: number[] = [];
  for (const landmark of scene.landmarks) {
    if (!landmark.observations.some((observation) => observation.viewId === viewId)) continue;
    const z = mvsDepth(view, landmark.xyz);
    if (z > 0 && Number.isFinite(z)) depths.push(z);
  }
  return Float64Array.from(depths).sort();
}

/**
 * 3.3 (Amendment 2): the normalized depths of all landmarks in the view's frustum, in ascending
 * order: in front of the view and projecting into its image, whether the view observes them or
 * not. The landmarks that the view observes are always in the list.
 */
export function mvsFrustumDepths(scene: MvsScene, viewId: number): Float64Array {
  const view = scene.views.find((candidate) => candidate.id === viewId);
  if (!view) throw new RangeError('Unknown view id ' + viewId);
  const { width, height } = viewSize(view);
  const { fx, fy, cx, cy } = view.camera;
  const { R, t } = view.pose;
  const depths: number[] = [];
  for (const landmark of scene.landmarks) {
    const x = landmark.xyz[0]!,
      y = landmark.xyz[1]!,
      w = landmark.xyz[2]!;
    const z = R[6]! * x + R[7]! * y + R[8]! * w + t[2]!;
    if (!(z > 0 && Number.isFinite(z))) continue;
    const observed = landmark.observations.some((observation) => observation.viewId === viewId);
    if (!observed) {
      const u = fx * ((R[0]! * x + R[1]! * y + R[2]! * w + t[0]!) / z) + cx,
        v = fy * ((R[3]! * x + R[4]! * y + R[5]! * w + t[1]!) / z) + cy;
      if (!(u >= -0.5 && u <= width - 0.5 && v >= -0.5 && v <= height - 0.5)) continue;
    }
    depths.push(z);
  }
  return Float64Array.from(depths).sort();
}

/**
 * 3.3: [nearStretch × the 1st percentile, farStretch × the 99th percentile] of ascending depths.
 * The defaults are 0.75 and 8.
 */
export function mvsDepthRange(
  sortedDepths: ArrayLike<number>,
  nearStretch: number = DEFAULTS.nearStretch,
  farStretch: number = DEFAULTS.farStretch,
): [number, number] | undefined {
  const n = sortedDepths.length;
  if (!n) return undefined;
  return [
    nearStretch * sortedDepths[Math.floor((n - 1) * 0.01)]!,
    farStretch * sortedDepths[Math.ceil((n - 1) * 0.99)]!,
  ];
}

/**
 * 3.3: the seeds (x, y, depth) of one view, for depths in `range`. Pass the normalized scene.
 * See the module comment for the rounding, the bounds and the rule for two seeds on one pixel.
 */
export function mvsSeeds(
  scene: MvsScene,
  viewId: number,
  range: readonly [number, number],
): Float64Array {
  const view = scene.views.find((candidate) => candidate.id === viewId);
  if (!view) throw new RangeError('Unknown view id ' + viewId);
  const { width, height } = viewSize(view);
  const camera = view.camera;
  const best = new Map<number, number>();
  for (const landmark of scene.landmarks) {
    const z = mvsDepth(view, landmark.xyz);
    if (!(z >= range[0] && z <= range[1])) continue;
    for (const observation of landmark.observations) {
      if (observation.viewId !== viewId) continue;
      const xy = unprojectPixel(camera, observation.pixel);
      if (!xy) continue;
      const x = Math.round(camera.fx * xy[0] + camera.cx),
        y = Math.round(camera.fy * xy[1] + camera.cy);
      if (!(x >= 0 && x < width && y >= 0 && y < height)) continue;
      const pixel = y * width + x,
        previous = best.get(pixel);
      if (previous === undefined || z < previous) best.set(pixel, z);
    }
  }
  const pixels = [...best.keys()].sort((a, b) => a - b);
  const seeds = new Float64Array(pixels.length * MVS_SEED_FLOATS);
  pixels.forEach((pixel, k) => {
    seeds[3 * k] = pixel % width;
    seeds[3 * k + 1] = Math.floor(pixel / width);
    seeds[3 * k + 2] = best.get(pixel)!;
  });
  return seeds;
}

function viewSize(view: MvsSceneView): { width: number; height: number } {
  const { width, height } = view;
  if (!(Number.isSafeInteger(width) && Number.isSafeInteger(height) && width! > 0 && height! > 0))
    throw new RangeError('View ' + view.id + ' needs a positive integer width and height');
  return { width: width!, height: height! };
}

/** The best `maxSources` candidates with a positive score, best first, ties to the lower index. */
function bestSources(
  scores: Float64Array,
  V: number,
  r: number,
  candidates: readonly boolean[],
  maxSources: number,
): number[] {
  const list: number[] = [];
  for (let s = 0; s < V; s++) if (s !== r && candidates[s] && scores[r * V + s]! > 0) list.push(s);
  list.sort((a, b) => scores[r * V + b]! - scores[r * V + a]! || a - b);
  return list.slice(0, maxSources);
}

/**
 * 3.1 to 3.3 and the packing of 4.1: normalizes the scene, selects the source views, computes
 * the depth range and the seeds of each reference, and packs `mvsCameras`, `mvsSources` and
 * `mvsSeeds`. Every view needs `width` and `height`, the size of its analysis image.
 */
export function planMvsViews(scene: MvsScene, options: MvsViewOptions = {}): MvsViewPlan {
  const o = resolveOptions(options);
  validateScene(scene);
  for (const view of scene.views) viewSize(view);
  const normalization = mvsNormalization(scene);
  const normalized = normalizeMvsScene(scene, normalization);
  const V = scene.views.length;
  const { scores, shared } = viewPairs(normalized, o);
  const observed = normalized.views.map((view) => mvsObservedDepths(normalized, view.id));
  const depths = normalized.views.map((view) => mvsFrustumDepths(normalized, view.id));
  const enoughLandmarks = observed.map((list) => list.length >= o.minLandmarks);
  const ready = enoughLandmarks.slice();
  let sources: number[][] = [];
  for (;;) {
    sources = Array.from({ length: V }, (_, r) =>
      ready[r] ? bestSources(scores, V, r, ready, o.maxSources) : [],
    );
    let changed = false;
    for (let r = 0; r < V; r++)
      if (ready[r] && !sources[r]!.length) {
        ready[r] = false;
        changed = true;
      }
    if (!changed) break;
  }
  const packed: number[] = [];
  const packedIndex = new Int32Array(V).fill(-1);
  for (let r = 0; r < V; r++)
    if (ready[r]) {
      packedIndex[r] = packed.length;
      packed.push(r);
    }
  const N = packed.length;
  const cameras = new Float32Array(N * MVS_CAMERA_FLOATS),
    sourceSlots = new Uint32Array(N * MVS_SOURCE_SLOTS),
    seedOffsets = new Uint32Array(N + 1),
    seedLists: Float64Array[] = [];
  const views: MvsViewPlanEntry[] = normalized.views.map((view, r) => {
    const range = mvsDepthRange(depths[r]!, o.nearStretch, o.farStretch);
    const status: MvsViewStatus = ready[r]
      ? 'ready'
      : enoughLandmarks[r]
        ? 'noSource'
        : 'fewLandmarks';
    const list = sources[r]!.map((s) => packedIndex[s]!);
    return {
      id: view.id,
      status,
      index: packedIndex[r]!,
      sources: list,
      depthRange: range,
      landmarks: observed[r]!.length,
    };
  });
  packed.forEach((r, v) => {
    const view = normalized.views[r]!,
      entry = views[r]!,
      range = entry.depthRange!,
      { width, height } = viewSize(view),
      base = v * MVS_CAMERA_FLOATS;
    for (let i = 0; i < 9; i++) cameras[base + i] = view.pose.R[i]!;
    for (let i = 0; i < 3; i++) cameras[base + 9 + i] = view.pose.t[i]!;
    const { fx, fy, cx, cy } = view.camera;
    cameras.set([fx, fy, cx, cy, width, height, range[0], range[1]], base + 12);
    cameras.set(mvsCameraCenter(view), base + 20);
    cameras[base + 23] = v;
    sourceSlots[v * MVS_SOURCE_SLOTS] = entry.sources.length;
    sourceSlots.set(entry.sources, v * MVS_SOURCE_SLOTS + 1);
    const seeds = mvsSeeds(normalized, view.id, range);
    seedLists.push(seeds);
    seedOffsets[v + 1] = seedOffsets[v]! + seeds.length / MVS_SEED_FLOATS;
  });
  const seeds = new Float32Array(seedOffsets[N]! * MVS_SEED_FLOATS);
  seedLists.forEach((list, v) => seeds.set(list, seedOffsets[v]! * MVS_SEED_FLOATS));
  // Amendment 3: each reference's fusion views, by shared landmarks, then ascending.
  const fusionOffsets = new Uint32Array(N + 1),
    fusionLists: number[][] = [];
  for (let v = 0; v < N; v++) {
    const r = packed[v]!;
    const others = Array.from({ length: N }, (_, w) => w).filter((w) => w !== v);
    others.sort((a, b) => shared[r * V + packed[b]!]! - shared[r * V + packed[a]!]! || a - b);
    const list = [v, ...others.slice(0, o.fusionViews)].sort((a, b) => a - b);
    fusionLists.push(list);
    fusionOffsets[v + 1] = fusionOffsets[v]! + list.length;
  }
  const fusion = new Uint32Array(fusionOffsets[N]!);
  fusionLists.forEach((list, v) => fusion.set(list, fusionOffsets[v]!));
  return {
    fusion,
    fusionOffsets,
    normalization,
    views,
    packed,
    cameras,
    sources: sourceSlots,
    seeds,
    seedOffsets,
  };
}
