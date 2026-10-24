import { isInverseDepthTracker } from '../reconstruction/inverse-depth-tracker.ts';
import {
  depthTrackingInput,
  publishDepth,
  recordDeferredDepth,
} from '../reconstruction/depth-publication.ts';
import { bundleAdjustInverseDepthGpu } from './inverse-depth.ts';
import {
  incrementalSteps,
  ReconstructionOperationError,
  type BundleSchedule,
  ReconstructionInitializationError,
  copyReconstruction,
  type IncrementalOptions,
  type IncrementalResult,
  type ReconstructionJob,
} from '../reconstruction/mapper.ts';
import type { ImageFeatures, PairMatches, SparseReconstruction } from '../reconstruction/types.ts';
import { MapHistory } from '../reconstruction/map-history.ts';
import {
  copyTracking,
  rotationTrackingControls,
  type ReconstructionTracking,
} from '../reconstruction/tracking.ts';
import { cameraCenter } from '../geometry/pose.ts';
import { GeometryEstimationError } from '../geometry/errors.ts';
import { ShadeDevice } from './device.ts';
import { estimatePosePnPGpu, type GpuGeometryExecution } from './pnp.ts';
import { triangulateTracksGpu } from './triangulation.ts';
import {
  snapshotCameraMasks,
  copyCameraSelection,
  selectCameraObservations,
  type CameraObservationMask,
  type CameraObservationSelection,
} from '../reconstruction/observation-selection.ts';
import { evaluateObservationsGpu } from './observations.ts';
import { bundleAdjustGpu } from './bundle.ts';
import type { GpuConsensusExecution } from './two-view.ts';
import { verifyImagePairGpu } from './image-pair.ts';
import { validateFeatures } from '../reconstruction/tracks.ts';
import {
  applyFocal,
  focalSearchControls,
  type FocalSearchResult,
} from '../reconstruction/focal-search.ts';
import {
  matchFloatDescriptorsGpu,
  type GpuResidentFloatDescriptors,
  type GpuFloatMatchOptions,
  type GpuFloatMatches,
} from './float-matching.ts';

type GpuMapperOperation = 'pairVerification' | 'pnp' | 'triangulation' | 'observations' | 'bundle';
export interface GpuIncrementalOptions extends IncrementalOptions {
  /** Local optimization window, 8..64 cameras. Historical output is retained outside this window. */
  readonly windowSize?: number;
  /** Every executed numerical stage, including work before a stable seed is found. */
  readonly onStage?: (stage: GpuIncrementalResult['execution']['stages'][number]) => void;
  /** Cache small shared-focal Schur systems; the default keeps the existing solver route. */
  readonly schurCache?: boolean;
  /**
   * Hypotheses in one batch of the PnP search. The result does not depend on it
   * (`tests/typeshade/pnp-batch.test.mjs`). Every dispatch checks the arrays it binds, so a
   * smaller batch makes each dispatch cheaper; the default is the device's.
   */
  readonly pnpBatchSize?: number;
  readonly onProgress?: (event: {
    operation: ReconstructionJob['kind'];
    index: number;
    phase: 'start' | 'complete';
  }) => void;
}

export interface GpuIncrementalResult extends IncrementalResult {
  readonly execution: {
    readonly backend: 'webgpu' | 'cpu';
    readonly stages: readonly {
      readonly operation: GpuMapperOperation;
      /** Wall time including first-use pipeline compilation and numerical readback. */
      readonly elapsedMs: number;
      readonly execution: GpuGeometryExecution;
      /** Actual bundle input size, including fixed boundary views and retained observations. */
      readonly problem?: {
        readonly views: number;
        readonly variableViewIds?: readonly number[];
        readonly landmarks: number;
        readonly observations: number;
        readonly directionalConstraints?: number;
        readonly directionalObservations?: number;
      };
    }[];
    readonly pairConsensus: readonly GpuConsensusExecution[];
    readonly gpuBundleAdjustments: number;
    readonly cpuStages: readonly string[];
  };
}

/** Keypoint coordinates use the camera's pixel coordinate system and descriptor row order. */
export interface GpuFloatImageFeatures extends ImageFeatures {
  readonly descriptors: GpuResidentFloatDescriptors;
}

export interface GpuFloatImagePair {
  readonly first: number;
  readonly second: number;
}

export interface GpuFloatIncrementalOptions extends GpuIncrementalOptions {
  readonly matching?: GpuFloatMatchOptions;
  readonly onMatchProgress?: (event: {
    readonly first: number;
    readonly second: number;
    readonly index: number;
    readonly total: number;
    readonly phase: 'start' | 'complete';
  }) => void;
}

export interface GpuFloatIncrementalResult extends GpuIncrementalResult {
  readonly matching: {
    /** Matching wall time; excludes extraction and subsequent mapper operations. */
    readonly elapsedMs: number;
    readonly pairs: readonly {
      readonly first: number;
      readonly second: number;
      readonly matches: number;
      readonly report: GpuFloatMatches['report'];
      readonly execution: GpuFloatMatches['execution'];
    }[];
  };
}

/**
 * Opt-in float-descriptor adapter for externally extracted features (for example ALIKED).
 * Borrows descriptors from the same TypeShade runtime/device until this promise settles:
 * callers must not mutate, recycle or dispose them meanwhile. Descriptors are neither read
 * to the host nor disposed here. Host coordinates, cameras and controls are snapshotted
 * before matching starts. Explicit image pairs avoid an implicit quadratic view graph.
 * All resulting edges undergo the existing mapper's geometric verification and filtering.
 */
export async function reconstructIncrementalFloatGpu(
  device: ShadeDevice,
  images: readonly GpuFloatImageFeatures[],
  pairs: readonly GpuFloatImagePair[],
  options: GpuFloatIncrementalOptions = {},
): Promise<GpuFloatIncrementalResult> {
  const views = images.map((image) => ({
    id: image.id,
    camera: { ...image.camera },
    keypoints: image.keypoints.slice(),
    descriptors: { ...image.descriptors },
  }));
  const edges = pairs.map((pair) => ({ first: pair.first, second: pair.second }));
  const controls = {
    ...options,
    onReconstruction: options.depthTracking ? undefined : options.onReconstruction,
    cameraObservationMasks: snapshotCameraMasks(options.cameraObservationMasks),
    matching: options.matching ? { ...options.matching } : undefined,
    initialPair: options.initialPair ? ([...options.initialPair] as const) : undefined,
    rotationTracking:
      rotationTrackingControls(options.rotationTracking, !options.focalSearch) ?? false,
    // The pre-verified rows are read after the driver suspends: keep a private copy.
    preVerified: options.preVerified?.map((e) => ({
      ...e,
      verified: {
        ...e.verified,
        matches: e.verified.matches.slice(),
        distances: e.verified.distances?.slice(),
      },
    })),
  };
  const byId = validateFeatures(views, []);
  if (views.length < 2) throw new RangeError('At least two float-feature views are required');
  for (const view of views) {
    const { count, dimensions, elements } = view.descriptors;
    if (
      !Number.isSafeInteger(count) ||
      count < 0 ||
      count !== view.keypoints.length / 2 ||
      !Number.isSafeInteger(dimensions) ||
      dimensions < 1 ||
      !Number.isSafeInteger(elements) ||
      elements !== count * dimensions ||
      !view.descriptors.descriptors ||
      typeof view.descriptors.descriptors.read !== 'function' ||
      typeof view.descriptors.descriptors.write !== 'function'
    )
      throw new RangeError('Float descriptor rows must correspond to keypoints');
  }
  const features = new Map(views.map((view) => [view.id, view]));
  const seen = new Set<string>();
  for (const pair of edges) {
    if (!byId.has(pair.first) || !byId.has(pair.second) || pair.first === pair.second)
      throw new RangeError('Invalid float-feature image pair');
    const key = [pair.first, pair.second].sort((a, b) => a - b).join(':');
    if (seen.has(key)) throw new RangeError('Duplicate float-feature image pair');
    seen.add(key);
    const first = features.get(pair.first)!,
      second = features.get(pair.second)!;
    if (first.descriptors.dimensions !== second.descriptors.dimensions)
      throw new RangeError('Float descriptor dimensions must agree within each pair');
  }
  const matched: PairMatches[] = [],
    diagnostics: GpuFloatIncrementalResult['matching']['pairs'][number][] = [];
  const started = performance.now();
  for (let index = 0; index < edges.length; index++) {
    const pair = edges[index]!,
      event = { ...pair, index: index + 1, total: edges.length };
    controls.onMatchProgress?.({ ...event, phase: 'start' });
    const result = await matchFloatDescriptorsGpu(
      device,
      features.get(pair.first)!.descriptors,
      features.get(pair.second)!.descriptors,
      controls.matching,
    );
    matched.push({ ...pair, matches: result.matches, distances: result.distances });
    diagnostics.push({
      ...pair,
      matches: result.matches.length / 2,
      report: result.report,
      execution: result.execution,
    });
    controls.onMatchProgress?.({ ...event, phase: 'complete' });
  }
  const elapsedMs = performance.now() - started;
  const result = await reconstructIncrementalGpu(device, views, matched, controls);
  return { ...result, matching: { elapsedMs, pairs: diagnostics } };
}

/**
 * Incremental mapper with TypeShade numerical geometry throughout initialization,
 * registration, triangulation, filtering and joint BA. Graph/control remains host-side.
 * No device failure is converted to a CPU retry or an ordinary rejected view.
 */
export async function reconstructIncrementalGpu(
  device: ShadeDevice,
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  options: GpuIncrementalOptions = {},
): Promise<GpuIncrementalResult> {
  return runIncrementalGpu(device, images, pairs, options);
}

/** Cache key of a verification request: a 64-bit FNV-1a digest of the pixels and cameras, plus the controls without the seed. */
function verificationKey(
  pixels: Float64Array,
  first: ImageFeatures['camera'],
  second: ImageFeatures['camera'],
  controls: object | undefined,
): string {
  const words = new Uint32Array(pixels.buffer, pixels.byteOffset, pixels.length * 2);
  let a = 0x811c9dc5,
    b = 0x01000193;
  for (let i = 0; i < words.length; i++) {
    const w = words[i]!;
    a = Math.imul(a ^ (w & 0xffff), 0x01000193) >>> 0;
    b = Math.imul(b ^ (w >>> 16), 0x01000193) >>> 0;
  }
  return `${pixels.length}:${a.toString(16)}:${b.toString(16)}:${JSON.stringify([
    first,
    second,
    { ...controls, seed: 0 },
  ])}`;
}

type SessionState = {
  model?: SparseReconstruction;
  verification: Map<string, Awaited<ReturnType<typeof verifyImagePairGpu>>>;
  /** The map size at the last global BA, across the session's runs. */
  schedule: BundleSchedule;
};

async function runIncrementalGpu(
  device: ShadeDevice,
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  options: GpuIncrementalOptions,
  session?: SessionState,
  finalize = true,
): Promise<GpuIncrementalResult> {
  if (options.schurCache !== undefined && typeof options.schurCache !== 'boolean')
    throw new RangeError('schurCache must be a boolean');
  if (options.depthTracking !== undefined && !isInverseDepthTracker(options.depthTracking))
    throw new RangeError('Invalid depth tracker');
  // The driver can suspend between operations, so capture all mutable caller data first.
  const views = images.map((v) => ({
    ...v,
    camera: { ...v.camera },
    keypoints: v.keypoints.slice(),
  }));
  const edges = pairs.map((p) => ({
    ...p,
    matches: p.matches.slice(),
    distances: p.distances?.slice(),
  }));
  const controls = {
    ...options,
    cameraObservationMasks: snapshotCameraMasks(options.cameraObservationMasks),
    initialPair: options.initialPair ? ([...options.initialPair] as const) : undefined,
    rotationTracking:
      rotationTrackingControls(options.rotationTracking, !options.focalSearch) ?? false,
  };
  const stages: GpuIncrementalResult['execution']['stages'][number][] = [],
    pairConsensus: GpuConsensusExecution[] = [];
  let gpuBundleAdjustments = 0;
  let operationIndex = 0;
  const execute = async (job: ReconstructionJob): Promise<unknown> => {
    const started = performance.now();
    switch (job.kind) {
      case 'verifyPair': {
        // Session inputs are immutable. Use stable per-edge seeds as new edges arrive.
        const key = session
          ? verificationKey(job.args[0], job.args[1], job.args[2], job.args[3])
          : '';
        const cached = session?.verification.get(key);
        if (cached) return cached;
        const args = session
          ? ([
              job.args[0],
              job.args[1],
              job.args[2],
              {
                ...job.args[3],
                seed: (options.seed ?? 17) + session.verification.size,
              },
            ] as const)
          : job.args;
        const result = await verifyImagePairGpu(device, args[0], args[1], args[2], args[3]);
        session?.verification.set(key, result);
        stages.push({
          operation: 'pairVerification',
          elapsedMs: performance.now() - started,
          execution: result.execution,
        });
        if (result.geometry) pairConsensus.push(result.execution);
        return result;
      }
      case 'pnp': {
        const [world, pixels, camera, pnpOptions] = job.args;
        const result = await estimatePosePnPGpu(device, world, pixels, camera, {
          ...(options.pnpBatchSize === undefined ? {} : { batchSize: options.pnpBatchSize }),
          ...pnpOptions,
        });
        stages.push({
          operation: 'pnp',
          elapsedMs: performance.now() - started,
          execution: result.execution,
        });
        return result;
      }
      case 'triangulation': {
        const result = await triangulateTracksGpu(device, job.observations, job.options);
        stages.push({
          operation: 'triangulation',
          elapsedMs: performance.now() - started,
          execution: result.execution,
        });
        return result.tracks.map((track, i) => ({
          status: result.status[i],
          ...(track ? { track } : {}),
        }));
      }
      case 'observations': {
        const result = await evaluateObservationsGpu(device, job.model, job.maxError, job.depth);
        stages.push({
          operation: 'observations',
          elapsedMs: performance.now() - started,
          execution: result.execution,
        });
        return result;
      }
      case 'bundle': {
        const result = await bundleAdjustGpu(device, job.args[0], {
          ...job.args[1],
          schurCache: controls.schurCache,
        });
        gpuBundleAdjustments++;
        stages.push({
          operation: 'bundle',
          elapsedMs: performance.now() - started,
          execution: result.execution,
          problem: {
            views: job.args[0].views.length,
            variableViewIds: job.args[1]?.variableViewIds?.slice(),
            landmarks: job.args[0].landmarks.length,
            observations:
              job.args[0].landmarks.reduce(
                (sum, landmark) => sum + landmark.observations.length,
                0,
              ) + (result.diagnostics.directionalObservations ?? 0),
            ...(result.directionalConstraints
              ? {
                  directionalConstraints: result.directionalConstraints.length,
                  directionalObservations: result.diagnostics.directionalObservations,
                }
              : {}),
          },
        });
        return result;
      }
    }
  };
  try {
    const machine = incrementalSteps(
      views,
      edges,
      controls,
      session?.model,
      session?.schedule,
      finalize,
      finalize ? 2 : (focalSearchControls(controls.focalSearch)?.views ?? 2),
    );
    let step = machine.next();
    while (!step.done) {
      let value: unknown;
      const event = { operation: step.value.kind, index: ++operationIndex };
      try {
        options.onProgress?.({ ...event, phase: 'start' });
        const beforeStages = stages.length;
        value = await execute(step.value);
        if (stages.length > beforeStages) options.onStage?.(stages[stages.length - 1]!);
        options.onProgress?.({ ...event, phase: 'complete' });
      } catch (error) {
        step = machine.throw(
          error instanceof GeometryEstimationError
            ? error
            : new ReconstructionOperationError(error),
        );
        continue;
      }
      step = machine.next(value);
    }
    const result: GpuIncrementalResult = {
      ...step.value,
      execution: {
        backend: device.backend,
        stages,
        pairConsensus,
        gpuBundleAdjustments,
        cpuStages: [
          'track graph, result indexing, data packing and mapper control',
          ...(controls.rotationTracking
            ? ['calibrated rotation hypotheses (f64 CPU, no extra GPU readback)']
            : []),
          ...(stages.some((s) => s.problem?.directionalConstraints)
            ? ['unresolved bearing seed/selection (f64 CPU, no extra GPU readback)']
            : []),
        ],
      },
    };
    if (!options.depthTracking) return result;
    const updated = await options.depthTracking.updateAsync(
      depthTrackingInput(result, views, controls.cameraObservationMasks),
      (m, c, o) => bundleAdjustInverseDepthGpu(device, m, c, o),
    );
    const published = publishDepth(result, updated);
    options.onReconstruction?.(published.reconstruction);
    return {
      ...published,
      execution: {
        ...published.execution,
        cpuStages: [
          ...published.execution.cpuStages,
          'inverse-depth raw-track fitting, host f64 Schur solve and undamped marginal covariance; measurement Jacobians on selected device',
        ],
      },
    };
  } catch (error) {
    if (options.depthTracking && error instanceof ReconstructionInitializationError)
      recordDeferredDepth(options.depthTracking, error, views, controls.cameraObservationMasks);
    throw error;
  }
}

/** Sliding local map with complete historical camera/point output.
 * Track indexing and triangulation are rebuilt within the local window; this is not a SLAM loop.
 * The caller owns the device. Calls must be awaited. Local GPU work uses at most 64 views.
 */
export class IncrementalSfmSession {
  private images: ImageFeatures[] = [];
  private pairs: PairMatches[] = [];
  private state: SessionState = { verification: new Map(), schedule: { views: 0, points: 0 } };
  private result?: GpuIncrementalResult;
  private busy = false;
  private options: GpuIncrementalOptions;
  private history = new MapHistory();
  private seen = new Set<number>();
  private unregistered = new Map<number, string>();
  /** The focal search's result once it ran (`focalSearch`); every view then takes its ratio. */
  private focal?: FocalSearchResult;
  private trackingState?: ReconstructionTracking;
  private selectionState?: CameraObservationSelection;

  constructor(
    private device: ShadeDevice,
    options: GpuIncrementalOptions = {},
  ) {
    this.options = {
      ...options,
      cameraObservationMasks: snapshotCameraMasks(options.cameraObservationMasks),
      initialPair: options.initialPair?.slice() as typeof options.initialPair,
      rotationTracking:
        rotationTrackingControls(options.rotationTracking, !options.focalSearch) ?? false,
    };
    const size = options.windowSize ?? 32;
    if (!Number.isInteger(size) || size < 8 || size > 64)
      throw new RangeError('windowSize must be an integer from 8 through 64');
    focalSearchControls(options.focalSearch);
    if (this.options.rotationTracking && options.focalSearch)
      throw new RangeError(
        'rotationTracking requires supplied calibration; focalSearch is not supported with it',
      );
  }

  /** The focal search's result, after the session's first map; undefined before or without `focalSearch`. */
  get focalSearch(): FocalSearchResult | undefined {
    return this.focal;
  }

  /** Inputs still eligible for new descriptor edges; historical output is stored separately. */
  get activeViewIds(): readonly number[] {
    return this.images.map((view) => view.id);
  }

  /** Active-window evidence before/after map initialization. A snapshot, with no invented pose or depth. */
  get tracking(): ReconstructionTracking | undefined {
    return this.trackingState ? copyTracking(this.trackingState) : undefined;
  }
  get observationSelection(): CameraObservationSelection | undefined {
    return this.selectionState ? copyCameraSelection(this.selectionState) : undefined;
  }

  async add(
    image: ImageFeatures,
    pairs: readonly PairMatches[],
    cameraObservationMask?: CameraObservationMask,
  ): Promise<GpuIncrementalResult | undefined> {
    if (this.busy) throw new Error('Await the previous SfM session update.');
    if (this.seen.has(image.id)) throw new RangeError('Duplicate keyframe id');
    const copy = { ...image, camera: { ...image.camera }, keypoints: image.keypoints.slice() };
    let images = [...this.images, this.focal ? applyFocal([copy], this.focal.ratio)[0]! : copy];
    let edges = [
      ...this.pairs,
      ...pairs.map((pair) => ({
        ...pair,
        matches: pair.matches.slice(),
        distances: pair.distances?.slice(),
      })),
    ];
    validateFeatures(images, edges);
    if (cameraObservationMask) {
      const mask = snapshotCameraMasks([cameraObservationMask])![0]!;
      if (mask.viewId !== image.id || mask.values.length !== image.keypoints.length / 2)
        throw new RangeError(
          'Session camera mask must address the added image and original features',
        );
      this.options = {
        ...this.options,
        cameraObservationMasks: [
          ...(this.options.cameraObservationMasks ?? []).filter((m) => m.viewId !== image.id),
          mask,
        ],
      };
    }
    const windowSize = this.options.windowSize ?? 32;
    if (images.length > windowSize) {
      const recent = images.slice(-(windowSize - 2));
      const recentIds = new Set(recent.map((view) => view.id));
      let anchors: readonly number[] | undefined = this.result?.diagnostics.initialPair;
      // Retain the first reference while waiting for sufficient initial parallax.
      if (!this.state.model) anchors = [images[0]!.id];
      if (this.state.model) {
        // Shared-landmark counts of every registered pair, from one pass over the observations.
        const shared = new Map<string, number>();
        for (const point of this.state.model.landmarks) {
          const ids = point.observations.map((o) => o.viewId).sort((a, b) => a - b);
          for (let i = 0; i < ids.length; i++)
            for (let j = i + 1; j < ids.length; j++) {
              const key = `${ids[i]}:${ids[j]}`;
              shared.set(key, (shared.get(key) ?? 0) + 1);
            }
        }
        const common = (a: number, b: number) => shared.get(a < b ? `${a}:${b}` : `${b}:${a}`) ?? 0;
        // Move the local gauge to existing registered cameras; their world poses are preserved.
        const candidates = this.state.model.views.filter((view) => recentIds.has(view.id));
        let best = 0;
        for (let i = 0; i < candidates.length; i++)
          for (let j = i + 1; j < candidates.length; j++) {
            const a = candidates[i]!,
              b = candidates[j]!;
            const ac = cameraCenter(a.pose),
              bc = cameraCenter(b.pose);
            const distance = Math.hypot(...ac.map((v, k) => v - bc[k]!));
            if (distance > best && distance > 1e-8 && common(a.id, b.id) >= 12) {
              anchors = [a.id, b.id];
              best = distance;
            }
          }
      }
      const keep = new Set([...recentIds, ...(anchors ?? [])]);
      images = images.filter((view) => keep.has(view.id));
      edges = edges.filter((pair) => keep.has(pair.first) && keep.has(pair.second));
      if (this.state.model && anchors) {
        const existing = new Set(
          edges.map((pair) => [pair.first, pair.second].sort((a, b) => a - b).join(':')),
        );
        // Carry real shared observations across an evicted chain; verify these edges normally.
        for (const anchor of anchors)
          for (const view of images) {
            if (view.id === anchor) continue;
            const key = [anchor, view.id].sort((a, b) => a - b).join(':');
            if (existing.has(key)) continue;
            const matches: number[] = [];
            for (const point of this.state.model.landmarks) {
              let a: number | undefined, b: number | undefined;
              for (const o of point.observations) {
                if (o.viewId === anchor) a = o.featureId;
                else if (o.viewId === view.id) b = o.featureId;
              }
              if (a !== undefined && b !== undefined) matches.push(a, b);
            }
            if (matches.length >= 16) {
              edges.push({ first: anchor, second: view.id, matches: Uint32Array.from(matches) });
              existing.add(key);
            }
          }
        this.state.model = {
          views: this.state.model.views.filter((view) => keep.has(view.id)),
          landmarks: this.state.model.landmarks
            .map((point) => ({
              ...point,
              observations: point.observations.filter((o) => keep.has(o.viewId)),
            }))
            .filter((point) => point.observations.length >= 2),
        };
        this.result = {
          ...this.result!,
          diagnostics: {
            ...this.result!.diagnostics,
            initialPair: anchors as readonly [number, number],
          },
        };
      }
      // Retire verification snapshots belonging to evicted inputs; future edges are verified again.
      this.state.verification.clear();
    }
    this.busy = true;
    try {
      let result: GpuIncrementalResult | undefined;
      // With the focal search, the first map waits for the search's subset of views.
      const searched = this.focal,
        search = !searched && focalSearchControls(this.options.focalSearch);
      if (images.length >= (search ? search.views : 2) && edges.length) {
        try {
          result = await this.update(images, edges, false);
        } catch (error) {
          if (!(error instanceof ReconstructionInitializationError)) throw error;
          this.trackingState = error.tracking ? copyTracking(error.tracking) : undefined;
          this.selectionState = error.observationSelection
            ? copyCameraSelection(error.observationSelection)
            : undefined;
        }
      }
      // The run that found the focal mapped every view at it. The inputs follow.
      if (!searched && this.focal) images = applyFocal(images, this.focal.ratio);
      this.images = images;
      this.pairs = edges;
      this.seen.add(image.id);
      if (this.options.cameraObservationMasks) {
        const active = new Set(images.map((v) => v.id));
        this.options = {
          ...this.options,
          cameraObservationMasks: this.options.cameraObservationMasks.filter(
            (m) => !this.seen.has(m.viewId) || active.has(m.viewId),
          ),
        };
      }
      if (this.options.rotationTracking && (images.length < 2 || !edges.length))
        this.trackingState = { pairs: [], rotations: [], tracks: [], untriangulatedTrackIds: [] };
      if (images.length < 2 || !edges.length) {
        const selection = selectCameraObservations(
          images,
          edges,
          this.options.cameraObservationMasks,
        ).selection;
        this.selectionState = selection ? copyCameraSelection(selection) : undefined;
      }
      if (!result) this.unregistered.set(image.id, 'Waiting for a stable connected initial map');
      return result;
    } finally {
      this.busy = false;
    }
  }

  async finish(): Promise<GpuIncrementalResult> {
    if (this.busy) throw new Error('Await the previous SfM session update.');
    this.busy = true;
    try {
      return await this.update(this.images, this.pairs, true);
    } finally {
      this.busy = false;
    }
  }

  private async update(images: ImageFeatures[], pairs: PairMatches[], final: boolean) {
    // The depth tracker can refine archived views too; the native mapper resumes only its inputs.
    if (this.options.depthTracking && this.state.model) {
      const active = new Set(images.map((v) => v.id));
      this.state.model = {
        views: this.state.model.views.filter((v) => active.has(v.id)),
        landmarks: this.state.model.landmarks.flatMap((p) => {
          const observations = p.observations.filter((o) => active.has(o.viewId));
          return observations.length >= 2 ? [{ ...p, observations }] : [];
        }),
      };
    }
    const result = await runIncrementalGpu(
      this.device,
      images,
      pairs,
      {
        ...this.options,
        onReconstruction: (model) => {
          const full = this.history.update(model);
          this.options.onReconstruction?.(full);
        },
        initialPair: this.result?.diagnostics.initialPair ?? this.options.initialPair,
        finalBundleIterations: final ? (this.options.finalBundleIterations ?? 8) : 0,
        refineSharedFocal: final && (this.options.refineSharedFocal ?? false),
        // The search runs once, for the session's first map; later runs resume that map.
        focalSearch: this.focal || this.state.model ? undefined : this.options.focalSearch,
      },
      this.state,
      final,
    );
    const found = result.diagnostics.focalSearch;
    this.selectionState = result.observationSelection
      ? copyCameraSelection(result.observationSelection)
      : undefined;
    // The search's own verdicts stay in the cache until the window evicts a view, as others do.
    if (found && !this.focal) this.focal = found;
    this.state.model = copyReconstruction(result.reconstruction);
    this.trackingState = result.tracking ? copyTracking(result.tracking) : undefined;
    this.result = {
      ...result,
      reconstruction: this.state.model,
      diagnostics: {
        ...result.diagnostics,
        initialPair: [...result.diagnostics.initialPair] as [number, number],
      },
    };
    const reconstruction = this.history.update(result.reconstruction);
    for (const failure of result.unregistered)
      this.unregistered.set(failure.viewId, failure.reason);
    for (const view of reconstruction.views) this.unregistered.delete(view.id);
    return {
      ...result,
      reconstruction,
      unregistered: [...this.unregistered].map(([viewId, reason]) => ({ viewId, reason })),
    };
  }
}
