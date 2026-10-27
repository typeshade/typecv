/// <reference lib="webworker" />
import {
  PhaseGate,
  legacyGateSkips,
  trackGateDecision,
  type KeyframeGateMode,
} from '../lib/keyframe-gate.ts';
import {
  guideEssential,
  guideEssentialOptions,
  guidedMatchingRuns,
  preferGuided,
} from '../lib/guided-matching.ts';
import {
  mergeTrackerPairs,
  medianTrackDisplacement,
  trackAnchor,
  trackerVerdicts,
  type TrackerPair,
} from '../lib/klt-join.ts';
import { featureCap, tooFewFeatures } from '../lib/lab-features.ts';
import {
  admitsPair,
  admitsPrecomputedPair,
  retainedFeatureIds,
  retainedReferences,
  tracksCoverPair,
} from '../lib/lab-matching.ts';
import type { DenseQualityOptions } from '../lib/dense-quality.ts';
import type { KltSetup } from '../lib/sift-quality.ts';
import {
  DENSE_LAB_FUSION,
  createDenseJob,
  denseCleanDefault,
  denseGpuTimestampFields,
  type DenseFusion,
  denseFailure,
  endDense,
  keepFrame,
  runDense,
  type DenseJob,
  type DenseRunContext,
} from '../lib/dense-run.ts';
import { analysisScale, deviceBudgets } from '../lib/working-budget.ts';
import { cancelMultiDense, handleMultiVideo, isMultiRequest } from './multi-video.ts';
import { adapterFacts } from '../lib/adapter-info.ts';
import { reportGpuMs } from '../lib/run-record.ts';
import { GpuRequestContext, type GpuLossContext } from '../lib/gpu-loss-context.ts';
import type { GpuPreparationPhase } from '../lib/gpu-preparation.ts';
import { initializationFailureRecord } from '../lib/initialization-failure.ts';
import { sceneViewWithFrame } from '../lib/sfm-scene.ts';
import {
  LAB_KLT_CAPACITY,
  LAB_OPTIONS,
  LAB_WORKER_DEFAULTS,
  ORB_LAB_FEATURES,
  ORB_LAB_RATIO,
  ORB_MAX_DISTANCE,
  PHOTO_PROFILE,
} from '../lib/lab-options.ts';
export {};
type Frame = {
  id: number;
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
  time: number;
  sourceWidth?: number;
  sourceHeight?: number;
  image?: string;
  sequence?: number;
  /** The final frame of an uploaded video: the phase gate always keeps it. */
  last?: boolean;
  /** Known intrinsics in analysis pixels; without them the focal ratio and zero distortion are assumed. */
  camera?: { fx: number; fy: number; cx: number; cy: number; k1: number; k2: number };
  /**
   * Keypoints and raw matches from another front end (the COLMAP sample's SIFT), in
   * analysis pixels. The GPU feature and matching stages are skipped for such a frame.
   */
  precomputed?: {
    keypoints: Float64Array;
    pairs: { first: number; matches: Uint32Array }[];
  };
};
/** The feature and matching controls: `video` for footage, `photo` for still photographs. */
export type Profile = 'video' | 'photo';
type Request =
  | {
      type: 'start';
      id: number;
      focalRatio: number;
      /**
       * Estimate the shared focal (`focalSearch` of the mapper, `docs/auto-focal.md`); `focalRatio`
       * is then only the camera before the estimate. Ignored for calibrated photographs.
       */
      focalSearch?: boolean;
      profile?: Profile;
      bounded?: boolean;
      /** A flavor of the emulated double forced by the page (`?fp64=`); the device picks otherwise. */
      fp64?: 'float' | 'integer';
      /** The device's time budget per submission in ms (`?budget=`); the device's default otherwise. */
      budget?: number;
      /** The long side of an analysis frame in pixels (640 by default). The feature cap grows with
       *  the pixel count, to the SIFT front end's limit of 16384. */
      analysisEdge?: number;
      /** Guided matching (`?guided=1`): each SIFT pair is matched again along its epipolar lines. */
      guided?: boolean;
      /** The WebGPU kernel of guided matching (`?guided=tiled`); scalar by default. */
      guidedKernel?: 'tiled' | 'scalar';
      /** SIFT's first octave (`?octave=-1`): -1 upsamples each frame 2× first; 0 by default. */
      firstOctave?: -1 | 0;
      /** CLAHE on the gray image before SIFT's pyramid (`?clahe=1`); off by default. */
      clahe?: boolean;
      /** The feature front end (`?features=orb`): ORB instead of SIFT, which is the default. */
      features?: 'sift' | 'orb';
      orbTuning?: { maxFeatures?: number; ratio?: number; maxDistance?: number };
      /** The dense MVS after the sparse result (`?dense=1`, `docs/dense-mvs.md` 5.3). */
      dense?: boolean;
      /** Explicit Dense profiling override (?gpuTimestamps=1); sparse profiling stays unchanged. */
      denseGpuTimestamps?: boolean;
      /**
       * Clean the fused cloud after the fusion (`docs/dense-mvs.md` 3.12): `?denseClean=1` or `0`,
       * and by default after the pixel fusion only (`denseCleanDefault`).
       */
      denseClean?: boolean;
      /** The fusion (`?denseFusion=pixel|tsdf`): `DENSE_LAB_FUSION` by default (10.4). */
      denseFusion?: DenseFusion;
      denseQuality?: DenseQualityOptions;
      /** Mesh the final cloud (`docs/dense-mvs.md` 3.13); false unless `?denseMesh=1`. */
      denseMesh?: boolean;
      /** The keyframe gate (`?gate=phase|legacy`): phase correlation before SIFT, or the match rule after it. */
      gate?: KeyframeGateMode;
      /**
       * KLT tracking (`?klt=1`, `docs/klt-tracking.md` 5.2): the tracker follows points through every
       * sampled frame of a video and joins them to the SIFT keypoints of each keyframe. Ignored for
       * photographs and for the ORB front end.
       */
      klt?: boolean;
      /** The step 6 switches of a KLT run (`kltSetup` in `sift-quality.ts`). */
      kltSetup?: KltSetup;
    }
  | { type: 'frame'; id: number; frame: Frame }
  | { type: 'finish'; id: number }
  /** Stop the dense MVS between chunks (5.3). The sparse result stays. */
  | { type: 'cancelDense'; id: number };

/** The controls of a run: `LAB_OPTIONS`, with `PHOTO_PROFILE` for still photographs (`lib/lab-options.ts`). */
const controls = (profile: Profile) =>
  profile === 'photo' ? { ...LAB_OPTIONS, ...PHOTO_PROFILE } : LAB_OPTIONS;
let apiPromise: Promise<any> | undefined;
let device: any;
let deviceOptions: Record<string, unknown> | undefined;
let deviceTimestampRequest = true;
let adapter: any;
let deviceUnavailable = false;
let job:
  | {
      id: number;
      focal: number;
      profile: Profile;
      options: Omit<ReturnType<typeof controls>, 'maxFeatures'> & {
        maxFeatures: number;
        guidedMatching: boolean;
        guidedKernel: 'tiled' | 'scalar';
        siftFirstOctave: -1 | 0;
        keyframeGate: KeyframeGateMode;
        clahe: boolean;
        focalSearch: boolean;
        features: 'sift' | 'orb';
        klt: boolean;
        kltSetup?: KltSetup;
      };
      /** The KLT tracker of a run with `options.klt`, made on the first frame. */
      klt?: {
        tracker: any;
        width: number;
        height: number;
        /** The id of the frame the tracker took last, so that a frame goes in once. */
        pushed?: number;
        /** The positions of the alive tracks at the last keyframe (the gate of 3.6). */
        anchor?: Map<number, [number, number]>;
        /** The tracker's totals at the end of the run, read before the tracker is disposed. */
        final?: any;
        counters: {
          pairs: number;
          trackerOnlyPairs: number;
          pairMatches: number;
          duplicateMatches: number;
          conflictingMatches: number;
        };
      };
      /** The phase gate of a video run, absent for the legacy gate and for photographs. */
      phaseGate?: PhaseGate;
      /** Frames the gate kept and registration was attempted for. */
      keyframes: number;
      session: any;
      started: number;
      before: any;
      reused: boolean;
      features: Map<number, { handle: any; points: Float64Array; frame: Frame }>;
      featureCounts: number[];
      rejections: { viewId: number; reason: string }[];
      matchedPairs: number;
      /** Guided matching: the pairs matched again and the matches they added. */
      guidedPairs: number;
      guidedGain: number;
      stages: any[];
      operation: string;
      failed?: boolean;
      frames: Map<number, Omit<Frame, 'pixels'>>;
      colors: Map<number, Float32Array>;
      pointColors: Map<number, [number, number, number]>;
      latest?: any;
      references: number[];
      lastKeyframe?: {
        id: number;
        time: number;
        points: Float64Array;
        width: number;
        height: number;
      };
      skippedFrames: number;
      frameMs: number[];
      summary: Record<
        string,
        {
          calls: number;
          elapsedMs: number;
          dispatches: number;
          submissions: number;
          gpuMs?: number;
        }
      >;
      kernels: Set<string>;
      /** The kernel that started last; named when the device is lost. */
      lastKernel: string;
      snapshot: { lastPostedAt: number; pending?: any; timer?: ReturnType<typeof setTimeout> };
      /** Set when the run asked for the dense MVS (5.3). */
      dense?: DenseJob;
      denseQuality: DenseQualityOptions;
      denseGpuTimestamps?: boolean;
      /** Frozen totals of completed devices when Dense changes the native feature set. */
      priorDeviceGpu?: { gpuKernels: any[]; gpuDropped: number; chunks: Record<string, unknown> };
    }
  | undefined;
let queue = Promise.resolve();
let lastProgress = 0;
/** When a kernel ran for the first time in this worker (its pipeline was compiled), in ms. */
let lastCompile = 0;
/** The id of the last request, so that an error outside a request's handler still names a job. */
let lastRequestId = 0;
let requestContext = new GpuRequestContext();
/**
 * An error as the failure record reads it: the stack where there is one (the bundle's line and
 * column locate a minified frame), else the string form, else the error's own words. Safari
 * gives a worker's uncaught error to the page with no message, so the worker says it itself.
 */
function describeError(error: unknown): string {
  if (error instanceof Error) return error.stack || String(error) || error.name || 'Error';
  if (error && typeof error === 'object') {
    const text = String(error);
    if (text && text !== '[object Object]') return text;
    try {
      return JSON.stringify(error);
    } catch {
      return text;
    }
  }
  return String(error) || '(an error with no message)';
}
self.addEventListener('error', (event) => {
  failJob(
    lastRequestId,
    'Uncaught error in the worker: ' +
      (event.message || '(no message)') +
      (event.filename ? ' at ' + event.filename + ':' + event.lineno + ':' + event.colno : '') +
      (event.error ? '\n' + describeError(event.error) : ''),
    '',
  );
});
self.addEventListener('unhandledrejection', (event) => {
  failJob(lastRequestId, 'Unhandled rejection in the worker: ' + describeError(event.reason), '');
});

const post = (message: object) => self.postMessage({ id: job!.id, ...message });
function scene(model: any) {
  const active = job!;
  const result = {
    views: model.views.map((view: any) => sceneViewWithFrame(view, active.frames.get(view.id))),
    landmarks: model.landmarks.map((point: any) => {
      let color = active.pointColors.get(point.trackId);
      if (!color)
        for (const observation of point.observations) {
          const colors = active.colors.get(observation.viewId);
          if (colors && observation.featureId * 3 + 2 < colors.length) {
            color = Array.from(
              colors.subarray(observation.featureId * 3, observation.featureId * 3 + 3),
            ) as [number, number, number];
            active.pointColors.set(point.trackId, color);
            break;
          }
        }
      return { ...point, color: color ?? [0.7, 0.7, 0.7] };
    }),
  };
  active.latest = result;
  return result;
}
function record(stage: any) {
  const active = job!,
    entry = {
      operation: stage.operation,
      elapsedMs: stage.elapsedMs,
      tier: stage.execution.report.tier,
      dispatches: stage.execution.report.dispatches,
      submissions: stage.execution.report.submissions,
      applicationUploadBytes: stage.execution.applicationUploadBytes,
      applicationReadbackBytes: stage.execution.applicationReadbackBytes,
      // The summed `gpuTime` of the stage's groups; absent without `timestamp-query`.
      gpuMs: reportGpuMs(stage.execution.report),
    };
  const total = active.summary[entry.operation] ?? {
    calls: 0,
    elapsedMs: 0,
    dispatches: 0,
    submissions: 0,
    gpuMs: undefined as number | undefined,
  };
  active.summary[entry.operation] = {
    calls: total.calls + 1,
    elapsedMs: total.elapsedMs + entry.elapsedMs,
    dispatches: total.dispatches + (entry.dispatches ?? 0),
    submissions: total.submissions + (entry.submissions ?? 0),
    ...(total.gpuMs !== undefined || entry.gpuMs !== undefined
      ? { gpuMs: Math.round(((total.gpuMs ?? 0) + (entry.gpuMs ?? 0)) * 1e3) / 1e3 }
      : {}),
  };
  active.stages.push(entry);
  // The detailed list is bounded; the totals above keep the whole run.
  trimStages(active.stages);
}
/**
 * Bound the detailed stage list to `detailedStageLimit`: drop the oldest entries, keyframe entries
 * last. The focal search runs hundreds of operations at once, and the page still lists each
 * keyframe.
 */
export function trimStages(stages: { operation: string }[]) {
  let excess = stages.length - LAB_OPTIONS.detailedStageLimit;
  for (let i = 0; excess > 0 && i < stages.length;)
    if (stages[i]!.operation === 'keyframe') i++;
    else {
      stages.splice(i, 1);
      excess--;
    }
  if (excess > 0) stages.splice(0, excess);
}
/** Post the map to the page at most every snapshotIntervalMs; the latest model wins. */
function publish(model: any, force = false) {
  const active = job!,
    now = performance.now(),
    wait = LAB_OPTIONS.snapshotIntervalMs - (now - active.snapshot.lastPostedAt);
  if (force || wait <= 0) {
    if (active.snapshot.timer) clearTimeout(active.snapshot.timer);
    active.snapshot.timer = undefined;
    active.snapshot.pending = undefined;
    active.snapshot.lastPostedAt = now;
    post({ type: 'snapshot', reconstruction: scene(model) });
    return;
  }
  active.snapshot.pending = model;
  active.snapshot.timer ??= setTimeout(() => {
    const current = job;
    if (!current || current !== active) return;
    active.snapshot.timer = undefined;
    if (active.snapshot.pending) publish(active.snapshot.pending, true);
  }, wait);
}
/** Median displacement of matched features between two frames, in the current frame's pixels. */
function medianMotionPx(
  matches: Uint32Array,
  previousPoints: Float64Array,
  previous: { width: number; height: number },
  points: Float64Array,
  frame: { width: number; height: number },
): number {
  const distances: number[] = [];
  for (let i = 0; i < matches.length; i += 2) {
    const a = matches[i] * 2,
      b = matches[i + 1] * 2;
    distances.push(
      Math.hypot(
        points[b] - (previousPoints[a] * frame.width) / previous.width,
        points[b + 1] - (previousPoints[a + 1] * frame.height) / previous.height,
      ),
    );
  }
  if (!distances.length) return Infinity;
  distances.sort((x, y) => x - y);
  return distances[distances.length >> 1];
}
async function releaseFeatures() {
  if (!job) return;
  for (const { handle } of job.features.values()) await handle?.dispose();
  job.features.clear();
  // The totals first: the record reads them after the tracker is gone.
  if (job.klt && !job.klt.final) {
    job.klt.final = job.klt.tracker.stats();
    await job.klt.tracker.dispose().catch(() => {});
  }
}

/** Record a KLT stage as the other stages are, so that the run record carries its time. */
function recordKlt(operation: string, elapsedMs: number) {
  record({
    operation,
    elapsedMs,
    execution: {
      report: { tier: device?.backend ?? 'webgpu', dispatches: 0, submissions: 0 },
      applicationUploadBytes: 0,
      applicationReadbackBytes: 0,
    },
  });
}

/**
 * Push a sampled frame through the tracker (5.2). A frame with SIFT features gives the tracker its
 * gray image (level 0); any other frame (a skipped one) goes in as RGBA. Selection is left to the
 * join, which seeds the keypoints first.
 */
async function kltPush(api: any, frame: Frame, rgba: Uint8ClampedArray, handle?: any) {
  const active = job!;
  if (!active.options.klt) return;
  if (active.klt && (active.klt.width !== frame.width || active.klt.height !== frame.height))
    throw new Error('KLT tracking needs frames of one size');
  if (active.klt?.pushed === frame.id) return;
  const setup = active.options.kltSetup;
  active.klt ??= {
    tracker: await api.createKltTracker(device, frame.width, frame.height, {
      // 512 tracks gave the gain of 1024 at half the cost in the step 6 measurement.
      options: { capacity: setup?.capacity ?? LAB_KLT_CAPACITY },
      // The refresh of 3.4 added nothing in the step 6 measurement: off unless `?kltFill=1`.
      fillIn: setup?.fillIn === true,
    }),
    width: frame.width,
    height: frame.height,
    counters: {
      pairs: 0,
      trackerOnlyPairs: 0,
      pairMatches: 0,
      duplicateMatches: 0,
      conflictingMatches: 0,
    },
  };
  const started = performance.now();
  active.operation = 'tracking frame ' + (frame.id + 1);
  await active.klt.tracker.push(handle?.gray ? { sift: handle } : { rgba }, {
    frameId: frame.id,
    select: false,
  });
  active.klt.pushed = frame.id;
  recordKlt('kltPush', performance.now() - started);
}

async function kltJoinTracks(frame: Frame, points: Float64Array, views: number[]) {
  const active = job!;
  const started = performance.now();
  active.operation = 'joining tracks ' + (frame.id + 1);
  const joined = await active.klt!.tracker.join({ frameId: frame.id, keypoints: points, views });
  // The tracks as they stand at this keyframe: the reference of the gate of 3.6.
  active.klt!.anchor = trackAnchor(active.klt!.tracker.tracks());
  recordKlt('kltJoin', performance.now() - started);
  return joined.pairs as TrackerPair[];
}

/**
 * Join the tracks to the keypoints of a keyframe and merge the matches into its pairs (3.3). The
 * frame was pushed with `kltPush`. `views` are the earlier keyframes that can take a pair. With
 * `joinedEarly` the join was done already (`kltJoinTracks`). Returns the tracker matches the pairs
 * hold, for the verdict after registration.
 */
async function kltJoin(
  frame: Frame,
  points: Float64Array,
  pairs: any[],
  views: number[],
  joinedEarly?: TrackerPair[],
): Promise<Map<string, number> | undefined> {
  const active = job!;
  if (!active.klt) return undefined;
  const joined = joinedEarly ?? (await kltJoinTracks(frame, points, views));
  const merged = mergeTrackerPairs(pairs, joined);
  pairs.splice(0, pairs.length, ...merged.pairs);
  const counters = active.klt.counters;
  counters.pairMatches += merged.added;
  counters.duplicateMatches += merged.duplicate;
  counters.conflictingMatches += merged.conflicting;
  for (const pair of merged.pairs) {
    if (pair.kltMatches) counters.pairs++;
    if (pair.source === 'klt') counters.trackerOnlyPairs++;
  }
  return merged.tracked;
}

/**
 * The verdict of the pair verification on the tracker's matches (3.5, item 6): a track that a
 * verified pair dropped and none kept ends. A pair that failed verification gives no verdict.
 */
async function kltFeedback(
  api: any,
  frame: Frame,
  tracked: Map<string, number> | undefined,
  result: any,
) {
  const active = job!;
  if (!active.klt || !tracked?.size) return;
  const verdicts = trackerVerdicts(tracked, result?.verifiedPairs, frame.id);
  const ids: Uint32Array = api.kltReject(verdicts);
  if (ids.length) await active.klt.tracker.reject(ids);
}

/** The run record's `klt` block (`RunRecordKlt`), or undefined without a tracker. */
function kltRecord(active: NonNullable<typeof job>) {
  const klt = active.klt;
  if (!klt) return undefined;
  const stats = klt.final ?? klt.tracker.stats();
  return {
    pushes: stats.pushes,
    joins: stats.joins,
    alive: stats.alive,
    ended: stats.ended,
    joined: stats.joined,
    seeded: stats.seeded,
    filledIn: stats.filledIn,
    filtered: stats.filtered,
    matches: stats.matches,
    ...klt.counters,
    rejected: stats.rejected,
    lengths: stats.lengths,
    meanLived: stats.meanLived,
    msPush: stats.msPush,
    msSelect: stats.msSelect,
    msJoin: stats.msJoin,
  };
}

async function start(
  id: number,
  focal: number,
  profile: Profile,
  boundedRequest: boolean | undefined,
  fp64?: 'float' | 'integer',
  budget?: number,
  analysisEdge = LAB_WORKER_DEFAULTS.analysisEdge,
  firstOctave: -1 | 0 = LAB_WORKER_DEFAULTS.firstOctave,
  guided = LAB_WORKER_DEFAULTS.guided,
  dense = LAB_WORKER_DEFAULTS.dense,
  clahe = false,
  denseClean: boolean | undefined = undefined,
  denseMesh = false,
  gate: KeyframeGateMode = LAB_WORKER_DEFAULTS.gate,
  features: 'sift' | 'orb' = 'sift',
  orbTuning: { maxFeatures?: number; ratio?: number; maxDistance?: number } = {},
  klt = false,
  kltSetup?: KltSetup,
  guidedKernel: 'tiled' | 'scalar' = 'scalar',
  focalSearchRequest = LAB_WORKER_DEFAULTS.focalSearch,
  denseFusion: DenseFusion = DENSE_LAB_FUSION,
  denseQuality: DenseQualityOptions = {},
  denseGpuTimestamps?: boolean,
) {
  if (job) throw new Error('Another SfM job is already running.');
  const preparing = (phase: GpuPreparationPhase) =>
    self.postMessage({ id, type: 'preparing', phase });
  preparing('runtime');
  apiPromise ??= import(/* @vite-ignore */ new URL('/runtime/gpu.js', self.location.origin).href);
  const api = await apiPromise;
  // Adapter identity for run records; the ShadeDevice opens its own device.
  preparing('adapter');
  let adapterInfo: any = null;
  if (!adapter && navigator.gpu) {
    const requested = await navigator.gpu.requestAdapter();
    if (requested) {
      adapter = adapterFacts(requested);
    }
  }
  adapterInfo = adapter;
  // `bounded` (the page's `?bounded=1`): fence every submission on a queue completion, with 8
  // models per batch, so that no command stream runs long enough for a GPU watchdog to reset the
  // device. Slower; a diagnostic mode for a device loss, not a remedy: a single long dispatch
  // still loses the device (`sweepPairModelSvd` on an iPhone, 2026-10-19). The remedy is the time
  // budget of `docs/gpu-resident-scene.md` step 2, which chunks the long kernels themselves.
  const bounded = boundedRequest === true;
  // A device opened in the other submission mode, or under another flavor of the emulated double
  // than the page forces, is replaced, not reused.
  // The working and pool budgets grow with the pixel count of the analysis frame (`analysisScale`
  // and `deviceBudgets` of `lib/working-budget.ts`): 64 and 32 MiB at 640 px, 256 and 128 MiB at
  // 1280 px. `docs/gpu-budget.md` section 2 and Amendment 3 of `docs/dense-mvs.md`: the budget
  // comes from the adapter's limits, half of its `maxBufferSize` when that is larger, and the
  // device requests them. The stages split their work to fit it: size changes speed, not success.
  const scale = analysisScale(analysisEdge, firstOctave);
  const { workingBytes, poolBytes } = deviceBudgets(scale, adapter?.limits?.maxBufferSize);
  const reused =
    !!device &&
    !deviceUnavailable &&
    deviceTimestampRequest === true &&
    device.boundedSubmissions === bounded &&
    (fp64 === undefined || device.fp64Flavor === fp64) &&
    (budget === undefined || device.timeBudgetMs === budget) &&
    device.workingBytes === workingBytes &&
    device.adapterLimits === true;
  preparing('device');
  if (device && !reused) {
    await device?.dispose();
    device = undefined;
    deviceUnavailable = false;
  }
  const started = performance.now();
  // The flavor of the emulated double (TypeShade change 0073): the page's `?fp64=`, else the
  // device's choice from the adapter, `'integer'` on an Apple GPU and `'float'` elsewhere.
  const openOptions = {
    backend: 'webgpu',
    fp64Flavor: fp64 ?? 'auto',
    // The page's `?budget=`, else the device's default (250 ms, TypeShade change 0074).
    timeBudgetMs: budget,
    workingBytes,
    poolBytes,
    adapterLimits: true,
    boundedSubmissions: bounded,
    solverCheckpoints: true,
    maxModelsPerBatch: bounded ? 64 : 1024,
    onProgress: ({ kernel, phase }: { kernel: string; phase: string }) => {
      if (!job) return;
      requestContext.progress(kernel, phase);
      if (phase === 'start') job.lastKernel = kernel;
      // The first job compiles each kernel on first use; the count shows the cold start moving.
      const first = !job.kernels.has(kernel);
      job.kernels.add(kernel);
      if (first) lastCompile = performance.now();
      // 5.3: during the MVS the page receives at most one message per reference and stage.
      if (job.dense?.running) return;
      if (first || performance.now() - lastProgress > 100) {
        lastProgress = performance.now();
        post({
          type: 'progress',
          detail: job.operation + ' · ' + kernel + ' (' + phase + ')',
          kernels: job.kernels.size,
          // The compile count is shown only while new kernels still arrive: once every kernel the
          // run needs is built, the time goes to the work itself, not to compilation.
          cold: !job.reused && performance.now() - lastCompile < 2000,
        });
      }
    },
  };
  if (!device) {
    device = await api.ShadeDevice.open(openOptions);
    deviceOptions = openOptions;
    deviceTimestampRequest = true;
  }
  const opened = device;
  preparing('session');
  // The feature cap follows the pixel count of the analysis frame (`featureCap`): 4096 at 640 px,
  // 16384 at 1280 px (the SIFT front end's limit). The keyframe motion threshold scales in
  // processFrame.
  const base = controls(profile);
  const options = {
    ...base,
    maxFeatures: featureCap(base.maxFeatures, scale),
    siftFirstOctave: firstOctave,
    guidedMatching: guided,
    guidedKernel,
    keyframeGate: gate,
    clahe,
    features,
    // Calibrated photographs keep their cameras: the search runs for video only.
    focalSearch: focalSearchRequest && profile === 'video',
    // Tracking needs a sequence and the SIFT gray: a video run with the SIFT front end.
    klt: klt && profile === 'video' && features === 'sift',
    ...(klt && profile === 'video' && features === 'sift' && kltSetup ? { kltSetup } : {}),
    // ORB is the binary front end: the record names it `corner` (FAST corners, Hamming matching).
    ...(features === 'orb'
      ? {
          frontEnd: 'corner' as const,
          maxFeatures: orbTuning.maxFeatures ?? ORB_LAB_FEATURES,
          matchRatio: orbTuning.ratio ?? ORB_LAB_RATIO,
          maxDistance: orbTuning.maxDistance ?? ORB_MAX_DISTANCE,
        }
      : {}),
  };
  // The per-kernel table of a reused device holds the earlier runs; this run starts it empty.
  device.resetKernelStatistics?.();
  job = {
    id,
    focal,
    profile,
    options,
    started,
    reused,
    before: device.statistics,
    features: new Map(),
    featureCounts: [],
    rejections: [],
    matchedPairs: 0,
    guidedPairs: 0,
    guidedGain: 0,
    stages: [],
    operation: 'initializing',
    frames: new Map(),
    colors: new Map(),
    pointColors: new Map(),
    references: [],
    skippedFrames: 0,
    keyframes: 0,
    phaseGate: gate === 'phase' && profile === 'video' ? new PhaseGate() : undefined,
    frameMs: [],
    summary: {},
    kernels: new Set(),
    lastKernel: '',
    snapshot: { lastPostedAt: -Infinity },
    denseQuality: { ...denseQuality },
    dense: dense
      ? createDenseJob(denseClean ?? denseCleanDefault(denseFusion), denseMesh, denseFusion)
      : undefined,
    denseGpuTimestamps,
    session: new api.IncrementalSfmSession(device, {
      windowSize: options.windowSize,
      ransacIterations: options.ransacIterations,
      maxErrorPx: options.maxErrorPx,
      minInitialAngleDeg: options.minInitialAngleDeg,
      minTriangulationAngleDeg: options.minTriangulationAngleDeg,
      bundleIterations: options.bundleIterations,
      finalBundleIterations: options.finalBundleIterations,
      refineSharedFocal: options.refineSharedFocal,
      focalSearch: options.focalSearch,
      onStage: record,
      onReconstruction: (model: any) => publish(model),
      onProgress: (event: any) => {
        job!.operation = event.operation + ' #' + event.index;
        post({ type: 'progress', detail: job!.operation + ' (' + event.phase + ')' });
      },
    }),
  };
  if (!reused)
    opened.lost?.then((info: any) => {
      if (device !== opened) return;
      deviceUnavailable = true;
      const location = requestContext.location(job?.operation ?? 'initializing');
      if (job?.dense?.running)
        endDense(denseContext(job), {
          failure: denseFailure(
            job.dense,
            'WebGPU device lost during the dense MVS at kernel ' +
              (job.lastKernel || 'none') +
              ': ' +
              info.message +
              ' [' +
              info.reason +
              ']',
            job.lastKernel,
            String(info.reason ?? ''),
            requestContext.snapshot(info.diagnostics ?? device.diagnostics),
          ),
        });
      else if (job)
        failJob(
          job.id,
          'WebGPU device lost during ' +
            location.operation +
            (location.kernel ? ' (last kernel ' + location.kernel + ')' : '') +
            (device.boundedSubmissions ? ' (bounded submissions)' : '') +
            ' (fp64 ' +
            device.fp64Flavor +
            '): ' +
            info.message +
            ' [' +
            info.reason +
            ']',
          String(info.reason ?? ''),
          requestContext.snapshot(info.diagnostics ?? device.diagnostics),
        );
    });
  post({ type: 'ready', adapter: adapterInfo });
}

async function processFrame(frame: Frame) {
  const active = job!;
  const api = await apiPromise!;
  const started = performance.now();
  const { pixels: _pixels, ...metadata } = frame;
  active.frames.set(frame.id, { ...metadata, sequence: active.featureCounts.length });
  // Retain only recent image cards; old frames remain addressable in the source video.
  if (active.frames.size > 24) {
    const old = [...active.frames.values()][active.frames.size - 25];
    delete old.image;
  }
  if (frame.precomputed) return processPrecomputedFrame(frame, started);
  // The phase gate decides before SIFT: a skipped frame costs no features and no matching.
  let legacyRule = true;
  if (active.phaseGate) {
    const decision = active.phaseGate.decide(
      frame.pixels,
      frame.width,
      frame.height,
      frame.time,
      frame.last === true,
    );
    if (decision.action === 'skip') {
      active.phaseGate.discard();
      active.skippedFrames++;
      await kltPush(api, frame, frame.pixels);
      const shift = decision.shift
        ? ', translation ' +
          (100 * Math.hypot(decision.shift.fx, decision.shift.fy)).toFixed(1) +
          '% of the frame'
        : '';
      post({
        type: 'frameComplete',
        frameId: frame.id,
        time: frame.time,
        elapsedMs: performance.now() - started,
        skipped: 'Not a keyframe: ' + decision.reason + shift,
        keyframe: false,
        unregistered: active.rejections,
      });
      return;
    }
    // A frame the gate kept needs no second opinion; a weak peak leaves the decision to the legacy rule.
    legacyRule = decision.action === 'fallback';
  }
  // The track gate (`?kltGate=1`, 3.6): with the legacy gate, the median displacement of the tracks
  // since the last keyframe decides before SIFT runs. Too few common tracks leave it to the rule below.
  let kept = !legacyRule;
  if (active.options.klt && active.options.kltSetup?.gate && !active.phaseGate) {
    await kltPush(api, frame, frame.pixels);
    const klt = active.klt!;
    if (active.lastKeyframe && klt.anchor) {
      const { median, tracks } = medianTrackDisplacement(klt.anchor, klt.tracker.tracks()),
        gap = frame.time - active.lastKeyframe.time,
        decision = trackGateDecision(median, gap, frame.width, LAB_OPTIONS);
      if (decision !== 'undecided') {
        if (decision === 'skip') {
          active.skippedFrames++;
          post({
            type: 'frameComplete',
            frameId: frame.id,
            time: frame.time,
            elapsedMs: performance.now() - started,
            skipped:
              'Not a keyframe: median track displacement ' +
              median.toFixed(1) +
              'px since the last keyframe (' +
              gap.toFixed(2) +
              's ago, ' +
              tracks +
              ' tracks)',
            keyframe: false,
            unregistered: active.rejections,
          });
          return;
        }
        legacyRule = false;
        kept = true;
      }
    }
  }
  active.operation = 'features frame ' + (frame.id + 1);
  const sift = active.options.frontEnd === 'sift';
  const orb = active.options.features === 'orb';
  const handle = sift
    ? await api.extractSiftGpuResident(device, frame.pixels, frame.width, frame.height, {
        maxFeatures: active.options.maxFeatures,
        firstOctave: active.options.siftFirstOctave,
        clahe: active.options.clahe,
        // The tracker takes the gray image of the frame as its level 0 (5.2).
        ...(active.options.klt && active.klt?.pushed !== frame.id
          ? { grayForKlt: { levels: api.kltOptionsFor(frame.width).levels } }
          : {}),
      })
    : orb
      ? await api.extractOrbGpuResident(device, frame.pixels, frame.width, frame.height, {
          maxFeatures: active.options.maxFeatures,
        })
      : await api.extractFeaturesGpuPyramidResident(
          device,
          frame.pixels,
          frame.width,
          frame.height,
          {
            threshold: active.options.threshold,
            maxFeatures: active.options.maxFeatures,
            smoothDescriptors: active.options.smoothDescriptors,
          },
        );
  const points = await handle.readKeypoints();
  sampleColors(frame, points);
  keepPixels(frame);
  // The tracker reads these bytes when the frame turns out not to be a keyframe.
  const rgba = frame.pixels;
  frame.pixels = new Uint8ClampedArray();
  record({
    operation: 'features',
    elapsedMs: performance.now() - started,
    execution: { report: handle.report, ...handle.execution },
  });
  active.featureCounts.push(handle.count);
  post({ type: 'features', frameId: frame.id, time: frame.time, points });
  if (tooFewFeatures(handle.count)) {
    await handle.dispose();
    await kltPush(api, frame, rgba);
    active.dense?.rgba.delete(frame.id);
    active.rejections.push({ viewId: frame.id, reason: 'Too few detected features' });
    post({
      type: 'frameComplete',
      frameId: frame.id,
      time: frame.time,
      elapsedMs: performance.now() - started,
      skipped: 'Too few detected features',
      unregistered: active.rejections,
    });
    return;
  }
  const pairs: any[] = [];
  const eligible = new Set<number>(active.session.activeViewIds);
  for (const [id, feature] of active.features)
    if (!eligible.has(id)) {
      await feature.handle?.dispose();
      active.features.delete(id);
    }
  const previous = [...active.features.values()]
    .filter((feature) => feature.handle)
    .sort((a, b) => a.frame.id - b.frame.id);
  // Match the last keyframe first: little motion since it means this frame is not a keyframe.
  const last = previous.at(-1);
  if (last) previous.unshift(previous.splice(previous.length - 1, 1)[0]);
  active.features.set(frame.id, { handle, points, frame });
  // `?kltMatching=tracks` (3.6): the frame is a keyframe already, so the tracks join its keypoints
  // now, and a pair that they cover with enough matches skips the descriptor matching.
  let early: TrackerPair[] | undefined;
  const covered = new Set<number>();
  if (active.options.klt && active.options.kltSetup?.matching === 'tracks' && sift && kept) {
    await kltPush(api, frame, rgba, handle);
    early = await kltJoinTracks(
      frame,
      points,
      previous.map((other) => other.frame.id),
    );
    for (const t of early) if (tracksCoverPair(t.matches.length)) covered.add(t.first);
  }
  for (const other of previous) {
    if (covered.has(other.frame.id)) {
      active.matchedPairs++;
      if (other === last)
        postMatchLines(
          other.points,
          other.frame,
          points,
          frame,
          early!.find((t) => t.first === other.frame.id)!.matches,
        );
      continue;
    }
    const matchStarted = performance.now();
    active.operation = 'matching ' + (other.frame.id + 1) + ' → ' + (frame.id + 1);
    let matched = sift
      ? await api.matchFloatDescriptorsGpu(device, other.handle.descriptors, handle.descriptors, {
          ratio: active.options.matchRatio,
          mutual: true,
        })
      : await api.matchResidentFeaturesGpu(device, other.handle, handle, {
          ratio: active.options.matchRatio,
          maxDistance: active.options.maxDistance,
        });
    record({
      operation: 'matches',
      elapsedMs: performance.now() - matchStarted,
      execution: { report: matched.report, ...matched.execution },
    });
    if (legacyRule && other === last && active.lastKeyframe) {
      const motion = medianMotionPx(matched.matches, other.points, other.frame, points, frame),
        gap = frame.time - active.lastKeyframe.time;
      if (legacyGateSkips(matched.matches.length, motion, gap, frame.width, LAB_OPTIONS)) {
        active.features.delete(frame.id);
        await handle.dispose();
        await kltPush(api, frame, rgba);
        active.dense?.rgba.delete(frame.id);
        active.skippedFrames++;
        active.featureCounts.pop();
        const reason =
          'Not a keyframe: median motion ' +
          motion.toFixed(1) +
          'px since the last keyframe (' +
          gap.toFixed(2) +
          's ago)';
        post({
          type: 'frameComplete',
          frameId: frame.id,
          time: frame.time,
          elapsedMs: performance.now() - started,
          skipped: reason,
          keyframe: false,
          unregistered: active.rejections,
        });
        return;
      }
    }
    if (guidedMatchingRuns(sift, active.options.guidedMatching, matched.matches.length)) {
      const guidedStarted = performance.now();
      active.operation = 'guided matching ' + (other.frame.id + 1) + ' → ' + (frame.id + 1);
      const guided = await guidedMatches(api, other, handle, points, frame, matched);
      if (guided) {
        record({
          operation: 'guidedMatches',
          elapsedMs: performance.now() - guidedStarted,
          execution: { report: guided.report, ...guided.execution },
        });
        if (preferGuided(matched.matches.length, guided.matches.length)) matched = guided;
      }
    }
    if (admitsPair(matched.matches.length)) {
      pairs.push({
        first: other.frame.id,
        second: frame.id,
        matches: matched.matches,
        distances: matched.distances,
      });
      active.matchedPairs++;
    }
    if (other === last) postMatchLines(other.points, other.frame, points, frame, matched.matches);
  }
  // KLT: the frame is a keyframe. The tracker takes its gray, joins its tracks to the keypoints
  // and adds the matches to the pairs; registration then judges them with the others.
  let tracked: Map<string, number> | undefined;
  if (active.options.klt) {
    await kltPush(api, frame, rgba, handle);
    tracked = await kltJoin(
      frame,
      points,
      pairs,
      previous.map((other) => other.frame.id),
      early,
    );
  }
  const registered = await registerFrame(frame, points, pairs, started);
  await kltFeedback(api, frame, tracked, registered);
}

/** A frame whose keypoints and matches another front end produced: no GPU feature or matching stage. */
async function processPrecomputedFrame(frame: Frame, started: number) {
  const active = job!;
  const { keypoints, pairs: given } = frame.precomputed!;
  const points = keypoints;
  sampleColors(frame, points);
  keepPixels(frame);
  frame.pixels = new Uint8ClampedArray();
  active.featureCounts.push(points.length / 2);
  post({ type: 'features', frameId: frame.id, time: frame.time, points });
  const eligible = new Set<number>(active.session.activeViewIds);
  for (const [id, feature] of active.features)
    if (!eligible.has(id)) {
      await feature.handle?.dispose();
      active.features.delete(id);
    }
  active.features.set(frame.id, { handle: undefined, points, frame });
  const pairs: any[] = [];
  let last: { first: number; matches: Uint32Array } | undefined;
  for (const pair of given) {
    const other = active.features.get(pair.first);
    if (!other || !admitsPrecomputedPair(pair.matches.length)) continue;
    pairs.push({ first: pair.first, second: frame.id, matches: pair.matches });
    active.matchedPairs++;
    if (!last || pair.first > last.first) last = pair;
  }
  if (last) {
    const other = active.features.get(last.first)!;
    postMatchLines(other.points, other.frame, points, frame, last.matches);
  }
  await registerFrame(frame, points, pairs, started);
}

/** 5.3: with the dense MVS on, keep the frame's RGBA until the MVS (`keepFrame`). */
function keepPixels(frame: Frame) {
  if (job!.dense) keepFrame(job!.dense, frame);
}

/** The colour under each keypoint, for the point cloud. */
function sampleColors(frame: Frame, points: Float64Array) {
  const active = job!;
  const colors = new Float32Array((points.length / 2) * 3);
  for (let i = 0; i < points.length / 2; i++) {
    const x = Math.max(0, Math.min(frame.width - 1, Math.round(points[i * 2])));
    const y = Math.max(0, Math.min(frame.height - 1, Math.round(points[i * 2 + 1])));
    const offset = (y * frame.width + x) * 4;
    for (let channel = 0; channel < 3; channel++)
      colors[i * 3 + channel] = frame.pixels[offset + channel] / 255;
  }
  active.colors.set(frame.id, colors);
  if (active.colors.size > 64) active.colors.delete(active.colors.keys().next().value!);
  return colors;
}

/** The candidate match lines between the last matched frame and this one, for the overlay. */
function postMatchLines(
  otherPoints: Float64Array,
  otherFrame: Frame,
  points: Float64Array,
  frame: Frame,
  matches: ArrayLike<number>,
) {
  const lines: number[] = [];
  for (let i = 0; i < Math.min(matches.length, 160); i += 2) {
    const a = matches[i] * 2,
      b = matches[i + 1] * 2;
    lines.push(
      (otherPoints[a] * frame.width) / otherFrame.width,
      (otherPoints[a + 1] * frame.height) / otherFrame.height,
      points[b],
      points[b + 1],
    );
  }
  post({ type: 'matches', frameId: frame.id, lines, count: matches.length / 2 });
}

/**
 * The camera of a frame in analysis pixels: the sample's calibration, else the focal search's
 * estimate once it ran, else the assumed focal.
 */
function frameCamera(frame: Frame) {
  const sourceWidth = frame.sourceWidth ?? frame.width;
  const sourceHeight = frame.sourceHeight ?? frame.height;
  const sourceFocal = sourceWidth * (job!.session.focalSearch?.ratio ?? job!.focal);
  return (
    frame.camera ?? {
      fx: (sourceFocal * frame.width) / sourceWidth,
      fy: (sourceFocal * frame.height) / sourceHeight,
      cx: (frame.width - 1) / 2,
      cy: (frame.height - 1) / 2,
      k1: 0,
      k2: 0,
    }
  );
}

/** Lines of a guided match lie within this many pixels of a candidate, as COLMAP's 4 px default. */
const GUIDED_LINE_PX = 4;

/**
 * Guided matching: estimate E from the plain matches of a pair (a short calibrated RANSAC, see
 * `guideEssentialOptions`), then match again with only the candidates within
 * `GUIDED_LINE_PX` of each query's epipolar line, in normalized camera coordinates with the
 * radial distortion inverted, so the ratio test compares the matches the geometry allows.
 * Returns the guided result, or undefined when the pair fails.
 */
async function guidedMatches(
  api: any,
  other: { handle: any; points: Float64Array; frame: Frame },
  handle: any,
  points: Float64Array,
  frame: Frame,
  plain: { matches: Uint32Array; distances: Float64Array },
) {
  const active = job!;
  const a = frameCamera(other.frame),
    b = frameCamera(frame),
    n = plain.matches.length / 2,
    pixels = new Float64Array(n * 4);
  for (let i = 0; i < n; i++) {
    const p = plain.matches[i * 2]! * 2,
      q = plain.matches[i * 2 + 1]! * 2;
    pixels.set([other.points[p]!, other.points[p + 1]!, points[q]!, points[q + 1]!], i * 4);
  }
  // A short RANSAC on E finds the band. The mapper's full verification of the chosen matches is
  // the one that classifies the pair (homography, parallax, pose), so the guide estimates E only.
  const E = await guideEssential(
    (options) => api.estimateEssentialRansacGpu(device, pixels, a, b, options),
    guideEssentialOptions(
      a,
      b,
      active.options.maxErrorPx,
      active.options.ransacIterations,
      101 + frame.id,
    ),
  );
  if (!E) return undefined;
  const normalized = (camera: any, source: Float64Array) => {
    const out = new Float64Array(source.length);
    for (let i = 0; i < source.length; i += 2) {
      const xy = api.unprojectPixel(camera, [source[i], source[i + 1]]);
      // A pixel beyond the distortion model's monotone range sits far off every line.
      out[i] = xy ? xy[0] : 1e6;
      out[i + 1] = xy ? xy[1] : 1e6;
    }
    return out;
  };
  const guided = await api.matchFloatDescriptorsGpu(
    device,
    other.handle.descriptors,
    handle.descriptors,
    {
      ratio: active.options.matchRatio,
      mutual: true,
      epipolar: {
        fundamental: E,
        firstPoints: normalized(a, other.points),
        secondPoints: normalized(b, points),
        maxLinePx: GUIDED_LINE_PX / Math.max(a.fx, a.fy, b.fx, b.fy),
        kernel: active.options.guidedKernel,
      },
    },
  );
  active.guidedPairs++;
  active.guidedGain += Math.max(0, guided.matches.length / 2 - n);
  return guided;
}

/** Hand the frame and its verified-candidate pairs to the session and report the keyframe. */
async function registerFrame(frame: Frame, points: Float64Array, pairs: any[], started: number) {
  const active = job!;
  // Keep a long-baseline first reference, two recent reference images and three adjacent frames.
  active.references = retainedReferences(active.references, active.features, frame);
  const keepFeatures = retainedFeatureIds(active.references, active.features.keys());
  for (const [id, feature] of active.features)
    if (!keepFeatures.has(id) && feature.handle) {
      await feature.handle.dispose();
      active.features.delete(id);
    }
  const result = await active.session.add(
    {
      id: frame.id,
      camera: frameCamera(frame),
      keypoints: points,
    },
    pairs,
  );
  active.keyframes++;
  active.phaseGate?.accept();
  active.lastKeyframe = {
    id: frame.id,
    time: frame.time,
    points,
    width: frame.width,
    height: frame.height,
  };
  if (result) publish(result.reconstruction, true);
  const frameElapsed = performance.now() - started;
  active.frameMs.push(frameElapsed);
  active.stages.push({ operation: 'keyframe', frame: frame.id, elapsedMs: frameElapsed });
  trimStages(active.stages);
  post({
    type: 'frameComplete',
    frameId: frame.id,
    time: frame.time,
    elapsedMs: frameElapsed,
    keyframe: true,
    registered: result?.reconstruction.views.length ?? 0,
    cameraRegistered:
      result?.reconstruction.views.some((view: any) => view.id === frame.id) ?? false,
    verifiedInliers: Math.max(
      0,
      ...(result?.diagnostics.pairVerification ?? [])
        .filter((pair: any) => pair.first === frame.id || pair.second === frame.id)
        .map((pair: any) => pair.inliers),
    ),
    waiting: !result,
    unregistered: [...(result?.unregistered ?? []), ...active.rejections],
  });
  return result;
}

/**
 * The per-kernel table of the device since the run began (`ShadeDevice.statistics.kernels`): the
 * page turns it into `timing.gpu`. A lost or disposed device may refuse the read.
 */
function gpuFields(active = job): { gpuKernels: any[]; gpuDropped: number } {
  let current: { rows: any[]; dropped: number } = { rows: [], dropped: 0 };
  try {
    const table = device.kernelStatistics;
    if (table && Array.isArray(table.rows)) current = table;
  } catch {
    /* Completed-device totals survive a loss or failed replacement. */
  }
  const rows = new Map<string, any>();
  for (const row of [...(active?.priorDeviceGpu?.gpuKernels ?? []), ...current.rows]) {
    const before = rows.get(row.name);
    if (!before) rows.set(row.name, { ...row });
    else {
      const combined = { ...before };
      for (const field of ['groups', 'dispatches', 'timedGroups', 'gpuNs'])
        combined[field] = before[field] + row[field];
      rows.set(row.name, combined);
    }
  }
  return {
    gpuKernels: [...rows.values()],
    gpuDropped: (active?.priorDeviceGpu?.gpuDropped ?? 0) + current.dropped,
  };
}

/** What a run reports whether it completes or fails: counts, timings, options and the kernels. */
function outcome(active: NonNullable<typeof job>) {
  let statistics: any;
  try {
    statistics = device?.statistics;
  } catch {
    statistics = undefined;
  }
  // What the device measured of each chunk key (TypeShade change 0074) up to now: at a failure,
  // what it measured before the loss. A lost or disposed device may refuse the read.
  let chunks: Record<string, unknown> = {};
  try {
    chunks = { ...(device?.chunkRates ?? {}) };
  } catch {
    chunks = {};
  }
  chunks = { ...active.priorDeviceGpu?.chunks, ...chunks };
  return {
    ...gpuFields(active),
    featureCounts: active.featureCounts,
    matchedPairs: active.matchedPairs,
    klt: kltRecord(active),
    guided: active.options.guidedMatching
      ? { pairs: active.guidedPairs, addedMatches: active.guidedGain }
      : undefined,
    skippedFrames: active.skippedFrames,
    keyframes: active.keyframes,
    frameMs: active.frameMs,
    summary: active.summary,
    kernels: [...active.kernels],
    chunks,
    // The session's current automatic calibration, including accepted final archive refinement.
    focalSearch: active.session.focalSearch
      ? {
          ratio: active.session.focalSearch.ratio,
          gridRatio: active.session.focalSearch.gridRatio,
          views: active.session.focalSearch.views.length,
        }
      : undefined,
    budgetMs: device?.timeBudgetMs ?? null,
    adapter,
    options: {
      ...active.options,
      profile: active.profile,
      focalRatio: active.focal,
      boundedSubmissions: device?.boundedSubmissions === true,
      fp64Flavor: device?.fp64Flavor ?? 'float',
    },
    stages: [
      {
        operation: 'runTotal',
        tier: device?.backend ?? 'webgpu',
        reusedDevice: active.reused,
        elapsedMs: performance.now() - active.started,
        applicationUploadBytes: statistics
          ? statistics.applicationUploadBytes - active.before.applicationUploadBytes
          : 0,
        applicationReadbackBytes: statistics
          ? statistics.applicationReadbackBytes - active.before.applicationReadbackBytes
          : 0,
      },
      ...active.stages,
    ],
  };
}
/** Report a run that ended in an error, with everything measured before it. */
function failJob(
  id: number,
  message: string,
  reason: string,
  context?: GpuLossContext,
  error?: unknown,
) {
  const active = job;
  // 5.3: an error after the sparse result ends the MVS only. The sparse result stays.
  if (active?.dense?.running) {
    endDense(denseContext(active), {
      failure: denseFailure(
        active.dense,
        message,
        active.lastKernel,
        reason,
        context ?? requestContext.snapshot(device?.diagnostics),
      ),
    });
    return;
  }
  if (active?.failed) return;
  if (active) active.failed = true;
  const location = requestContext.location(active?.operation ?? 'initializing');
  const initialization = initializationFailureRecord(error);
  self.postMessage({
    id,
    type: 'error',
    message,
    failure: active
      ? {
          message: message.slice(0, 2000),
          operation: location.operation,
          kernel: location.kernel,
          reason: reason.slice(0, 64),
          framesProcessed: active.featureCounts.length,
          context: context ?? requestContext.snapshot(device?.diagnostics),
          ...(initialization ? { initialization } : {}),
        }
      : undefined,
    partial: active
      ? {
          ...outcome(active),
          latest: active.latest
            ? {
                views: active.latest.views.length,
                landmarks: active.latest.landmarks.length,
                observations: active.latest.landmarks.reduce(
                  (sum: number, point: any) => sum + point.observations.length,
                  0,
                ),
              }
            : null,
        }
      : undefined,
  });
}
async function finish() {
  const active = job!;
  active.operation = active.latest ? 'final refinement' : 'initializing 3D map';
  const result = await active.session.finish();
  await releaseFeatures();
  const reconstruction = scene(result.reconstruction);
  post({
    type: 'result',
    reconstruction,
    diagnostics: result.diagnostics,
    unregistered: [...result.unregistered, ...active.rejections],
    ...outcome(active),
  });
  if (active.dense) {
    active.operation = 'dense MVS';
    try {
      await prepareDenseDevice(active);
    } catch (error) {
      endDense(denseContext(active), {
        failure: denseFailure(
          active.dense,
          'Could not open the dense GPU device: ' + describeError(error),
          '',
          '',
        ),
      });
      job = undefined;
      return;
    }
    await runDense(denseContext(active), reconstruction);
  }
  job = undefined;
}

/** Dense may omit profiling queries; the sparse result is already saved and has no GPU consumers. */
async function prepareDenseDevice(active: NonNullable<typeof job>): Promise<void> {
  const previous = device;
  const requested = active.denseGpuTimestamps ?? previous.boundedSubmissions === true;
  const features = previous.info?.webgpu?.features;
  if (
    deviceTimestampRequest === requested ||
    (!requested && Array.isArray(features) && !features.includes('timestamp-query'))
  )
    return;
  const settings = {
    ...deviceOptions,
    gpuTimestamps: requested,
    fp64Flavor: previous.fp64Flavor,
    timeBudgetMs: previous.timeBudgetMs,
  };
  let previousChunks: Record<string, unknown> = {};
  try {
    previousChunks = { ...previous.chunkRates };
  } catch {
    /* Keep earlier completed keys. */
  }
  active.priorDeviceGpu = {
    ...gpuFields(active),
    chunks: { ...active.priorDeviceGpu?.chunks, ...previousChunks },
  };
  device = undefined;
  await previous.dispose();
  const opened = await (await apiPromise!).ShadeDevice.open(settings);
  device = opened;
  deviceOptions = settings;
  deviceTimestampRequest = requested;
  deviceUnavailable = false;
  opened.lost?.then((info: any) => {
    if (device !== opened || job !== active || active.dense?.ended) return;
    deviceUnavailable = true;
    endDense(denseContext(active), {
      failure: denseFailure(
        active.dense!,
        'WebGPU device lost during the dense MVS: ' + info.message + ' [' + info.reason + ']',
        active.lastKernel,
        String(info.reason ?? ''),
        requestContext.snapshot(info.diagnostics ?? opened.diagnostics),
      ),
    });
  });
}

/** What `runDense` and `endDense` need of the worker (`website/src/lib/dense-run.ts`). */
function denseContext(active: NonNullable<typeof job>): DenseRunContext {
  return {
    dense: active.dense!,
    options: { ...LAB_OPTIONS.dense, ...active.denseQuality },
    statistics: () => device.statistics,
    reconstruct: async (scene, images, options) =>
      (await apiPromise!).reconstructDenseGpu(device, scene, images, options),
    // 3.12: the library defaults of the outlier removal, the voxel size and DBSCAN.
    clean: async (points) =>
      (await apiPromise!).cleanDenseCloudGpu(device, points, LAB_OPTIONS.denseCleanOptions),
    // 3.13: the defaults of `meshDenseCloudGpu`.
    mesh: async (points) => (await apiPromise!).meshDenseCloudGpu(device, points),
    post: (message, transfer = []) => self.postMessage({ id: active.id, ...message }, transfer),
    finalFields: () => {
      // A lost or disposed device may refuse the read.
      let chunks: Record<string, unknown> = {};
      try {
        chunks = { ...(device?.chunkRates ?? {}) };
      } catch {
        chunks = {};
      }
      chunks = { ...active.priorDeviceGpu?.chunks, ...chunks };
      return {
        kernels: [...active.kernels],
        chunks,
        ...gpuFields(active),
        ...denseGpuTimestampFields(device),
      };
    },
    lastKernel: () => active.lastKernel,
  };
}

self.onmessage = (event: MessageEvent<Request>) => {
  // The multi-video job (`docs/multi-video.md` 6.2) keeps its own state in `workers/multi-video.ts`.
  if (isMultiRequest(event.data as { type: string })) {
    const multi = event.data as unknown as Parameters<typeof handleMultiVideo>[0];
    lastRequestId = multi.id;
    queue = queue.then(() => handleMultiVideo(multi));
    return;
  }
  const message = event.data;
  // 5.3: a cancel reaches the MVS while `finish` runs it, so it does not wait in the queue.
  if (message.type === 'cancelDense') {
    if (job?.id === message.id) job.dense?.abort.abort();
    cancelMultiDense(message.id);
    return;
  }
  lastRequestId = message.id;
  queue = queue.then(async () => {
    if (message.type === 'start') requestContext = new GpuRequestContext();
    requestContext.begin(message.type, message.type === 'frame' ? message.frame.id : undefined);
    if (job?.id === message.id && message.type === 'frame')
      job.operation = 'processing frame ' + message.frame.id;
    try {
      if (message.type === 'start')
        await start(
          message.id,
          message.focalRatio,
          message.profile ?? 'video',
          message.bounded,
          message.fp64,
          message.budget,
          message.analysisEdge,
          message.firstOctave,
          message.guided,
          message.dense,
          message.clahe,
          message.denseClean,
          message.denseMesh,
          message.gate,
          message.features,
          message.orbTuning,
          message.klt,
          message.kltSetup,
          message.guidedKernel,
          message.focalSearch,
          message.denseFusion,
          message.denseQuality,
          message.denseGpuTimestamps,
        );
      else if (job?.id === message.id) {
        if (message.type === 'frame') await processFrame(message.frame);
        else await finish();
      }
    } catch (error) {
      failJob(message.id, describeError(error), '', undefined, error);
      await releaseFeatures().catch(() => {});
      job = undefined;
      const failed = device;
      device = undefined;
      deviceUnavailable = false;
      await failed?.dispose().catch(() => {});
    } finally {
      requestContext.end();
    }
  });
};
