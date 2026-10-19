/// <reference lib="webworker" />
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
  | { type: 'start'; id: number; focalRatio: number; profile?: Profile }
  | { type: 'frame'; id: number; frame: Frame }
  | { type: 'finish'; id: number };

/** The lab's fixed numerical controls; reported with every result so a run record can carry them. */
export const LAB_OPTIONS = {
  windowSize: 32,
  ransacIterations: 500,
  maxErrorPx: 3,
  minInitialAngleDeg: 1,
  minTriangulationAngleDeg: 0.5,
  bundleIterations: 4,
  finalBundleIterations: 8,
  refineSharedFocal: true,
  threshold: 0.045,
  maxFeatures: 800,
  smoothDescriptors: true,
  matchRatio: 0.82,
  maxDistance: 80,
  /** A frame becomes a keyframe when its median match displacement from the last keyframe reaches this many analysis pixels... */
  keyframeMinMotionPx: 8,
  /** ...or when this much time has passed since the last keyframe. The first frame is always a keyframe. */
  keyframeMaxGapSeconds: 1,
  /** Map snapshots to the page are posted at most this often; the final one is always posted. */
  snapshotIntervalMs: 500,
  /** Detailed stage entries kept for the page; per-operation totals cover every stage. */
  detailedStageLimit: 400,
} as const;
/**
 * Still photographs with wide baselines: a lower corner threshold, more features and a
 * looser ratio test, because the lab's descriptor matches far fewer features between
 * such views than between video frames (measured on the COLMAP south-building sample:
 * 12 to 43 consecutive-pair matches with the video controls, 75 to 165 with these).
 * The sample ships calibrated intrinsics, so the shared focal is not refined.
 */
export const PHOTO_PROFILE = {
  threshold: 0.02,
  maxFeatures: 2000,
  matchRatio: 0.9,
  maxDistance: 120,
  refineSharedFocal: false,
} as const;
const controls = (profile: Profile) =>
  profile === 'photo' ? { ...LAB_OPTIONS, ...PHOTO_PROFILE } : LAB_OPTIONS;
let apiPromise: Promise<any> | undefined;
let device: any;
let adapter: any;
let deviceUnavailable = false;
let job:
  | {
      id: number;
      focal: number;
      profile: Profile;
      options: ReturnType<typeof controls>;
      session: any;
      started: number;
      before: any;
      reused: boolean;
      features: Map<number, { handle: any; points: Float64Array; frame: Frame }>;
      featureCounts: number[];
      rejections: { viewId: number; reason: string }[];
      matchedPairs: number;
      stages: any[];
      operation: string;
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
        { calls: number; elapsedMs: number; dispatches: number; submissions: number }
      >;
      kernels: Set<string>;
      snapshot: { lastPostedAt: number; pending?: any; timer?: ReturnType<typeof setTimeout> };
    }
  | undefined;
let queue = Promise.resolve();
let lastProgress = 0;

const post = (message: object) => self.postMessage({ id: job!.id, ...message });
function scene(model: any) {
  const active = job!;
  const result = {
    views: model.views.map((view: any) => ({ ...view, ...active.frames.get(view.id) })),
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
    };
  const total = active.summary[entry.operation] ?? {
    calls: 0,
    elapsedMs: 0,
    dispatches: 0,
    submissions: 0,
  };
  active.summary[entry.operation] = {
    calls: total.calls + 1,
    elapsedMs: total.elapsedMs + entry.elapsedMs,
    dispatches: total.dispatches + (entry.dispatches ?? 0),
    submissions: total.submissions + (entry.submissions ?? 0),
  };
  active.stages.push(entry);
  // The detailed list is bounded; the totals above keep the whole run.
  if (active.stages.length > LAB_OPTIONS.detailedStageLimit)
    active.stages.splice(0, active.stages.length - LAB_OPTIONS.detailedStageLimit);
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
  for (const { handle } of job.features.values()) await handle.dispose();
  job.features.clear();
}

async function start(id: number, focal: number, profile: Profile) {
  if (job) throw new Error('Another SfM job is already running.');
  apiPromise ??= import(/* @vite-ignore */ new URL('/runtime/gpu.js', self.location.origin).href);
  const api = await apiPromise;
  const reused = !!device && !deviceUnavailable;
  if (deviceUnavailable) {
    await device?.dispose();
    device = undefined;
    deviceUnavailable = false;
  }
  const started = performance.now();
  // Adapter identity for run records; the ShadeDevice opens its own device.
  let adapterInfo: any = null;
  if (!adapter && navigator.gpu) {
    const requested = await navigator.gpu.requestAdapter();
    if (requested) {
      const info = requested.info ?? {};
      adapter = {
        vendor: info.vendor ?? '',
        architecture: info.architecture ?? '',
        device: info.device ?? '',
        description: info.description ?? '',
        fallbackAdapter: (requested as any).isFallbackAdapter === true,
        limits: {
          maxStorageBufferBindingSize: requested.limits.maxStorageBufferBindingSize,
          maxBufferSize: requested.limits.maxBufferSize,
        },
      };
    }
  }
  adapterInfo = adapter;
  device ??= await api.ShadeDevice.open({
    backend: 'webgpu',
    boundedSubmissions: false,
    solverCheckpoints: true,
    maxModelsPerBatch: 32,
    onProgress: ({ kernel, phase }: { kernel: string; phase: string }) => {
      if (!job) return;
      // The first job compiles each kernel on first use; the count shows the cold start moving.
      const first = !job.kernels.has(kernel);
      job.kernels.add(kernel);
      if (first || performance.now() - lastProgress > 100) {
        lastProgress = performance.now();
        post({
          type: 'progress',
          detail: job.operation + ' · ' + kernel + ' (' + phase + ')',
          kernels: job.kernels.size,
          cold: !job.reused,
        });
      }
    },
  });
  const opened = device;
  const options = controls(profile);
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
    stages: [],
    operation: 'initializing',
    frames: new Map(),
    colors: new Map(),
    pointColors: new Map(),
    references: [],
    skippedFrames: 0,
    frameMs: [],
    summary: {},
    kernels: new Set(),
    snapshot: { lastPostedAt: -Infinity },
    session: new api.IncrementalSfmSession(device, {
      windowSize: options.windowSize,
      ransacIterations: options.ransacIterations,
      maxErrorPx: options.maxErrorPx,
      minInitialAngleDeg: options.minInitialAngleDeg,
      minTriangulationAngleDeg: options.minTriangulationAngleDeg,
      bundleIterations: options.bundleIterations,
      finalBundleIterations: options.finalBundleIterations,
      refineSharedFocal: options.refineSharedFocal,
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
      if (job) post({ type: 'error', message: 'WebGPU device lost: ' + info.message });
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
  active.operation = 'features frame ' + (frame.id + 1);
  const handle = await api.extractFeaturesGpuPyramidResident(
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
  frame.pixels = new Uint8ClampedArray();
  record({
    operation: 'features',
    elapsedMs: performance.now() - started,
    execution: { report: handle.report, ...handle.execution },
  });
  active.featureCounts.push(handle.count);
  post({ type: 'features', frameId: frame.id, time: frame.time, points });
  if (handle.count < 8) {
    await handle.dispose();
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
  for (const other of previous) {
    const matchStarted = performance.now();
    active.operation = 'matching ' + (other.frame.id + 1) + ' → ' + (frame.id + 1);
    const matched = await api.matchResidentFeaturesGpu(device, other.handle, handle, {
      ratio: active.options.matchRatio,
      maxDistance: active.options.maxDistance,
    });
    record({
      operation: 'matches',
      elapsedMs: performance.now() - matchStarted,
      execution: { report: matched.report, ...matched.execution },
    });
    if (other === last && active.lastKeyframe) {
      const motion = medianMotionPx(matched.matches, other.points, other.frame, points, frame),
        gap = frame.time - active.lastKeyframe.time,
        scaledMinimum = (LAB_OPTIONS.keyframeMinMotionPx * frame.width) / 640;
      if (
        matched.matches.length >= 16 &&
        motion < scaledMinimum &&
        gap < LAB_OPTIONS.keyframeMaxGapSeconds
      ) {
        active.features.delete(frame.id);
        await handle.dispose();
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
    if (matched.matches.length >= 16) {
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
  await registerFrame(frame, points, pairs, started);
}

/** A frame whose keypoints and matches another front end produced: no GPU feature or matching stage. */
async function processPrecomputedFrame(frame: Frame, started: number) {
  const active = job!;
  const { keypoints, pairs: given } = frame.precomputed!;
  const points = keypoints;
  sampleColors(frame, points);
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
    if (!other || pair.matches.length < 32) continue;
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

/** Hand the frame and its verified-candidate pairs to the session and report the keyframe. */
async function registerFrame(frame: Frame, points: Float64Array, pairs: any[], started: number) {
  const active = job!;
  active.references = active.references.filter((id) => active.features.has(id));
  const reference = active.features.get(active.references.at(-1) ?? -1);
  if (!reference || frame.time - reference.frame.time >= 0.5) active.references.push(frame.id);
  // Keep a long-baseline first reference, two recent reference images and three adjacent frames.
  if (active.references.length > 3) active.references.splice(1, active.references.length - 3);
  const keepFeatures = new Set([...active.references, ...[...active.features.keys()].slice(-3)]);
  for (const [id, feature] of active.features)
    if (!keepFeatures.has(id) && feature.handle) {
      await feature.handle.dispose();
      active.features.delete(id);
    }
  const sourceWidth = frame.sourceWidth ?? frame.width;
  const sourceHeight = frame.sourceHeight ?? frame.height;
  const sourceFocal = sourceWidth * active.focal;
  const result = await active.session.add(
    {
      id: frame.id,
      camera: frame.camera ?? {
        fx: (sourceFocal * frame.width) / sourceWidth,
        fy: (sourceFocal * frame.height) / sourceHeight,
        cx: (frame.width - 1) / 2,
        cy: (frame.height - 1) / 2,
        k1: 0,
        k2: 0,
      },
      keypoints: points,
    },
    pairs,
  );
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
  if (active.stages.length > LAB_OPTIONS.detailedStageLimit)
    active.stages.splice(0, active.stages.length - LAB_OPTIONS.detailedStageLimit);
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
}

async function finish() {
  const active = job!;
  active.operation = 'final refinement';
  const result = await active.session.finish();
  await releaseFeatures();
  post({
    type: 'result',
    reconstruction: scene(result.reconstruction),
    diagnostics: result.diagnostics,
    unregistered: [...result.unregistered, ...active.rejections],
    featureCounts: active.featureCounts,
    matchedPairs: active.matchedPairs,
    skippedFrames: active.skippedFrames,
    frameMs: active.frameMs,
    summary: active.summary,
    adapter,
    options: { ...active.options, profile: active.profile, focalRatio: active.focal },
    stages: [
      {
        operation: 'runTotal',
        tier: device.backend,
        reusedDevice: active.reused,
        elapsedMs: performance.now() - active.started,
        applicationUploadBytes:
          device.statistics.applicationUploadBytes - active.before.applicationUploadBytes,
        applicationReadbackBytes:
          device.statistics.applicationReadbackBytes - active.before.applicationReadbackBytes,
      },
      ...active.stages,
    ],
  });
  job = undefined;
}

self.onmessage = (event: MessageEvent<Request>) => {
  const message = event.data;
  queue = queue.then(async () => {
    try {
      if (message.type === 'start')
        await start(message.id, message.focalRatio, message.profile ?? 'video');
      else if (job?.id === message.id) {
        if (message.type === 'frame') await processFrame(message.frame);
        else await finish();
      }
    } catch (error) {
      self.postMessage({ id: message.id, type: 'error', message: String(error) });
      await releaseFeatures().catch(() => {});
      job = undefined;
      const failed = device;
      device = undefined;
      deviceUnavailable = false;
      await failed?.dispose().catch(() => {});
    }
  });
};
