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
};
type Request =
  | { type: 'start'; id: number; focalRatio: number }
  | { type: 'frame'; id: number; frame: Frame }
  | { type: 'finish'; id: number };

let apiPromise: Promise<any> | undefined;
let device: any;
let deviceUnavailable = false;
let job:
  | {
      id: number;
      focal: number;
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
  job!.stages.push({
    operation: stage.operation,
    elapsedMs: stage.elapsedMs,
    tier: stage.execution.report.tier,
    dispatches: stage.execution.report.dispatches,
    submissions: stage.execution.report.submissions,
    applicationUploadBytes: stage.execution.applicationUploadBytes,
    applicationReadbackBytes: stage.execution.applicationReadbackBytes,
  });
}
async function releaseFeatures() {
  if (!job) return;
  for (const { handle } of job.features.values()) await handle.dispose();
  job.features.clear();
}

async function start(id: number, focal: number) {
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
  device ??= await api.ShadeDevice.open({
    backend: 'webgpu',
    boundedSubmissions: false,
    solverCheckpoints: true,
    maxModelsPerBatch: 32,
    onProgress: ({ kernel, phase }: { kernel: string; phase: string }) => {
      if (job && performance.now() - lastProgress > 100) {
        lastProgress = performance.now();
        post({ type: 'progress', detail: job.operation + ' · ' + kernel + ' (' + phase + ')' });
      }
    },
  });
  const opened = device;
  job = {
    id,
    focal,
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
    session: new api.IncrementalSfmSession(device, {
      ransacIterations: 500,
      maxErrorPx: 3,
      minInitialAngleDeg: 1,
      minTriangulationAngleDeg: 0.5,
      bundleIterations: 4,
      finalBundleIterations: 8,
      refineSharedFocal: true,
      onStage: record,
      onReconstruction: (model: any) => post({ type: 'snapshot', reconstruction: scene(model) }),
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
  post({ type: 'ready' });
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
  active.operation = 'features frame ' + (frame.id + 1);
  const handle = await api.extractFeaturesGpuPyramidResident(
    device,
    frame.pixels,
    frame.width,
    frame.height,
    { threshold: 0.045, maxFeatures: 800, smoothDescriptors: true },
  );
  const points = await handle.readKeypoints();
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
      await feature.handle.dispose();
      active.features.delete(id);
    }
  const previous = [...active.features.values()].sort((a, b) => a.frame.id - b.frame.id);
  active.features.set(frame.id, { handle, points, frame });
  for (const other of previous) {
    const matchStarted = performance.now();
    active.operation = 'matching ' + (other.frame.id + 1) + ' → ' + (frame.id + 1);
    const matched = await api.matchResidentFeaturesGpu(device, other.handle, handle, {
      ratio: 0.82,
      maxDistance: 80,
    });
    record({
      operation: 'matches',
      elapsedMs: performance.now() - matchStarted,
      execution: { report: matched.report, ...matched.execution },
    });
    if (matched.matches.length >= 16) {
      pairs.push({
        first: other.frame.id,
        second: frame.id,
        matches: matched.matches,
        distances: matched.distances,
      });
      active.matchedPairs++;
    }
    if (other === previous.at(-1)) {
      const lines: number[] = [];
      for (let i = 0; i < Math.min(matched.matches.length, 160); i += 2) {
        const a = matched.matches[i] * 2,
          b = matched.matches[i + 1] * 2;
        lines.push(
          (other.points[a] * frame.width) / other.frame.width,
          (other.points[a + 1] * frame.height) / other.frame.height,
          points[b],
          points[b + 1],
        );
      }
      post({ type: 'matches', frameId: frame.id, lines, count: matched.matches.length / 2 });
    }
  }
  active.references = active.references.filter((id) => active.features.has(id));
  const reference = active.features.get(active.references.at(-1) ?? -1);
  if (!reference || frame.time - reference.frame.time >= 0.5) active.references.push(frame.id);
  // Keep a long-baseline first reference, two recent reference images and three adjacent frames.
  if (active.references.length > 3) active.references.splice(1, active.references.length - 3);
  const keepFeatures = new Set([...active.references, ...[...active.features.keys()].slice(-3)]);
  for (const [id, feature] of active.features)
    if (!keepFeatures.has(id)) {
      await feature.handle.dispose();
      active.features.delete(id);
    }
  const sourceWidth = frame.sourceWidth ?? frame.width;
  const sourceHeight = frame.sourceHeight ?? frame.height;
  const sourceFocal = sourceWidth * active.focal;
  const result = await active.session.add(
    {
      id: frame.id,
      camera: {
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
  if (result) post({ type: 'snapshot', reconstruction: scene(result.reconstruction) });
  active.stages.push({
    operation: 'keyframe',
    frame: frame.id,
    elapsedMs: performance.now() - started,
  });
  post({
    type: 'frameComplete',
    frameId: frame.id,
    time: frame.time,
    elapsedMs: performance.now() - started,
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
      if (message.type === 'start') await start(message.id, message.focalRatio);
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
