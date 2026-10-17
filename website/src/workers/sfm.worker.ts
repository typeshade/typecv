/// <reference lib="webworker" />
import type { ShadeDeviceOptions } from '../../../src/gpu/device.ts';

type Frame = {
  id: number;
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
  time: number;
  sourceWidth?: number;
  sourceHeight?: number;
};
type SfmRequest = { id: number; frames: Frame[]; focalRatio: number };

type ActiveJob = {
  id: number;
  operation: string;
  lastKernel: string;
  lastUpdate: number;
  deviceLost: boolean;
};

let apiPromise: Promise<any> | undefined;
let device: any;
let deviceUnavailable = false;
let activeJob: ActiveJob | undefined;

async function openDevice() {
  if (device && !deviceUnavailable) return device;
  if (device) {
    const previous = device;
    device = undefined;
    await previous.dispose();
  }
  apiPromise ??= import(/* @vite-ignore */ new URL('/runtime/gpu.js', self.location.origin).href);
  const api = await apiPromise;
  const opened = await api.ShadeDevice.open({
    backend: 'webgpu',
    boundedSubmissions: false,
    solverCheckpoints: true,
    maxModelsPerBatch: 32,
    onProgress: ({ kernel, phase }: { kernel: string; phase: string }) => {
      const job = activeJob;
      if (!job) return;
      job.lastKernel = `${kernel} (${phase})`;
      if (Date.now() - job.lastUpdate < 100) return;
      job.lastUpdate = Date.now();
      self.postMessage({
        id: job.id,
        type: 'progress',
        stage: 'mapping',
        completed: 0,
        total: 1,
        detail: `${job.operation} · ${job.lastKernel}`,
      });
    },
  } satisfies ShadeDeviceOptions);
  device = opened;
  deviceUnavailable = false;
  opened.lost?.then((info: { message: string; reason: string }) => {
    if (device !== opened) return;
    deviceUnavailable = true;
    const job = activeJob;
    if (!job) return;
    job.deviceLost = true;
    self.postMessage({
      id: job.id,
      type: 'error',
      message: `WebGPU device lost during ${job.operation} / ${job.lastKernel} (${info.reason}): ${info.message}`,
    });
  });
  return opened;
}

self.onmessage = async (event: MessageEvent<SfmRequest>) => {
  const { id, frames, focalRatio } = event.data;
  if (activeJob) {
    self.postMessage({ id, type: 'error', message: 'Another SfM job is already running.' });
    return;
  }
  const job: ActiveJob = {
    id,
    operation: 'opening WebGPU',
    lastKernel: '',
    lastUpdate: 0,
    deviceLost: false,
  };
  activeJob = job;
  let response: any;
  let keepDevice = false;
  const transfer: Transferable[] = [];
  const featureHandles = new Set<any>();
  const runStarted = performance.now();
  try {
    self.postMessage({ id, type: 'progress', stage: 'gpu', completed: 0, total: 1 });
    const api = await apiPromiseOrLoad();
    const reusedDevice = !!device && !deviceUnavailable;
    device = await openDevice();
    const beforeRun = device.statistics;
    const images = [],
      pairs: any[] = [],
      detected = new Map<number, any>(),
      featureCounts: number[] = [],
      featureStages: any[] = [];
    const windowSize = 3;
    const pairCount = Array.from({ length: windowSize }, (_, i) =>
      Math.max(0, frames.length - i - 1),
    ).reduce((a, b) => a + b, 0);
    let matchedPairs = 0;
    const prepProgress = () =>
      5 + (70 * (images.length + matchedPairs)) / (frames.length + pairCount);
    for (let index = 0; index < frames.length; index++) {
      if (job.deviceLost) throw new Error('WebGPU device was lost.');
      job.operation = `feature extraction frame ${index + 1}`;
      const frame = frames[index];
      const featureStarted = performance.now();
      const beforeFeatures = device.statistics;
      const features = await api.extractFeaturesGpuResident(
        device,
        frame.pixels,
        frame.width,
        frame.height,
        { threshold: 0.045, maxFeatures: 400 },
      );
      featureHandles.add(features);
      if (features.count < 8)
        throw new Error(`Frame ${index + 1} has only ${features.count} detected features.`);
      const sourceWidth = frame.sourceWidth ?? frame.width;
      const sourceHeight = frame.sourceHeight ?? frame.height;
      const sourceFocal = sourceWidth * focalRatio;
      const fx = (sourceFocal * frame.width) / sourceWidth;
      const fy = (sourceFocal * frame.height) / sourceHeight;
      images.push({
        id: frame.id,
        camera: { fx, fy, cx: (frame.width - 1) / 2, cy: (frame.height - 1) / 2, k1: 0, k2: 0 },
        keypoints: await features.readKeypoints(),
      });
      detected.set(index, features);
      featureCounts.push(features.count);
      featureStages.push({
        operation: 'features',
        frame: frame.id,
        ...features.report,
        elapsedMs: performance.now() - featureStarted,
        applicationUploadBytes:
          device.statistics.applicationUploadBytes - beforeFeatures.applicationUploadBytes,
        applicationReadbackBytes:
          device.statistics.applicationReadbackBytes - beforeFeatures.applicationReadbackBytes,
        residentSnapshotBytes:
          device.statistics.residentSnapshotBytes - beforeFeatures.residentSnapshotBytes,
      });
      self.postMessage({
        id,
        type: 'progress',
        stage: 'features',
        completed: index + 1,
        total: frames.length,
        features: features.count,
        progress: prepProgress(),
      });
      for (let previous = Math.max(0, index - windowSize); previous < index; previous++) {
        if (job.deviceLost) throw new Error('WebGPU device was lost.');
        job.operation = `descriptor matching pair ${matchedPairs + 1}`;
        const matchStarted = performance.now();
        const found = await api.matchResidentFeaturesGpu(device, detected.get(previous), features, {
          ratio: 0.82,
          maxDistance: 80,
        });
        if (found.matches.length >= 8)
          pairs.push({
            first: frames[previous].id,
            second: frame.id,
            matches: found.matches,
            distances: found.distances,
          });
        featureStages.push({
          operation: 'matches',
          first: frames[previous].id,
          second: frame.id,
          ...found.report,
          ...found.execution,
          elapsedMs: performance.now() - matchStarted,
        });
        matchedPairs++;
        self.postMessage({
          id,
          type: 'progress',
          stage: 'matches',
          completed: matchedPairs,
          total: pairCount,
          accepted: pairs.length,
          progress: prepProgress(),
        });
      }
      // Only the previous three descriptors are needed by the temporal matcher.
      const expiredIndex = index - windowSize;
      const expired = detected.get(expiredIndex);
      if (expired) {
        await expired.dispose();
        featureHandles.delete(expired);
        detected.delete(expiredIndex);
      }
    }
    for (const features of detected.values()) {
      await features.dispose();
      featureHandles.delete(features);
    }
    detected.clear();
    // Retain the original step-major geometry order regardless of streaming extraction.
    const frameOrder = new Map(frames.map((frame, index) => [frame.id, index]));
    pairs.sort(
      (a, b) =>
        frameOrder.get(a.second)! -
          frameOrder.get(a.first)! -
          (frameOrder.get(b.second)! - frameOrder.get(b.first)!) ||
        frameOrder.get(a.first)! - frameOrder.get(b.first)!,
    );
    if (!pairs.length)
      throw new Error(
        'No usable feature matches were found. Try a video with textured surfaces and steady camera motion.',
      );
    self.postMessage({
      id,
      type: 'progress',
      stage: 'mapping',
      completed: 0,
      total: 1,
      pairs: pairs.length,
    });
    const result = await api.reconstructIncrementalGpu(device, images, pairs, {
      ransacIterations: 500,
      maxErrorPx: 3,
      minInitialAngleDeg: 1,
      minTriangulationAngleDeg: 0.5,
      bundleIterations: 4,
      finalBundleIterations: 8,
      refineSharedFocal: true,
      onProgress: (event: { operation: string; index: number; phase: string }) => {
        job.operation = `${event.operation} #${event.index}`;
        self.postMessage({
          id,
          type: 'progress',
          stage: 'mapping',
          completed: 0,
          total: 1,
          detail: `${job.operation} (${event.phase})`,
        });
      },
    });
    if (job.deviceLost) throw new Error('WebGPU device was lost.');
    const reconstruction = result.reconstruction;
    for (const view of reconstruction.views) {
      transfer.push(view.pose.R.buffer, view.pose.t.buffer);
    }
    for (const point of reconstruction.landmarks) transfer.push(point.xyz.buffer);
    response = {
      id,
      type: 'result',
      reconstruction,
      unregistered: result.unregistered,
      diagnostics: result.diagnostics,
      featureCounts,
      matchedPairs: pairs.length,
      stages: [
        {
          operation: 'runTotal',
          tier: device.backend,
          reusedDevice,
          elapsedMs: performance.now() - runStarted,
          applicationUploadBytes:
            device.statistics.applicationUploadBytes - beforeRun.applicationUploadBytes,
          applicationReadbackBytes:
            device.statistics.applicationReadbackBytes - beforeRun.applicationReadbackBytes,
          residentSnapshotBytes:
            device.statistics.residentSnapshotBytes - beforeRun.residentSnapshotBytes,
        },
        ...featureStages,
        ...result.execution.stages.map((stage: any) => ({
          operation: stage.operation,
          elapsedMs: stage.elapsedMs,
          tier: stage.execution.report.tier,
          dispatches: stage.execution.report.dispatches,
          submissions: stage.execution.report.submissions,
          applicationUploadBytes: stage.execution.applicationUploadBytes,
          applicationReadbackBytes: stage.execution.applicationReadbackBytes,
          hypothesisBatches: stage.execution.hypothesisBatches,
          hypothesesScored: stage.execution.hypothesesScored,
        })),
      ],
    };
    keepDevice = true;
  } catch (error) {
    response = {
      id,
      type: 'error',
      message: `${job.operation} / ${job.lastKernel}: ${error instanceof Error ? error.message : String(error)}`,
    };
  } finally {
    try {
      for (const features of featureHandles) await features.dispose();
    } finally {
      if (!keepDevice || job.deviceLost) {
        const failed = device;
        device = undefined;
        deviceUnavailable = false;
        await failed?.dispose();
      }
      activeJob = undefined;
    }
  }
  self.postMessage(response, transfer);
};

async function apiPromiseOrLoad() {
  apiPromise ??= import(/* @vite-ignore */ new URL('/runtime/gpu.js', self.location.origin).href);
  return apiPromise;
}
