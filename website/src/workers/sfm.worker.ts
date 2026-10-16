/// <reference lib="webworker" />

type Frame = { id: number; width: number; height: number; pixels: Uint8ClampedArray; time: number };
type SfmRequest = { id: number; frames: Frame[]; focalRatio: number };

self.onmessage = async (event: MessageEvent<SfmRequest>) => {
  const { id, frames, focalRatio } = event.data;
  let device: any;
  let response: any;
  let completed = false;
  let operation = 'opening WebGPU';
  let lastKernel = '';
  let lastUpdate = 0;
  const transfer: Transferable[] = [];
  try {
    self.postMessage({ id, type: 'progress', stage: 'gpu', completed: 0, total: 1 });
    const runtimeUrl = new URL('/runtime/gpu.js', self.location.origin).href;
    const api = await import(/* @vite-ignore */ runtimeUrl);
    device = await api.ShadeDevice.open({
      backend: 'webgpu',
      boundedSubmissions: true,
      maxModelsPerBatch: 8,
      onProgress: ({ kernel, phase }: { kernel: string; phase: string }) => {
        lastKernel = `${kernel} (${phase})`;
        if (Date.now() - lastUpdate < 100) return;
        lastUpdate = Date.now();
        self.postMessage({
          id,
          type: 'progress',
          stage: 'mapping',
          completed: 0,
          total: 1,
          detail: `${operation} · ${lastKernel}`,
        });
      },
    });
    let deviceLost = false;
    device.lost?.then((info: { message: string; reason: string }) => {
      if (completed) return;
      deviceLost = true;
      self.postMessage({
        id,
        type: 'error',
        message: `WebGPU device lost during ${operation} / ${lastKernel} (${info.reason}): ${info.message}`,
      });
    });
    const images = [],
      detected = [],
      featureCounts: number[] = [];
    for (let index = 0; index < frames.length; index++) {
      if (deviceLost) throw new Error('WebGPU device was lost.');
      operation = `feature extraction frame ${index + 1}`;
      const frame = frames[index];
      const features = await api.extractFeaturesGpu(
        device,
        frame.pixels,
        frame.width,
        frame.height,
        { threshold: 0.045, maxFeatures: 400 },
      );
      if (features.scores.length < 8)
        throw new Error(`Frame ${index + 1} has only ${features.scores.length} detected features.`);
      const fx = frame.width * focalRatio;
      images.push({
        id: frame.id,
        camera: { fx, fy: fx, cx: (frame.width - 1) / 2, cy: (frame.height - 1) / 2, k1: 0, k2: 0 },
        keypoints: features.keypoints,
      });
      detected.push(features);
      featureCounts.push(features.scores.length);
      self.postMessage({
        id,
        type: 'progress',
        stage: 'features',
        completed: index + 1,
        total: frames.length,
        features: features.scores.length,
      });
    }
    const candidatePairs = [];
    for (let step = 1; step <= 3; step++)
      for (let i = 0; i + step < frames.length; i++) candidatePairs.push([i, i + step]);
    const pairs = [];
    for (let index = 0; index < candidatePairs.length; index++) {
      if (deviceLost) throw new Error('WebGPU device was lost.');
      operation = `descriptor matching pair ${index + 1}`;
      const [a, b] = candidatePairs[index];
      const found = await api.matchFeaturesGpu(device, detected[a], detected[b], {
        ratio: 0.82,
        maxDistance: 80,
      });
      if (found.matches.length >= 8)
        pairs.push({
          first: frames[a].id,
          second: frames[b].id,
          matches: found.matches,
          distances: found.distances,
        });
      self.postMessage({
        id,
        type: 'progress',
        stage: 'matches',
        completed: index + 1,
        total: candidatePairs.length,
        accepted: pairs.length,
      });
    }
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
      onProgress: (event: { operation: string; index: number; phase: string }) => {
        operation = `${event.operation} #${event.index}`;
        self.postMessage({
          id,
          type: 'progress',
          stage: 'mapping',
          completed: 0,
          total: 1,
          detail: `${operation} (${event.phase})`,
        });
      },
    });
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
      stages: result.execution.stages.map((stage: any) => ({
        operation: stage.operation,
        tier: stage.execution.report.tier,
        dispatches: stage.execution.report.dispatches,
        submissions: stage.execution.report.submissions,
      })),
    };
  } catch (error) {
    response = {
      id,
      type: 'error',
      message: `${operation} / ${lastKernel}: ${error instanceof Error ? error.message : String(error)}`,
    };
  } finally {
    completed = true;
    await device?.dispose();
  }
  self.postMessage(response, transfer);
};
