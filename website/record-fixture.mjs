import { RUN_RECORD_FORMAT } from './src/lib/run-record.ts';

/** A valid run record for tests; `overrides` replace top-level fields. */
export function sampleRecord(overrides = {}) {
  return {
    format: RUN_RECORD_FORMAT,
    createdAt: '2026-10-19T06:00:00.000Z',
    site: { commit: 'abc123', runtime: '/runtime/gpu.js' },
    client: {
      userAgent: 'test',
      platform: 'Linux',
      language: 'en',
      hardwareConcurrency: 4,
      deviceMemoryGb: null,
      screen: { width: 1440, height: 900, dpr: 1 },
    },
    gpu: {
      vendor: 'google',
      architecture: 'swiftshader',
      device: '',
      description: '',
      fallbackAdapter: true,
      limits: {
        maxStorageBufferBindingSize: 134217728,
        maxBufferSize: 268435456,
        maxComputeWorkgroupStorageSize: 32768,
        maxComputeInvocationsPerWorkgroup: 256,
        maxStorageBuffersPerShaderStage: 10,
      },
      features: ['float32-filterable', 'subgroups', 'timestamp-query'],
    },
    input: {
      kind: 'sample',
      name: 'Pexels pyramid sample',
      frames: 8,
      width: 640,
      height: 360,
      sourceWidth: 640,
      sourceHeight: 360,
      durationSeconds: 20,
    },
    options: {
      frontEnd: 'corner',
      focalRatio: 1.2,
      windowSize: 32,
      ransacIterations: 500,
      maxErrorPx: 3,
      bundleIterations: 4,
      finalBundleIterations: 8,
      refineSharedFocal: true,
      maxFeatures: 800,
      matchRatio: 0.82,
      maxDistance: 80,
      boundedSubmissions: false,
      fp64Flavor: 'float',
    },
    result: {
      registeredViews: 8,
      landmarks: 726,
      observations: 1895,
      meanReprojectionErrorPx: 0.29,
      unregistered: 0,
      matchedPairs: 18,
      featureCounts: [800, 800, 800, 800, 800, 800, 800, 800],
    },
    timing: {
      totalMs: 2411.9,
      reusedDevice: false,
      stages: {
        pairVerification: {
          calls: 18,
          elapsedMs: 1003.6,
          dispatches: 2412,
          submissions: 72,
          gpuMs: 812.3,
        },
      },
      frameMs: [300, 280, 290, 310, 300, 295, 305, 330],
      kernels: ['pyramidBlur', 'fastCorners', 'pairModels'],
      gpu: [
        { name: 'pyramidBlur', groups: 24, dispatches: 24, timedGroups: 24, gpuMs: 41.25 },
        { name: 'fastCorners+pyramidBlur', groups: 8, dispatches: 8, timedGroups: 8, gpuMs: 12.5 },
        { name: 'pairModels', groups: 2412, dispatches: 2412, timedGroups: 2400, gpuMs: 812.3 },
      ],
      gpuTimestamps: true,
      gpuDropped: 0,
      chunks: {
        scorePairModels: { msPerUnit: 0.41, chunks: 37, longestMs: 212.4, lastUnits: 480 },
        bundleCg: { msPerUnit: 0.0021, chunks: 52, longestMs: 96.2, lastUnits: 41200 },
      },
      budgetMs: 250,
      mainThread: {
        longTasks: 4,
        longTaskMs: 3366.2,
        longestTaskMs: 2851.4,
        longestTaskStage: 'pnp · scorePnP (start)',
        frameGaps: 3,
        frameGapMs: 3362,
        longestGapMs: 2854.1,
        longestGapStage: 'pnp · scorePnP (start)',
      },
    },
    transfers: { applicationUploadBytes: 1234, applicationReadbackBytes: 567 },
    benchmark: { name: 'pyramid-8', pass: 'cold' },
    reference: null,
    failure: null,
    note: '',
    ...overrides,
  };
}

/**
 * The same record as an older format wrote it: v5 has none of the v6 fields (GPU time per kernel,
 * the adapter features, the compute limits, the stage `gpuMs`).
 */
export function olderRecord(format, overrides = {}) {
  const record = sampleRecord();
  const { features: _features, ...gpu } = record.gpu;
  const { maxBufferSize, maxStorageBufferBindingSize } = gpu.limits;
  const { gpu: _gpu, gpuTimestamps: _ts, gpuDropped: _dropped, ...timing } = record.timing;
  const stages = Object.fromEntries(
    Object.entries(timing.stages).map(([name, { gpuMs: _ms, ...stage }]) => [name, stage]),
  );
  return {
    ...record,
    format,
    gpu: { ...gpu, limits: { maxStorageBufferBindingSize, maxBufferSize } },
    timing: { ...timing, stages },
    ...overrides,
  };
}
