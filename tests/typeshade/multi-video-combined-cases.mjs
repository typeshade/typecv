/** Generated-track integration evidence. No video decoding, feature extraction or Dense MVS. */

/** Canonical lossless evidence: every number and typed view retains its original bits. */
export function combinedEvidence(value) {
  if (typeof value === 'number') {
    const words = new Uint32Array(new Float64Array([value]).buffer);
    return { f64: [words[0], words[1]] };
  }
  if (ArrayBuffer.isView(value)) {
    const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    return {
      type: value.constructor.name,
      bytes: Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join(''),
    };
  }
  if (Array.isArray(value)) return value.map(combinedEvidence);
  if (value && typeof value === 'object')
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, combinedEvidence(value[key])]),
    );
  return value;
}

export async function combinedDigest(value) {
  const bytes = new TextEncoder().encode(JSON.stringify(combinedEvidence(value)));
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('');
}

export function combinedInput(serialized) {
  return serialized.map((frame) => ({
    ...frame,
    camera: { ...frame.camera },
    keypoints: Float64Array.from(frame.keypoints),
    descriptors: Uint8Array.from(frame.descriptors),
    global: Float32Array.from(frame.global),
  }));
}

export function combinedGraphSummary(graph) {
  return {
    components: graph.components,
    videos: graph.videos,
    stats: graph.stats,
    edges: graph.edges.map(
      ({
        first,
        second,
        crossVideo,
        inliers,
        homographyInliers,
        medianAngleDeg,
        seedEligible,
        source,
      }) => ({
        first,
        second,
        crossVideo,
        inliers,
        homographyInliers,
        medianAngleDeg,
        seedEligible,
        source,
      }),
    ),
  };
}

/** A shareable aggregate receipt; no source inputs, point coordinates or filesystem paths. */
export function summarizeCombinedReceipt(receipt) {
  const cacheKeys = [
    'hits',
    'misses',
    'uncached',
    'decodes',
    'decodedBytes',
    'applicationUploadBytes',
    'evictions',
    'entries',
    'residentBytes',
    'peakEntries',
    'peakResidentBytes',
    'limitBytes',
  ];
  const counters = ({ kernels, ...values }) => ({
    ...values,
    kernelDispatches: kernels.reduce((n, kernel) => n + kernel.dispatches, 0),
    kernelGroups: kernels.reduce((n, kernel) => n + kernel.groups, 0),
    kernelTimedGroups: kernels.reduce((n, kernel) => n + kernel.timedGroups, 0),
  });
  const row = (run) => {
    const stageTotals = {};
    for (const stage of run.mapperStages) {
      const total = (stageTotals[stage.operation] ??= {
        calls: 0,
        dispatches: 0,
        submissions: 0,
        applicationUploadBytes: 0,
        applicationReadbackBytes: 0,
      });
      total.calls++;
      for (const key of [
        'dispatches',
        'submissions',
        'applicationUploadBytes',
        'applicationReadbackBytes',
      ])
        total[key] += stage[key];
    }
    return {
      ...(run.variant ? { variant: run.variant, pair: run.pair } : {}),
      counts: run.counts,
      digests: run.digests,
      observedWallMs: { mapping: run.mappingMs, explicitObservations: run.observationMs },
      observedGraphStageMs: run.stageMs,
      mapperWork: stageTotals,
      counters: counters(run.counters),
      observationCounters: counters(run.observationCounters),
      sourceContainerLength: run.sourceContainerLength,
      ...(run.hostRetention ? { hostRetention: run.hostRetention } : {}),
      ...(run.descriptorCache
        ? {
            descriptorCache: Object.fromEntries(
              cacheKeys.map((key) => [key, run.descriptorCache[key]]),
            ),
          }
        : {}),
    };
  };
  const adapter = receipt.devices?.candidate;
  return {
    format: 'typecv-combined-multi-video-evidence-v1',
    scope: receipt.scope,
    excludes: receipt.excludes,
    fixture: receipt.fixture,
    modules: receipt.modules,
    environment: {
      backend: receipt.backend,
      tier: adapter?.tier,
      vendor: adapter?.webgpu?.vendor,
      architecture: adapter?.webgpu?.architecture,
      fallback: adapter?.webgpu?.fallback,
    },
    protocol: {
      pairs: receipt.pairs,
      timeoutSeconds: receipt.timeoutSeconds,
      timing: receipt.timing,
      ownedTiming: receipt.ownedTiming,
      hostLifetime: receipt.hostLifetime,
      observationDevice: receipt.observationDevice,
      wallSpeedClaim: false,
    },
    exactEvidenceEqual: receipt.exactEvidenceEqual,
    rows: receipt.rows.map(row),
    owned: row(receipt.owned),
  };
}

/** Both normal and consuming APIs use the same generated arrays, options and callbacks. */
export async function runCombinedPipeline(
  api,
  device,
  serialized,
  options,
  owned = false,
  observations = { api, device },
) {
  const input = combinedInput(serialized);
  // These external test references check bytes only. They are outside the owned lifetime budget.
  const arrays = input.flatMap((frame) => [frame.keypoints, frame.descriptors, frame.global]);
  const inputBefore = await combinedDigest(arrays);
  const callbacks = [],
    stages = [],
    releases = [];
  const before = device.statistics;
  const observationBefore = observations.device.statistics;
  device.resetKernelStatistics();
  if (observations.device !== device) observations.device.resetKernelStatistics();
  const start = performance.now();
  const result = await (owned ? api.reconstructMultiVideoOwnedGpu : api.reconstructMultiVideoGpu)(
    device,
    input,
    {
      ...options,
      mapper: {
        ...options.mapper,
        onStage(stage) {
          const execution = stage.execution;
          stages.push({
            operation: stage.operation,
            elapsedMs: stage.elapsedMs,
            ...(stage.problem ? { problem: stage.problem } : {}),
            dispatches: execution.report.dispatches,
            submissions: execution.report.submissions,
            applicationUploadBytes: execution.applicationUploadBytes,
            applicationReadbackBytes: execution.applicationReadbackBytes,
          });
        },
      },
      onEvent: (event) => callbacks.push(structuredClone(event)),
      ...(owned ? { onRelease: (event) => releases.push(structuredClone(event)) } : {}),
    },
  );
  const mappingMs = performance.now() - start;
  const observationStart = performance.now();
  const evaluated = [];
  for (const component of result.components) {
    if (!component.reconstruction) continue;
    const evaluatedComponent = await observations.api.evaluateObservationsGpu(
      observations.device,
      component.reconstruction,
      3,
      { maxRelativeDepthStdDev: 0.1 },
    );
    const { execution: _execution, ...evidence } = evaluatedComponent;
    evaluated.push({ component: component.index, ...evidence });
  }
  const observationMs = performance.now() - observationStart;
  const after = device.statistics;
  const observationAfter = observations.device.statistics;
  const components = result.components.map(({ ms: _ms, ...component }) => component);
  const identities = result.keyframes.map(({ viewId, videoId, frameKey, time }) => ({
    viewId,
    videoId,
    frameKey,
    time,
  }));
  const graph = result.graph;
  const { timing: _timing, ...completeGraph } = graph;
  return {
    mappingMs,
    observationMs,
    combinedMs: mappingMs + observationMs,
    stageMs: { ...graph.timing, reconstructionMs: result.timing.reconstructionMs },
    mapperStages: stages,
    counts: {
      keyframes: identities.length,
      candidates: graph.stats.candidatePairs,
      verifiedPairs: graph.stats.verifiedPairs,
      edges: graph.edges.length,
      components: components.length,
      registered: components.reduce(
        (sum, component) => sum + (component.reconstruction?.views.length ?? 0),
        0,
      ),
      landmarks: components.reduce(
        (sum, component) => sum + (component.reconstruction?.landmarks.length ?? 0),
        0,
      ),
      observations: components.reduce(
        (sum, component) =>
          sum +
          (component.reconstruction?.landmarks.reduce(
            (n, point) => n + point.observations.length,
            0,
          ) ?? 0),
        0,
      ),
      bundles: stages.filter((stage) => stage.operation === 'bundle').length,
    },
    digests: {
      identities: await combinedDigest(identities),
      graphSummary: await combinedDigest(combinedGraphSummary(graph)),
      ...(owned ? {} : { completeGraph: await combinedDigest(completeGraph) }),
      components: await combinedDigest(components),
      callbacks: await combinedDigest(callbacks),
      observations: await combinedDigest(evaluated),
      inputBefore,
      inputAfter: await combinedDigest(arrays),
    },
    sourceContainerLength: input.length,
    ...(owned ? { hostRetention: result.hostRetention, releases } : {}),
    ...(result.descriptorCache ? { descriptorCache: result.descriptorCache } : {}),
    counters: {
      applicationUploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
      applicationReadbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
      residentSnapshotBytes: after.residentSnapshotBytes - before.residentSnapshotBytes,
      reads: after.reads - before.reads,
      readWaves: after.readWaves - before.readWaves,
      allocations: after.allocations - before.allocations,
      liveBuffers: after.liveBuffers,
      liveBytes: after.liveBytes,
      pooledBytes: after.pooledBytes,
      kernels: after.kernels,
      kernelsDropped: after.kernelsDropped,
    },
    observationCounters: {
      applicationUploadBytes:
        observationAfter.applicationUploadBytes - observationBefore.applicationUploadBytes,
      applicationReadbackBytes:
        observationAfter.applicationReadbackBytes - observationBefore.applicationReadbackBytes,
      residentSnapshotBytes:
        observationAfter.residentSnapshotBytes - observationBefore.residentSnapshotBytes,
      reads: observationAfter.reads - observationBefore.reads,
      readWaves: observationAfter.readWaves - observationBefore.readWaves,
      liveBuffers: observationAfter.liveBuffers,
      liveBytes: observationAfter.liveBytes,
      kernels: observationAfter.kernels,
      kernelsDropped: observationAfter.kernelsDropped,
    },
  };
}
