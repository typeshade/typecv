// Shared CPU/WebGPU assertions are performed by the caller; no source poses enter this adapter.
export async function runExpansionCases(api, device, input) {
  const frames = input.frames.map((f) => ({
    ...f,
    gray: Float32Array.from(f.gray),
    keypoints: Float64Array.from(f.keypoints),
  }));
  const options = { radius: 2, klt: { levels: 3, halfWindow: 5 }, maxSeedsPerFrame: 32 };
  const start = performance.now();
  const expanded = await api.expandObservationsGpu(device, frames, [], options);
  const flat = await api.expandObservationsGpu(
    device,
    [frames[0], { ...frames[1], gray: new Float32Array(frames[1].gray.length).fill(0.5) }],
    [],
    options,
  );
  const masked = await api.expandObservationsGpu(
    device,
    [frames[0], { ...frames[1], validMask: new Uint8Array(frames[1].gray.length) }],
    [],
    options,
  );
  const cut = await api.expandObservationsGpu(
    device,
    frames.map((f, i) => ({ ...f, shotId: i === 1 ? 'B' : 'A' })),
    [],
    options,
  );
  const repeat = input.repeated.map((f) => ({
    ...f,
    gray: Float32Array.from(f.gray),
    keypoints: Float64Array.from(f.keypoints),
  }));
  const ambiguous = await api.expandObservationsGpu(device, repeat, [], {
    ...options,
    ambiguityRadiusPx: 8,
  });
  return {
    expanded: {
      counts: expanded.images.map((f) => f.keypoints.length / 2),
      images: expanded.images.map((f) => ({ id: f.id, keypoints: Array.from(f.keypoints) })),
      pairs: expanded.candidatePairs.map((p) => ({
        first: p.first,
        second: p.second,
        matches: Array.from(p.matches),
      })),
      observations: expanded.newObservations,
      diagnostics: expanded.diagnostics,
    },
    flat: flat.diagnostics,
    masked: masked.diagnostics,
    cut: cut.diagnostics,
    ambiguous: ambiguous.diagnostics,
    original: frames.map((f) => Array.from(f.keypoints)),
    elapsedMs: performance.now() - start,
    liveBytes: device.statistics.liveBytes,
  };
}
