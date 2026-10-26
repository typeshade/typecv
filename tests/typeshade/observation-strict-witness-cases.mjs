// The Node fixture supplies exact grayscale samples; this adapter also runs in a browser.
export async function runStrictWitnessCases(api, device, input, { onCase } = {}) {
  const started = performance.now(),
    inputSnapshot = JSON.stringify(input),
    cases = {};
  const equal = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);
  for (const [name, entry] of Object.entries(input.cases)) {
    const frames = entry.keypoints.map((keypoints, i) => ({
        ...input.frames[i],
        camera: { ...input.frames[i].camera },
        gray: Float32Array.from(input.frames[i].gray),
        keypoints: Float64Array.from(keypoints),
      })),
      pairs = entry.pairs.map((pair) => ({
        ...pair,
        matches: Uint32Array.from(pair.matches),
        ...(pair.distances ? { distances: Float64Array.from(pair.distances) } : {}),
      })),
      originalFrames = frames.map((frame) => ({
        gray: frame.gray.slice(),
        keypoints: frame.keypoints.slice(),
        metadata: JSON.stringify({
          id: frame.id,
          width: frame.width,
          height: frame.height,
          camera: frame.camera,
        }),
      })),
      originalPairs = pairs.map((pair) => ({
        first: pair.first,
        second: pair.second,
        matches: pair.matches.slice(),
        distances: pair.distances?.slice(),
      })),
      options = {
        radius: entry.radius ?? 3,
        maxSeedsPerFrame: entry.maxSeedsPerFrame ?? 16,
        klt: { levels: 2, halfWindow: 5 },
        ...(entry.mode === undefined ? {} : { multiViewConsistency: entry.mode }),
        ...(entry.verified ? { verifiedPairs: pairs } : {}),
      },
      result = await api.expandObservationsGpu(device, frames, pairs, options),
      inputsUnchanged =
        frames.length === originalFrames.length &&
        frames.every(
          (frame, i) =>
            equal(frame.gray, originalFrames[i].gray) &&
            equal(frame.keypoints, originalFrames[i].keypoints) &&
            JSON.stringify({
              id: frame.id,
              width: frame.width,
              height: frame.height,
              camera: frame.camera,
            }) === originalFrames[i].metadata,
        ) &&
        pairs.length === originalPairs.length &&
        pairs.every(
          (pair, i) =>
            pair.first === originalPairs[i].first &&
            pair.second === originalPairs[i].second &&
            equal(pair.matches, originalPairs[i].matches) &&
            (pair.distances === undefined
              ? originalPairs[i].distances === undefined
              : originalPairs[i].distances !== undefined &&
                equal(pair.distances, originalPairs[i].distances)),
        );
    cases[name] = {
      images: result.images.map((frame) => ({
        id: frame.id,
        keypoints: Array.from(frame.keypoints),
      })),
      pairs: result.candidatePairs.map((pair) => ({
        first: pair.first,
        second: pair.second,
        matches: Array.from(pair.matches),
        ...(pair.distances ? { distances: Array.from(pair.distances) } : {}),
      })),
      newEdges: result.newEdges,
      newObservations: result.newObservations,
      diagnostics: result.diagnostics,
      liveBytes: device.statistics.liveBytes,
      inputsUnchanged,
    };
    await onCase?.(name, cases[name]);
  }
  return {
    cases,
    inputsUnchanged:
      JSON.stringify(input) === inputSnapshot &&
      Object.values(cases).every((entry) => entry.inputsUnchanged),
    liveBytes: device.statistics.liveBytes,
    elapsedMs: performance.now() - started,
  };
}
