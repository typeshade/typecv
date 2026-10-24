const typedFrames = (frames) =>
  frames.map((frame) => ({
    ...frame,
    gray: Float32Array.from(frame.gray),
    keypoints: Float64Array.from(frame.keypoints),
  }));
const report = (result) => ({
  observations: result.newObservations,
  diagnostics: result.diagnostics,
  images: result.images.map((frame) => ({ id: frame.id, keypoints: Array.from(frame.keypoints) })),
  pairs: result.candidatePairs.map((pair) => ({
    first: pair.first,
    second: pair.second,
    matches: Array.from(pair.matches),
  })),
});

export async function runSurfaceCases(api, device, input) {
  const frames = typedFrames(input.frames),
    model = {
      views: input.model.views.map((view) => ({
        ...view,
        pose: {
          R: Float64Array.from(view.pose.R),
          t: Float64Array.from(view.pose.t),
        },
      })),
      landmarks: input.model.landmarks.map((landmark) => ({
        ...landmark,
        xyz: Float64Array.from(landmark.xyz),
      })),
    },
    options = {
      initialModel: model,
      geometryPatchWarp: true,
      maxSeedsPerFrame: 128,
      klt: { levels: 2, halfWindow: 7, epsilonPx: 0.003, maxIterations: 40, nccThreshold: 0.8 },
    };
  const rotation = await api.expandObservationsGpu(device, frames, [], options),
    surface = await api.expandObservationsGpu(device, frames, [], {
      ...options,
      surfacePatchWarp: true,
    });
  return {
    rotation: report(rotation),
    surface: report(surface),
    liveBytes: device.statistics.liveBytes,
  };
}

export async function runConsensusCases(api, device, input) {
  const verifiedPairs = input.verifiedPairs.map((pair) => ({
      ...pair,
      matches: Uint32Array.from(pair.matches),
    })),
    options = {
      verifiedPairs,
      radius: 2,
      multiViewConsistency: true,
      maxConsistencyErrorPx: 0.5,
      klt: { levels: 2, halfWindow: 5, epsilonPx: 0.005, maxIterations: 30 },
    },
    good = await api.expandObservationsGpu(
      device,
      typedFrames(input.frames),
      verifiedPairs,
      options,
    ),
    bad = await api.expandObservationsGpu(
      device,
      typedFrames(input.corrupted),
      verifiedPairs,
      options,
    ),
    displaced = await api.expandObservationsGpu(
      device,
      typedFrames(input.displacedAnchor),
      verifiedPairs,
      options,
    ),
    joined = await api.expandObservationsGpu(
      device,
      typedFrames(input.displacedTarget),
      verifiedPairs,
      options,
    ),
    twoView = await api.expandObservationsGpu(device, typedFrames(input.frames.slice(0, 2)), [], {
      ...options,
      verifiedPairs: [],
    });
  return {
    good: report(good),
    bad: report(bad),
    displaced: report(displaced),
    joined: report(joined),
    twoView: report(twoView),
    liveBytes: device.statistics.liveBytes,
  };
}
