import { inverseDepthScene } from './inverse-depth-cases.mjs';

const controls = {
  initialPair: [0, 3],
  ransacIterations: 64,
  bundleIterations: 1,
  finalBundleIterations: 1,
};
const keys = (observations) =>
  observations
    .map((o) => `${o.viewId}:${o.featureId}`)
    .sort()
    .join(',');
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const check = (condition, message, details) => {
  if (!condition) throw new Error(message + ': ' + JSON.stringify(details));
};
const mutateCallbackModel = (model) => {
  model.views[0].pose.R[0] += 1;
  model.views[0].camera.fx += 1000;
  if (model.landmarks.length) model.landmarks[0].xyz[0] += 1000;
};

function inputs(model) {
  const images = model.views.map((v) => ({
      id: v.id,
      camera: { ...v.camera },
      keypoints: Float64Array.from(
        model.landmarks.flatMap((p) => p.observations.find((o) => o.viewId === v.id).pixel),
      ),
    })),
    pairs = images.flatMap((b, j) =>
      images.slice(0, j).map((a) => ({
        first: a.id,
        second: b.id,
        matches: Uint32Array.from(model.landmarks.flatMap((_, i) => [i, i])),
      })),
    );
  return { images, pairs };
}

/** Real mapper/matching math; only the later tracker-refinement failure is injected. */
export async function runDepthPublicationCase(api, backend, mode) {
  const model = inverseDepthScene(),
    { images, pairs } = inputs(model),
    device = await api.ShadeDevice.open({ backend, poolBytes: 0 }),
    owned = [];
  let receipt;
  try {
    const native = await api.reconstructIncrementalGpu(device, images, pairs, controls),
      initialSnapshot = structuredClone(native.reconstruction),
      tracker = new api.InverseDepthTracker({
        iterations: 0,
        pixelNoiseStdDevPx: 0.25,
        // Deliberately require more precision than these real observations provide.
        maxRelativeStddev: 1e-12,
      });
    check(
      native.reconstruction.landmarks.length > 0,
      'Native fixture must have finite seeds',
      mode,
    );
    let floatImages;
    if (mode === 'float') {
      const count = model.landmarks.length,
        values = new Float32Array(count * count);
      for (let i = 0; i < count; i++) values[i * count + i] = 1;
      floatImages = images.map((image) => {
        const descriptors = device.allocate(values.length, values);
        owned.push(descriptors);
        return {
          ...image,
          descriptors: { descriptors, count, dimensions: count, elements: values.length },
        };
      });
    }
    const invoke = (onReconstruction) => {
      const options = { ...controls, depthTracking: tracker, onReconstruction };
      if (mode === 'direct') return api.reconstructIncrementalGpu(device, images, pairs, options);
      if (mode === 'refine')
        return api.refineIncrementalGpu(device, images, pairs, native, options);
      if (mode === 'float')
        return api.reconstructIncrementalFloatGpu(device, floatImages, pairs, {
          ...options,
          matching: { ratio: 0.8, maxDistance: 0.001, mutual: true },
        });
      throw new Error('Unknown depth publication case: ' + mode);
    };
    const callbacks = [],
      result = await invoke((m) => {
        callbacks.push(structuredClone(m));
        mutateCallbackModel(m);
      }),
      beforeFailure = tracker.snapshot(),
      failedCallbacks = [],
      originalUpdate = tracker.updateAsync;
    let injectedCalls = 0,
      failure = '';
    tracker.updateAsync = (input) =>
      originalUpdate.call(tracker, input, async () => {
        injectedCalls++;
        throw new Error('injected depth publication refinement failure');
      });
    try {
      await invoke((m) => failedCallbacks.push(structuredClone(m)));
    } catch (error) {
      failure = error.message;
    } finally {
      tracker.updateAsync = originalUpdate;
    }
    receipt = {
      backend,
      mode,
      nativeLandmarks: native.reconstruction.landmarks.length,
      registered: result.reconstruction.views.length,
      admittedLandmarks: result.reconstruction.landmarks.length,
      admittedStates: result.inverseDepth?.diagnostics.finite,
      callbackLandmarks: callbacks.map((m) => m.landmarks.length),
      callbackMatchesResult: callbacks.length === 1 && same(callbacks[0], result.reconstruction),
      initialPreserved: same(native.reconstruction, initialSnapshot),
      failure,
      injectedCalls,
      failedCallbacks: failedCallbacks.length,
      rollbackPreserved: same(tracker.snapshot(), beforeFailure),
      borrowedDescriptorsIntact: owned.length ? (await owned[0].read())[0] === 1 : true,
      liveBorrowedBuffers: device.statistics.liveBuffers,
      ownedBuffers: owned.length,
    };
    return receipt;
  } finally {
    for (const buffer of owned) device.drop(buffer);
    if (receipt) receipt.liveBytes = device.statistics.liveBytes;
    await device.dispose();
  }
}

export function assertDepthPublication(r) {
  check(r.nativeLandmarks > 0 && r.registered === 4, 'Fixture must run real finite mapping', r);
  check(
    r.admittedLandmarks === 0 && r.admittedStates === 0,
    'Strict uncertainty must defer depth',
    r,
  );
  check(r.callbackMatchesResult, 'Publish exactly one admitted result, without native seeds', r);
  check(r.initialPreserved, 'Refinement must preserve the accepted input map', r);
  check(
    /injected depth publication refinement failure/.test(r.failure) && r.injectedCalls === 1,
    'Tracker failure must propagate once',
    r,
  );
  check(
    r.failedCallbacks === 0 && r.rollbackPreserved,
    'Failed update must publish no geometry',
    r,
  );
  check(
    r.borrowedDescriptorsIntact && r.liveBorrowedBuffers === r.ownedBuffers && r.liveBytes === 0,
    'The adapter must preserve borrowed descriptors and release its own arrays',
    r,
  );
}

function matchesAdmission(model, snapshot) {
  if (!snapshot) return false;
  const finite = new Map(
    snapshot.tracks.filter((t) => t.status === 'finite').map((t) => [t.id, t]),
  );
  return (
    finite.size === model.landmarks.length &&
    model.landmarks.every(
      (p) =>
        finite.has(p.trackId) && keys(p.observations) === keys(finite.get(p.trackId).observations),
    )
  );
}

/** Pending id zero forces stable finite ids to differ from a newly numbered MapHistory. */
export async function runDepthSessionPublicationCase(api, backend) {
  const model = inverseDepthScene(10),
    { images } = inputs(model),
    tracker = new api.InverseDepthTracker({ pixelNoiseStdDevPx: 0.25, iterations: 1 }),
    device = await api.ShadeDevice.open({ backend, poolBytes: 0 }),
    callbacks = [];
  let receipt,
    callbackTrackerSnapshotsPreserved = true;
  try {
    const session = new api.IncrementalSfmSession(device, {
      depthTracking: tracker,
      windowSize: 8,
      ransacIterations: 64,
      bundleIterations: 1,
      finalBundleIterations: 1,
      onReconstruction: (m) => {
        const snapshot = tracker.snapshot();
        callbacks.push({ model: structuredClone(m), snapshot });
        mutateCallbackModel(m);
        callbackTrackerSnapshotsPreserved &&= same(tracker.snapshot(), snapshot);
      },
    });
    let result,
      updatesConsistent = true,
      oneCallbackPerUpdate = true,
      callbacksMatchReturned = true;
    for (const image of images) {
      const before = callbacks.length,
        pairs = images
          .filter((v) => session.activeViewIds.includes(v.id))
          .map((v) => ({
            first: v.id,
            second: image.id,
            matches: Uint32Array.from(
              model.landmarks.flatMap((_, i) =>
                i !== 0 || (v.id < 2 && image.id < 2) ? [i, i] : [],
              ),
            ),
          }));
      result = await session.add(image, pairs);
      oneCallbackPerUpdate &&= callbacks.length - before === (result ? 1 : 0);
      if (result) {
        updatesConsistent &&= matchesAdmission(result.reconstruction, result.inverseDepth);
        callbacksMatchReturned &&= same(callbacks.at(-1)?.model, result.reconstruction);
      }
    }
    check(result?.inverseDepth, 'Session must publish a depth snapshot', backend);
    const pending = result.inverseDepth.tracks.find((t) =>
        t.observations.some((o) => o.featureId === 0),
      ),
      finiteBefore = result.inverseDepth.diagnostics.finite,
      archivedViews = result.reconstruction.views.length - session.activeViewIds.length;
    // Revoke a previously finite track, including its archived observations. The mask persists
    // in the shared tracker, so finishing the native session must not resurrect that point.
    tracker.update({
      images,
      tracks: result.tracks,
      reconstruction: result.reconstruction,
      cameraObservationMasks: images.map((v) => ({
        viewId: v.id,
        source: 'manual',
        values: Uint8Array.from({ length: 25 }, (_, i) => (i === 1 ? 0 : 1)),
      })),
    });
    const beforeFinish = callbacks.length,
      finished = await session.finish();
    receipt = {
      backend,
      pendingZero: pending?.id === 0 && pending.status !== 'finite',
      finiteBefore,
      archivedViews,
      updatesConsistent,
      oneCallbackPerUpdate,
      callbacksMatchReturned:
        callbacksMatchReturned && same(callbacks.at(-1)?.model, finished.reconstruction),
      callbackTrackerSnapshotsPreserved,
      callbacksConsistent: callbacks.every((c) => matchesAdmission(c.model, c.snapshot)),
      finishCallbackCount: callbacks.length - beforeFinish,
      finishConsistent: matchesAdmission(finished.reconstruction, finished.inverseDepth),
      finiteAfter: finished.inverseDepth.diagnostics.finite,
      revokedAbsent: !finished.reconstruction.landmarks.some((p) =>
        p.observations.some((o) => o.featureId === 1),
      ),
      rawObservations: finished.inverseDepth.diagnostics.uniqueObservations,
    };
    return receipt;
  } finally {
    if (receipt) receipt.liveBytes = device.statistics.liveBytes;
    await device.dispose();
  }
}

export function assertDepthSessionPublication(r) {
  check(
    r.pendingZero && r.finiteBefore === 24 && r.archivedViews > 0,
    'Exercise pending ids and archived cameras',
    r,
  );
  check(
    r.updatesConsistent && r.callbacksConsistent,
    'Session must preserve admitted persistent ids and observations',
    r,
  );
  check(
    r.callbacksMatchReturned && r.callbackTrackerSnapshotsPreserved,
    'Mutating callback snapshots must not change returned geometry or tracker history',
    r,
  );
  check(
    r.oneCallbackPerUpdate && r.finishCallbackCount === 1,
    'Publish one completed update per callback',
    r,
  );
  check(
    r.finishConsistent && r.finiteAfter === 23 && r.revokedAbsent,
    'Finishing must preserve historical revocation',
    r,
  );
  check(
    r.rawObservations === 232 && r.liveBytes === 0,
    'History must remain deduplicated without leaked arrays',
    r,
  );
}
