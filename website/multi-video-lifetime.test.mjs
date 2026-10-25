import test from 'node:test';
import assert from 'node:assert/strict';
import { MultiVideoJob } from './src/lib/multi-video-job.ts';
import { multiVideoQuery } from './src/lib/multi-video.ts';
import { sparsePerformanceRecord, runRecordProblem, denseRecord } from './src/lib/run-record.ts';
import { sampleRecord } from './record-fixture.mjs';

const WIDTH = 16,
  HEIGHT = 16,
  FEATURES = 8;
const identity = () => new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const camera = { fx: 15, fy: 15, cx: 7.5, cy: 7.5 };
const frame = (id) => ({
  id,
  time: id,
  width: WIDTH,
  height: HEIGHT,
  pixels: new Uint8ClampedArray(WIDTH * HEIGHT * 4).fill(id + 5),
  image: `data:image/jpeg;base64,frame${id}`,
  last: true,
  camera,
});
const model = (ids) => ({
  views: ids.map((id) => ({
    id,
    camera,
    pose: { R: identity(), t: new Float64Array([id, 0, 0]) },
  })),
  landmarks: [
    {
      trackId: ids[0],
      xyz: new Float64Array([1, 2, 10]),
      observations: ids.map((viewId) => ({ viewId, featureId: 0, pixel: [2, 2] })),
    },
  ],
});
function graphFor(keyframes, components) {
  const videoIds = [...new Set(keyframes.map((k) => k.videoId))];
  return {
    edges: [],
    components,
    videos: videoIds.map((videoId, component) => ({ videoId, component, tag: 'alone', share: 1 })),
    stats: {
      keyframes: keyframes.length,
      videos: videoIds.length,
      candidatePairs: 0,
      verifiedPairs: 0,
      edges: 0,
      exhaustivePairs: 6,
    },
    timing: { retrievalMs: 0, pairMatchingMs: 0, pairVerificationMs: 0, componentsMs: 0 },
  };
}

async function setup({ dense = true, mode = 'success', nativeFeatures } = {}) {
  const posted = [],
    captures = [],
    featureArrays = [];
  let job;
  const api = {
    async extractSiftGpuResident() {
      const points = new Float64Array(FEATURES * 2).fill(2);
      featureArrays.push(points);
      return { count: FEATURES, readKeypoints: async () => points, dispose: async () => {} };
    },
    async describeKeyframeGpu() {
      const descriptors = new Uint8Array(FEATURES * 128).fill(7);
      const global = new Float32Array(4096).fill(0.25);
      featureArrays.push(descriptors, global);
      return { descriptors, global };
    },
    async reconstructMultiVideoGpu() {
      throw new Error('The consuming worker route was not used');
    },
    async reconstructMultiVideoOwnedGpu(_device, input, options) {
      const featuresBeforeGraphBytes = input.reduce(
        (bytes, row) =>
          bytes + row.keypoints.byteLength + row.descriptors.byteLength + row.global.byteLength,
        0,
      );
      const keyframes = input.map((row, viewId) => ({
        viewId,
        videoId: row.videoId,
        frameKey: row.frameKey,
        time: row.time,
      }));
      // The real owned adapter consumes this independent container, never the worker's records.
      input.length = 0;
      const components =
        mode === 'empty'
          ? []
          : [
              { index: 0, viewIds: [0, 1], videoIds: ['first'], reconstructed: true },
              { index: 1, viewIds: [2, 3], videoIds: ['second'], reconstructed: true },
            ];
      options.onEvent({ stage: 'keyframes', viewIds: keyframes.map((k) => k.viewId) });
      captures.push({ stage: 'beforeGraph', ...job.hostInputs() });
      options.onRelease({ stage: 'descriptors', viewIds: keyframes.map((k) => k.viewId) });
      captures.push({ stage: 'afterGraph', ...job.hostInputs() });
      if (mode === 'lateFailure') {
        // The first partial is immediate; the second waits. Failure must cancel that timer.
        const reconstruction = model([0, 1]);
        options.onEvent({ stage: 'reconstruction', component: 0, registered: 1, reconstruction });
        options.onEvent({ stage: 'reconstruction', component: 0, registered: 2, reconstruction });
        throw new Error('mapper failed');
      }
      const results = components.map((component) => {
        if (mode === 'initializationFailure') {
          options.onRelease({ stage: 'component', viewIds: component.viewIds });
          return { ...component, error: 'could not initialize', ms: 0 };
        }
        const reconstruction = model(component.viewIds);
        options.onEvent({
          stage: 'componentDone',
          component: component.index,
          registered: 2,
          reconstruction,
          meanReprojectionErrorPx: 0,
        });
        options.onRelease({ stage: 'component', viewIds: component.viewIds });
        captures.push({ stage: `afterComponent${component.index}`, ...job.hostInputs() });
        return { ...component, reconstruction, ms: 0, registrationOrder: component.viewIds };
      });
      return {
        keyframes,
        graph: graphFor(keyframes, components),
        components: results,
        canonical: true,
        timing: { reconstructionMs: 0 },
        hostRetention: {
          featuresBeforeGraphBytes,
          featuresAfterGraphBytes: 4 * FEATURES * 16,
          featuresAfterMappingBytes: 0,
          graphMatchesBeforeMappingBytes: 0,
          graphMatchesAfterMappingBytes: 0,
        },
        descriptorCache: {
          hits: 0,
          misses: 0,
          uncached: 0,
          decodes: 0,
          decodedBytes: 0,
          applicationUploadBytes: 0,
          evictions: 0,
          entries: 0,
          residentBytes: 0,
          peakEntries: 0,
          peakResidentBytes: 0,
          limitBytes: 0,
        },
      };
    },
    async reconstructDenseGpu(_device, scene, images, options) {
      captures.push({ stage: `dense${scene.views[0].id}`, ...job.hostInputs() });
      assert.deepEqual(
        [...images.keys()],
        scene.views.map((view) => view.id),
      );
      assert.ok([...images.values()].every((image) => image.rgba.length === WIDTH * HEIGHT * 4));
      assert.equal(
        scene.landmarks[0].observations.length,
        2,
        'original sparse evidence remains intact',
      );
      if (mode === 'denseFailure') throw new Error('dense failed');
      options.signal.throwIfAborted();
      return {
        points: { count: 0 },
        views: scene.views.map(({ id }) => ({
          id,
          status: 'done',
          sources: [],
          depthPixels: 1,
          points: 0,
        })),
      };
    },
  };
  const device = {
    statistics: { liveBytes: 0, pooledBytes: 0 },
    chunkRates: {},
    ...(nativeFeatures ? { info: { webgpu: { features: nativeFeatures } } } : {}),
  };
  job = new MultiVideoJob(
    api,
    device,
    new Float32Array(0),
    {
      query: multiVideoQuery(''),
      maxFeatures: FEATURES,
      firstOctave: 0,
      mapper: {},
      snapshotIntervalMs: 40,
      dense: dense ? { clean: false, mesh: false, options: {}, lastKernel: () => '' } : undefined,
    },
    (message) => posted.push(structuredClone(message)),
  );
  for (const [i, videoId] of ['first', 'second'].entries()) {
    job.video({ videoId, label: videoId, durationSeconds: 2, focalRatio: 1 });
    await job.frame(videoId, frame(i * 2));
    await job.frame(videoId, frame(i * 2 + 1));
    job.endVideo(videoId);
  }
  return { job, posted, captures, featureArrays, device };
}
const featureBytes = 4 * (FEATURES * (16 + 128 + 3) + 4096 * 4);

test('the worker retires features at final readers and preserves all metadata, colors and Dense inputs', async () => {
  const { job, posted, captures, featureArrays, device } = await setup();
  const snapshots = featureArrays.map((array) => array.slice());
  await job.finish();
  assert.equal(captures[0].featureBytes, featureBytes);
  assert.equal(captures[1].featureBytes, 4 * FEATURES * (16 + 3));
  assert.equal(captures[2].featureBytes, 2 * FEATURES * (16 + 3));
  assert.equal(captures[3].featureBytes, 0);
  assert.equal(job.hostInputs().featureBytes, 0);
  assert.equal(job.hostInputs().rgbaBytes, 4 * WIDTH * HEIGHT * 4);
  assert.equal(job.hostInputs().pendingComponents, 2);
  assert.equal(job.hostInputs().pendingObservations, 4);
  const models = posted.filter((message) => message.type === 'model' && !message.partial);
  assert.equal(models.length, 2);
  for (const message of models) {
    assert.ok(
      message.reconstruction.views.every(
        (view) =>
          view.image.startsWith('data:image/jpeg') &&
          typeof view.time === 'number' &&
          typeof view.sequence === 'number' &&
          view.videoId,
      ),
    );
    assert.ok(message.reconstruction.landmarks[0].color.every((channel) => channel > 0));
  }
  await job.runDense(device);
  const [first, second] = captures.filter((capture) => capture.stage.startsWith('dense'));
  assert.equal(first.pendingComponents, 1);
  assert.equal(first.pendingObservations, 2);
  assert.equal(first.activeObservations, 2);
  assert.equal(second.pendingComponents, 0);
  assert.equal(second.activeObservations, 2);
  assert.equal(
    second.rgbaBytes,
    2 * WIDTH * HEIGHT * 4,
    'completed component pixels retire before the next MVS',
  );
  assert.equal(
    second.sparseBytes,
    first.sparseBytes / 2,
    'completed sparse model is no longer owned',
  );
  assert.equal(job.hostInputs().retainedArrayBufferBytes, 0);
  featureArrays.forEach((array, i) => assert.deepEqual(array, snapshots[i]));
});

for (const mode of ['empty', 'initializationFailure', 'denseFailure', 'cancel']) {
  test(`input cleanup and final messages survive ${mode}`, async () => {
    const { job, posted, device } = await setup({ mode });
    await job.finish();
    assert.equal(job.hostInputs().featureBytes, 0);
    if (mode === 'cancel') job.cancelDense();
    await job.runDense(device);
    assert.equal(job.hostInputs().retainedArrayBufferBytes, 0);
    assert.equal(posted.at(-1).type, 'denseDone');
    assert.equal(posted.filter((m) => m.type === 'result').length, 1);
    if (mode === 'cancel')
      assert.ok(posted.filter((m) => m.type === 'denseResult').every((m) => m.cancelled));
  });
}

test('sparse failure cancels delayed partial maps; explicit close drops every input root', async () => {
  const { job, posted } = await setup({ mode: 'lateFailure' });
  await assert.rejects(job.finish(), /mapper failed/);
  const count = posted.length;
  job.releaseInputs();
  assert.equal(job.hostInputs().retainedArrayBufferBytes, 0);
  await new Promise((resolve) => setTimeout(resolve, 80));
  assert.equal(posted.length, count, 'a failed/closed run cannot post a late retained snapshot');
  job.releaseInputs();
});

test('a sparse-only job keeps the same result/model metadata with no Dense roots', async () => {
  const { job, posted } = await setup({ dense: false });
  await job.finish();
  assert.equal(job.hostInputs().retainedArrayBufferBytes, 0);
  assert.equal(posted.at(-1).type, 'result');
  assert.equal(posted.at(-1).summary.videos.length, 2);
});

test('multi-video result measurements survive compact record selection, including measured zeros', async () => {
  const { job, posted } = await setup({ dense: false });
  await job.finish();
  const result = posted.at(-1);
  const measurements = sparsePerformanceRecord(result);
  assert.deepEqual(measurements.hostRetention, result.hostRetention);
  assert.deepEqual(measurements.descriptorCache, result.descriptorCache);
  assert.equal(Object.keys(measurements.hostRetention).length, 7);
  assert.equal(Object.keys(measurements.descriptorCache).length, 12);
  assert.equal(measurements.hostRetention.featuresAfterSfMBytes, 0);
  assert.equal(measurements.descriptorCache.limitBytes, 0);
  const record = JSON.parse(
    JSON.stringify(
      sampleRecord({
        result: {
          ...sampleRecord().result,
          ...measurements,
        },
      }),
    ),
  );
  assert.equal(runRecordProblem(record), undefined);
});

test('each multi-video Dense component reports its actual native timestamp mode', async () => {
  for (const nativeFeatures of [[], ['timestamp-query'], undefined]) {
    const { job, posted, device } = await setup({ nativeFeatures });
    await job.finish();
    await job.runDense(device);
    const expected = nativeFeatures ? nativeFeatures.includes('timestamp-query') : undefined;
    const results = posted.filter((message) => message.type === 'denseResult');
    assert.equal(results.length, 2);
    for (const message of results) {
      assert.equal(message.gpuTimestamps, expected);
      assert.equal(denseRecord(message).options.gpuTimestamps, expected);
    }
  }
});
