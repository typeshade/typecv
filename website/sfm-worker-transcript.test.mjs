// Phase 0 of the refactoring programme (`docs/library-structure.md`, section 5, "Lab
// transcripts"; audit `docs/audits/2026-10-26-lab-structure.md`, P0): the transcripts of the
// single-video SfM worker. Each case drives the real worker through `onmessage` with the fake
// runtime of `lab-fake-runtime.mjs` and records every message in and out, every call into the
// runtime (every `session.add` input among them) and every timer. The fixtures are under
// `tests/fixtures/lab-transcripts/`. A structure-only phase leaves them unchanged.
import test from 'node:test';
import assert from 'node:assert/strict';
import { checkBaseline } from './lab-baseline.mjs';
import { OTHER, SCENE, WorkerHarness, bundleWorker, framePixels } from './lab-worker-harness.mjs';

const code = await bundleWorker();
const fixture = (name) =>
  new URL(`../tests/fixtures/lab-transcripts/single-${name}.jsonl`, import.meta.url);

/** The start message as the page writes it (`SfmLab.tsx`, `runSfM`), with `extra` on top. */
const start = (id, extra = {}) => ({
  type: 'start',
  id,
  focalRatio: 0.8,
  focalSearch: false,
  profile: 'video',
  bounded: false,
  fp64: undefined,
  budget: undefined,
  dense: false,
  denseGpuTimestamps: undefined,
  denseClean: false,
  denseFusion: 'tsdf',
  denseQuality: {},
  denseMesh: false,
  gate: 'legacy',
  firstOctave: 0,
  guided: false,
  clahe: false,
  features: 'sift',
  guidedKernel: 'scalar',
  orbTuning: {},
  klt: false,
  kltSetup: undefined,
  analysisEdge: 640,
  ...extra,
});

/** A frame of 128 × 96 pixels at `time` whose content moved by `shift` (spec `shift[id]`). */
const frame = (job, id, time, spec, { source = SCENE, ...extra } = {}) => ({
  type: 'frame',
  id: job,
  frame: {
    id,
    width: 128,
    height: 96,
    time,
    pixels: framePixels(id, spec.shift?.[id] ?? [id * 4, 0], source),
    ...extra,
  },
});

/** The messages of one type that the worker posted. */
const posted = (h, type) => h.outgoing.filter((message) => message.type === type);
const completed = (h, frameId) =>
  posted(h, 'frameComplete').find((message) => message.frameId === frameId);

/**
 * Run a case twice, each time in a new realm, check that both transcripts agree and compare the
 * transcript with its fixture. `check` asserts what the case is for on the first run.
 */
async function transcript(name, spec, script, check) {
  const runs = [];
  for (let i = 0; i < 2; i++) {
    const h = new WorkerHarness(code, spec);
    await script(h);
    if (i === 0) check?.(h);
    runs.push(h.transcript());
  }
  assert.equal(runs[1], runs[0], 'the transcript repeats');
  await checkBaseline(fixture(name), runs[0]);
}

test('phase gate: the first frame, the gaps, a frame that barely moved and a weak peak', async () => {
  const spec = {
    shift: {
      0: [0, 0],
      1: [0, 0],
      2: [2, 0],
      3: [20, 0],
      4: [21, 0],
      5: [30, 0],
      6: [30, 0],
      7: [31, 0],
    },
  };
  await transcript(
    'phase-gate',
    spec,
    async (h) => {
      await h.send(start(1, { gate: 'phase' }));
      await h.send(frame(1, 0, 0, spec));
      await h.send(frame(1, 1, 0.1, spec));
      await h.send(frame(1, 2, 0.5, spec));
      await h.send(frame(1, 3, 0.75, spec));
      // The other scene gives a weak correlation peak: the legacy rule decides.
      await h.send(frame(1, 4, 1, spec, { source: OTHER }));
      await h.send(frame(1, 5, 1.25, spec, { source: OTHER }));
      await h.send(frame(1, 6, 2.4, spec, { source: OTHER }));
      await h.send(frame(1, 7, 2.6, spec, { source: OTHER, last: true }));
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      assert.equal(completed(h, 1).skipped, 'Not a keyframe: minimum gap');
      assert.match(completed(h, 2).skipped, /^Not a keyframe: barely moved, translation/);
      assert.equal(completed(h, 3).keyframe, true);
      assert.match(completed(h, 4).skipped, /^Not a keyframe: median motion 1\.0px/);
      for (const id of [5, 6, 7]) assert.equal(completed(h, id).keyframe, true, `frame ${id}`);
      assert.equal(posted(h, 'result').length, 1);
    },
  );
});

test('legacy gate: motion, the gap rule, pair admission, retention and the window', async () => {
  const spec = {
    window: 3,
    shift: {
      0: [0, 0],
      1: [1, 0],
      2: [6, 0],
      3: [7, 0],
      4: [7.5, 0],
      5: [20, 0],
      6: [26, 0],
      7: [33, 0],
    },
    // The pair (2, 5) has 7 matches, under the 8 that a pair needs.
    matchLimit: { '2:5': 7 },
  };
  await transcript(
    'legacy-gate',
    spec,
    async (h) => {
      await h.send(start(1));
      const times = [0, 0.25, 0.5, 0.75, 1.6, 1.9, 2.2, 2.5];
      for (const [id, time] of times.entries())
        await h.send(frame(1, id, time, spec, id === 7 ? { last: true } : {}));
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      assert.match(completed(h, 1).skipped, /^Not a keyframe: median motion 1\.0px/);
      assert.equal(completed(h, 0).waiting, true);
      assert.equal(completed(h, 2).keyframe, true);
      assert.match(completed(h, 3).skipped, /^Not a keyframe: median motion/);
      assert.equal(completed(h, 4).keyframe, true, 'a frame after the maximum gap is a keyframe');
      for (const id of [5, 6, 7]) assert.equal(completed(h, id).keyframe, true, `frame ${id}`);
    },
  );
});

test('feature minimum: a frame with too few features is rejected and named', async () => {
  const spec = { features: { 1: 5 }, shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0] } };
  await transcript(
    'feature-minimum',
    spec,
    async (h) => {
      await h.send(start(1));
      for (const [id, time] of [0, 0.5, 1].entries()) await h.send(frame(1, id, time, spec));
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      assert.equal(completed(h, 1).skipped, 'Too few detected features');
      const [result] = posted(h, 'result');
      assert.ok(
        result.unregistered.some((u) => u.viewId === 1 && u.reason === 'Too few detected features'),
      );
    },
  );
});

test('guided matching: the guide, a failed guide, the focal search, octave -1 and CLAHE', async () => {
  const spec = {
    shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0], 3: [24, 0] },
    // The second guide (frames 1 and 2) finds no essential matrix.
    essentialFailure: [1],
  };
  await transcript(
    'guided',
    spec,
    async (h) => {
      await h.send(start(1, { guided: true, focalSearch: true, firstOctave: -1, clahe: true }));
      for (const [id, time] of [0, 0.5, 1, 1.5].entries()) await h.send(frame(1, id, time, spec));
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      const [result] = posted(h, 'result');
      assert.ok(result.guided.pairs > 0 && result.guided.addedMatches > 0);
      assert.ok(result.focalSearch);
    },
  );
});

test('device options: bounded submissions, fp64, the time budget and the tiled guided kernel', async () => {
  const spec = { shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0] } };
  await transcript(
    'device-options',
    spec,
    async (h) => {
      await h.send(
        start(1, {
          bounded: true,
          fp64: 'integer',
          budget: 500,
          guided: true,
          guidedKernel: 'tiled',
          analysisEdge: 1280,
        }),
      );
      for (const [id, time] of [0, 0.5, 1].entries()) await h.send(frame(1, id, time, spec));
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      const [result] = posted(h, 'result');
      assert.equal(result.options.boundedSubmissions, true);
      assert.equal(result.options.fp64Flavor, 'integer');
      assert.equal(result.budgetMs, 500);
    },
  );
});

test('KLT: the track gate and tracks matching, the verdict and the tracker record', async () => {
  const spec = {
    defaultFeatures: 80,
    kltTracks: 80,
    shift: { 0: [0, 0], 1: [1, 0], 2: [8, 0], 3: [9, 0] },
  };
  await transcript(
    'klt-gate-tracks',
    spec,
    async (h) => {
      await h.send(
        start(1, {
          klt: true,
          kltSetup: { capacity: 256, fillIn: true, matching: 'tracks', gate: true },
        }),
      );
      for (const [id, time] of [0, 0.25, 0.5, 1.6].entries())
        await h.send(frame(1, id, time, spec));
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      assert.match(completed(h, 1).skipped, /^Not a keyframe: median track displacement 1\.0px/);
      const [result] = posted(h, 'result');
      assert.ok(result.klt && result.klt.trackerOnlyPairs > 0);
    },
  );
});

test('KLT: the join after descriptor matching merges and adds a tracker-only pair', async () => {
  const spec = {
    kltTracks: 40,
    shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0] },
    // The descriptors of (0, 2) give 3 matches; the tracks give a pair of their own.
    matchLimit: { '0:2': 3 },
  };
  await transcript(
    'klt-join',
    spec,
    async (h) => {
      await h.send(start(1, { klt: true }));
      for (const [id, time] of [0, 0.5, 1].entries()) await h.send(frame(1, id, time, spec));
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      const [result] = posted(h, 'result');
      assert.ok(result.klt.pairMatches > 0 && result.klt.duplicateMatches > 0);
      assert.ok(result.klt.conflictingMatches > 0 && result.klt.trackerOnlyPairs > 0);
    },
  );
});

test('ORB: the binary front end and its tuning', async () => {
  const spec = { shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0] } };
  await transcript(
    'orb',
    spec,
    async (h) => {
      await h.send(
        start(1, {
          features: 'orb',
          orbTuning: { maxFeatures: 1000, ratio: 0.9, maxDistance: 64 },
        }),
      );
      for (const [id, time] of [0, 0.5, 1].entries()) await h.send(frame(1, id, time, spec));
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      const [result] = posted(h, 'result');
      assert.equal(result.options.frontEnd, 'corner');
      assert.equal(result.options.features, 'orb');
    },
  );
});

test('precomputed features: calibrated photographs, the 32-entry pair minimum', async () => {
  const spec = { shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0] } };
  const camera = { fx: 110, fy: 110, cx: 63.5, cy: 47.5, k1: -0.01, k2: 0 };
  const keypoints = (id) =>
    Float64Array.from({ length: 48 }, (_, i) => (i % 2 ? 10 + (i % 17) * 4 : 6 + i * 2 + id * 3));
  const pairs = (list) =>
    list.map(([first, n]) => ({
      first,
      matches: Uint32Array.from({ length: 2 * n }, (_, i) => i >> 1),
    }));
  const photo = (id, time, list) =>
    frame(1, id, time, spec, {
      camera,
      sourceWidth: 1280,
      sourceHeight: 960,
      image: `data:image/jpeg;base64,photo${id}`,
      precomputed: { keypoints: keypoints(id), pairs: pairs(list) },
    });
  await transcript(
    'precomputed',
    spec,
    async (h) => {
      await h.send(start(1, { profile: 'photo', focalSearch: true, analysisEdge: 128 }));
      await h.send(photo(0, 0, []));
      await h.send(photo(1, 1, [[0, 20]]));
      // 10 matches are 20 entries, under the 32 of a precomputed pair.
      await h.send(
        photo(2, 2, [
          [0, 10],
          [1, 24],
        ]),
      );
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      const adds = h.events.filter((e) => e.t === 'call' && e.fn === 'session.add');
      assert.equal(adds.length, 3);
      assert.deepEqual(
        adds[2].args[1].map((pair) => pair.first),
        [1],
      );
      assert.equal(posted(h, 'result')[0].options.refineSharedFocal, false);
    },
  );
});

test('dense: the device swap, the fake MVS, the archive, the cleanup and the mesh', async () => {
  const spec = { depthDiagnostics: true, shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0] } };
  await transcript(
    'dense',
    spec,
    async (h) => {
      await h.send(
        start(1, {
          dense: true,
          denseClean: true,
          denseMesh: true,
          denseFusion: 'tsdf',
          denseQuality: { boundaryAware: true, minFiniteGain: 0.5 },
        }),
      );
      for (const [id, time] of [0, 0.5, 1].entries()) await h.send(frame(1, id, time, spec));
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      const [dense] = posted(h, 'denseResult');
      assert.equal(dense.failure, null);
      assert.equal(dense.gpuTimestamps, false);
      assert.ok(posted(h, 'denseArtifact').length >= 2);
      assert.equal(posted(h, 'denseMesh').length, 1);
      assert.equal(posted(h, 'denseClean').length, 1);
    },
  );
});

test('dense: a cancel during the MVS keeps the sparse result', async () => {
  const spec = { denseHold: 'photometric', shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0] } };
  await transcript(
    'dense-cancel',
    spec,
    async (h) => {
      await h.send(start(1, { dense: true, denseClean: true, denseFusion: 'pixel' }));
      for (const [id, time] of [0, 0.5, 1].entries()) await h.send(frame(1, id, time, spec));
      await h.send({ type: 'finish', id: 1 }, { wait: false });
      await h.send({ type: 'cancelDense', id: 1 }, { wait: false });
      await h.release();
    },
    (h) => {
      assert.equal(posted(h, 'result').length, 1);
      const [dense] = posted(h, 'denseResult');
      assert.equal(dense.cancelled, true);
    },
  );
});

test('dense: a device loss during the MVS ends the MVS only', async () => {
  const spec = {
    loss: {
      call: 'reconstructDenseGpu',
      at: 0,
      info: { reason: 'unknown', message: 'dense loss' },
    },
    shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0] },
  };
  await transcript(
    'dense-loss',
    spec,
    async (h) => {
      await h.send(start(1, { dense: true }));
      for (const [id, time] of [0, 0.5, 1].entries()) await h.send(frame(1, id, time, spec));
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      const [dense] = posted(h, 'denseResult');
      assert.match(dense.failure.message, /WebGPU device lost during the dense MVS/);
      assert.equal(posted(h, 'error').length, 0);
    },
  );
});

test('failure: an error in the middle of a frame reports the partial run', async () => {
  const spec = {
    fail: { call: 'matchFloatDescriptorsGpu', at: 1, message: 'synthetic matcher failure' },
    shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0] },
  };
  await transcript(
    'failure-frame',
    spec,
    async (h) => {
      await h.send(start(1));
      for (const [id, time] of [0, 0.5, 1].entries()) await h.send(frame(1, id, time, spec));
    },
    (h) => {
      const [error] = posted(h, 'error');
      assert.match(error.message, /synthetic matcher failure/);
      assert.equal(error.failure.framesProcessed, 3);
      assert.ok(error.partial);
    },
  );
});

test('failure: an initialization failure at the end keeps its pair evidence', async () => {
  const spec = { minInliers: 1000, finishFailure: true, shift: { 0: [0, 0], 1: [8, 0] } };
  await transcript(
    'failure-initialization',
    spec,
    async (h) => {
      await h.send(start(1));
      for (const [id, time] of [0, 0.5].entries()) await h.send(frame(1, id, time, spec));
      await h.send({ type: 'finish', id: 1 });
    },
    (h) => {
      const [error] = posted(h, 'error');
      assert.equal(error.failure.operation, 'initializing 3D map');
      assert.ok(error.failure.initialization.pairs.length > 0);
    },
  );
});

test('loss: a device loss while the worker waits for the next request', async () => {
  const spec = { shift: { 0: [0, 0], 1: [8, 0] } };
  await transcript(
    'loss-idle',
    spec,
    async (h) => {
      await h.send(start(1));
      for (const [id, time] of [0, 0.5].entries()) await h.send(frame(1, id, time, spec));
      await h.lose(0);
    },
    (h) => {
      const [error] = posted(h, 'error');
      assert.equal(error.failure.operation, 'waiting for next request');
      assert.equal(error.failure.context.state, 'idle');
    },
  );
});

test('loss: a device loss during a frame is reported once', async () => {
  const spec = {
    loss: { call: 'session.add', at: 1, reject: 'readback after device loss' },
    shift: { 0: [0, 0], 1: [8, 0] },
  };
  await transcript(
    'loss-active',
    spec,
    async (h) => {
      await h.send(start(1));
      for (const [id, time] of [0, 0.5].entries()) await h.send(frame(1, id, time, spec));
    },
    (h) => {
      const errors = posted(h, 'error');
      assert.equal(errors.length, 1);
      assert.equal(errors[0].failure.reason, 'destroyed');
      assert.equal(errors[0].failure.context.state, 'active');
    },
  );
});

test('device reuse: a second run reuses the device and a run in another mode replaces it', async () => {
  const spec = { shift: { 0: [0, 0], 1: [8, 0] } };
  await transcript(
    'device-reuse',
    spec,
    async (h) => {
      for (const [id, extra] of [
        [1, {}],
        [2, {}],
        [3, { bounded: true }],
      ]) {
        await h.send(start(id, extra));
        for (const [frameId, time] of [0, 0.5].entries())
          await h.send(frame(id, frameId, time, spec));
        await h.send({ type: 'finish', id });
      }
      // The device of the first two runs was replaced: its loss is no longer the worker's.
      await h.lose(0);
    },
    (h) => {
      const totals = posted(h, 'result').map((result) => result.stages[0].reusedDevice);
      assert.deepEqual(totals, [false, true, false]);
      assert.equal(posted(h, 'error').length, 0);
    },
  );
});
