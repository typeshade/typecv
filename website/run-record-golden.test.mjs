// Phase 0 of the refactoring programme (`docs/library-structure.md`, section 5, "Baselines": the
// golden run records): records that today's `buildRunRecord` and `buildMultiRunRecord`
// (`src/components/SfmLab.tsx`) build from fixed inputs, compared byte for byte with
// `tests/fixtures/lab-golden-records/`. The worker messages they read come from the transcript harness
// (`lab-worker-harness.mjs`); the page's part (rows, frames, decode path, note) is written here
// as the page assembles it.
//
// Masked: `createdAt`, the time of the build, becomes `MASKED_CREATED_AT`. Fixed, not masked: the
// `client` block comes from `ENVIRONMENT` (`lab-page-module.mjs`), the `gpu` block from the fake
// adapter, `site.commit` from `COMMIT`, and the timings from the harness clock.
import test from 'node:test';
import assert from 'node:assert/strict';
import { checkBaseline } from './lab-baseline.mjs';
import { loadPageModule } from './lab-page-module.mjs';
import {
  VIDEOS,
  WorkerHarness,
  bundleWorker,
  pageFrame,
  pageMultiStart,
  pageStart,
  pageThreeVideos,
} from './lab-worker-harness.mjs';
import { RUN_RECORD_FORMAT, denseRecord, runRecordProblem } from './src/lib/run-record.ts';
import { multiVideoQuery, partialVideos } from './src/lib/multi-video.ts';
import { advance, newBoard, stageTimings, tick } from './src/lib/multi-video-stages.ts';
import { olderRecord, sampleRecord } from './record-fixture.mjs';

const worker = await bundleWorker();
const page = await loadPageModule();
const fixture = (name) =>
  new URL(`../tests/fixtures/lab-golden-records/${name}.json`, import.meta.url);

export const MASKED_CREATED_AT = '2000-01-01T00:00:00.000Z';
const COMMIT = 'abc1234';
const MAIN_THREAD = {
  longTasks: 2,
  longTaskMs: 180.5,
  longestTaskMs: 120.25,
  longestTaskStage: 'features frame 2',
  frameGaps: 1,
  frameGapMs: 90,
  longestGapMs: 90,
  longestGapStage: 'matching 1 → 2',
};
const [A, B, C] = VIDEOS;
/** The video rows of the multi-video runs, as the page keeps them (`VideoRow`). */
const ROWS = [
  {
    key: 'a',
    label: 'Sample A',
    videoId: A,
    sampleId: 'sample-a',
    focal: 0.8,
    durationSeconds: 2,
    range: { start: 0, end: 2 },
  },
  { key: 'b', label: 'upload.mp4', videoId: B, focal: 0.8, durationSeconds: 2 },
  { key: 'c', label: 'Sample C', videoId: C, sampleId: 'sample-c', focal: 0.8, durationSeconds: 2 },
];

const posted = (h, type) => h.outgoing.filter((message) => message.type === type);

/** The analysed frames that the page collected, as `buildRunRecord` reads them. */
const collected = (count) =>
  Array.from({ length: count }, () => ({
    width: 128,
    height: 96,
    sourceWidth: 1280,
    sourceHeight: 960,
  }));

/**
 * The text of a golden record: valid before and after the mask, format v6, written as JSON with
 * two-space indentation. The mask keeps the position of `createdAt`.
 */
function golden(record) {
  assert.equal(record.format, RUN_RECORD_FORMAT);
  assert.equal(runRecordProblem(record), undefined);
  assert.ok(Number.isFinite(Date.parse(record.createdAt)));
  const masked = { ...record, createdAt: MASKED_CREATED_AT };
  assert.equal(runRecordProblem(masked), undefined);
  return JSON.stringify(masked, null, 2) + '\n';
}

/** Run a single-video case and return the harness after it. */
async function single(spec, startExtra, frames, finish = true) {
  const h = new WorkerHarness(worker, spec);
  await h.send(pageStart(1, startExtra));
  for (const [id, time] of frames.entries()) await h.send(pageFrame(1, id, time, spec));
  if (finish) await h.send({ type: 'finish', id: 1 });
  return h;
}

/** Run a multi-video case of three videos and return the harness after it. */
async function multi(spec, startExtra) {
  const h = new WorkerHarness(worker, spec);
  await h.send(pageMultiStart(1, startExtra));
  await pageThreeVideos(h, 1);
  await h.send({ type: 'multiFinish', id: 1 });
  return h;
}

/** What the page keeps of a multi-video run: the graph, the final models and their errors. */
function multiInputs(h) {
  const models = new Map(),
    meanErrors = new Map();
  for (const message of posted(h, 'model'))
    if (!message.partial && message.reconstruction) {
      models.set(message.component, message.reconstruction);
      meanErrors.set(message.component, message.meanReprojectionErrorPx ?? null);
    }
  return {
    graph: posted(h, 'graph')[0],
    models,
    meanErrors,
    ready: posted(h, 'multiReady')[0],
    fed: posted(h, 'frameComplete').length,
  };
}

test('single-video record: guided matching, KLT and the focal search of an uploaded video', async () => {
  const spec = { kltTracks: 24, shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0], 3: [24, 0] } };
  const h = await single(spec, { guided: true, focalSearch: true, klt: true }, [0, 0.5, 1, 1.5]);
  const [result] = posted(h, 'result');
  const record = page.buildRunRecord({
    result,
    mainThread: MAIN_THREAD,
    adapter: result.adapter,
    commit: COMMIT,
    kind: 'video',
    sampleName: null,
    frames: collected(4),
    durationSeconds: 1.75,
    range: { start: 0, end: 1.75 },
    videoSampleFps: 4,
    videoDecode: 'worker',
    videoSource: undefined,
    benchmark: undefined,
    reference: null,
    note: 'golden single-video record',
  });
  assert.ok(record.result.klt && record.result.guided && record.result.focalSearch);
  await checkBaseline(fixture('single'), golden(record));
});

test('dense record: the dense block of a sample run with its reference comparison', async () => {
  const spec = { depthDiagnostics: true, shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0] } };
  const h = await single(
    spec,
    { dense: true, denseClean: true, denseMesh: true, denseQuality: { boundaryAware: true } },
    [0, 0.5, 1],
  );
  const [result] = posted(h, 'result');
  const [dense] = posted(h, 'denseResult');
  const record = page.buildRunRecord({
    result,
    dense,
    mainThread: MAIN_THREAD,
    adapter: result.adapter,
    commit: COMMIT,
    kind: 'sample',
    sampleName: 'Golden sample',
    frames: collected(3),
    durationSeconds: 1.25,
    videoSampleFps: 4,
    videoDecode: 'main-thread',
    videoSource: 'source',
    reference: {
      name: 'Golden sample',
      status: 'aligned',
      commonViews: 3,
      medianRotationDeg: 0.25,
      maxRotationDeg: 0.5,
      medianCenterOfRadius: 0.01,
      maxCenterOfRadius: 0.02,
    },
    note: 'golden dense record',
  });
  assert.equal(record.dense.status, 'completed');
  await checkBaseline(fixture('dense'), golden(record));
});

test('failed record: the partial outcome and the failure of a run that ended in an error', async () => {
  const spec = {
    fail: { call: 'matchFloatDescriptorsGpu', at: 1, message: 'synthetic matcher failure' },
    shift: { 0: [0, 0], 1: [8, 0], 2: [16, 0] },
  };
  const h = await single(spec, {}, [0, 0.5, 1], false);
  const [failed] = posted(h, 'error');
  // As the page's catch block builds it (`runSfM`).
  const record = page.buildRunRecord({
    result: failed.partial,
    mainThread: MAIN_THREAD,
    failure: failed.failure,
    adapter: failed.partial.adapter,
    commit: COMMIT,
    kind: 'video',
    sampleName: null,
    frames: collected(3),
    durationSeconds: 1,
    videoSampleFps: 4,
    videoDecode: 'worker',
    videoSource: undefined,
    reference: null,
    note: '',
  });
  assert.match(record.failure.message, /synthetic matcher failure/);
  await checkBaseline(fixture('failed'), golden(record));
});

test('multi-video record: components, the view graph and the multi-video options', async () => {
  const spec = { features: { 21: 5 }, multi: { components: [[A, B], [C]], descriptorCache: true } };
  const h = await multi(spec, {});
  const { graph, models, meanErrors, ready, fed } = multiInputs(h);
  const [result] = posted(h, 'result');
  const record = page.buildMultiRunRecord({
    result,
    graph,
    models,
    meanErrors,
    adapter: ready.adapter,
    commit: COMMIT,
    rows: ROWS,
    query: multiVideoQuery(''),
    maxFeatures: ready.maxFeatures,
    firstOctave: 0,
    frames: fed,
    first: { width: 128, height: 96, sourceWidth: 1280, sourceHeight: 960 },
    durationSeconds: 6,
    setName: 'golden-set',
    videoDecode: 'main-thread',
    videoSource: 'mixed',
    videoSampleFps: 4,
    mainThread: MAIN_THREAD,
    note: 'golden multi-video record',
    dense: undefined,
  });
  assert.equal(record.result.components.length, 2);
  await checkBaseline(fixture('multi'), golden(record));
});

test('multi-video dense record: a dense block on each component and their sum', async () => {
  const spec = { multi: { components: [[A, B], [C]] } };
  const h = await multi(spec, {
    dense: true,
    denseClean: undefined,
    denseFusion: 'pixel',
    denseMesh: true,
  });
  const { graph, models, meanErrors, ready, fed } = multiInputs(h);
  const [result] = posted(h, 'result');
  // As the page does on `denseResult` and `denseDone`.
  const blocks = new Map();
  let chunks;
  for (const message of posted(h, 'denseResult')) {
    blocks.set(message.component, denseRecord(message));
    chunks = message.chunks ?? chunks;
  }
  const [done] = posted(h, 'denseDone');
  result.gpuKernels = [...(result.gpuKernels ?? []), ...done.gpuKernels];
  result.gpuDropped = (result.gpuDropped ?? 0) + (done.gpuDropped ?? 0);
  const record = page.buildMultiRunRecord({
    result,
    graph,
    models,
    meanErrors,
    adapter: ready.adapter,
    commit: COMMIT,
    rows: ROWS,
    query: multiVideoQuery(''),
    maxFeatures: ready.maxFeatures,
    firstOctave: 0,
    frames: fed,
    first: { width: 128, height: 96, sourceWidth: 1280, sourceHeight: 960 },
    durationSeconds: 6,
    setName: null,
    videoDecode: 'main-thread',
    videoSource: 'source',
    videoSampleFps: 4,
    mainThread: MAIN_THREAD,
    note: '',
    dense: { blocks, chunks },
  });
  assert.equal(record.result.components.filter((c) => c.dense).length, 2);
  await checkBaseline(fixture('multi-dense'), golden(record));
});

test('multi-video failed record: the partial maps, videos and stage times of a failed run', async () => {
  const spec = { multi: { components: [[A, B], [C]], failure: 'synthetic mapper failure' } };
  const h = await multi(spec, {});
  const { graph, models, meanErrors, fed } = multiInputs(h);
  const [failed] = posted(h, 'error');
  // As the page's catch block builds it (`runMultiVideo`): the stage board from the `multiStage`
  // messages at fixed times, the facts of each video, the maps that existed at the failure.
  let board = newBoard(0);
  for (const [i, message] of posted(h, 'multiStage').entries())
    board = advance(board, message, 100 * (i + 1));
  board = tick(board, 5000);
  const facts = new Map(
    posted(h, 'videoComplete').map((m) => [
      m.videoId,
      { frames: m.frames, width: 128, height: 96, keyframes: m.keyframes },
    ]),
  );
  const partialModels = new Map();
  for (const message of posted(h, 'model'))
    if (message.partial && message.reconstruction)
      partialModels.set(message.component, message.reconstruction);
  const scenes = new Map([...partialModels, ...models]);
  const record = page.buildMultiRunRecord({
    result: undefined,
    partial: {
      stages: stageTimings(board),
      totalMs: 5000 - board.startedAt,
      videos: partialVideos({ rows: ROWS, facts, scenes }),
      scenes,
      gpuKernels: failed.gpuKernels,
      gpuDropped: failed.gpuDropped,
    },
    graph,
    models,
    meanErrors,
    adapter: posted(h, 'multiReady')[0].adapter,
    commit: COMMIT,
    rows: ROWS,
    query: multiVideoQuery(''),
    maxFeatures: 0,
    firstOctave: 0,
    frames: fed,
    first: { width: 128, height: 96, sourceWidth: 1280, sourceHeight: 960 },
    durationSeconds: 6,
    setName: null,
    mainThread: MAIN_THREAD,
    note: '',
    dense: undefined,
    failure: {
      message: String(new Error(failed.message)).slice(0, 2000),
      operation: 'finish',
      kernel: '',
      reason: '',
      framesProcessed: fed,
    },
  });
  assert.equal(record.failure.operation, 'finish');
  await checkBaseline(fixture('multi-failed'), golden(record));
});

test('the v3 to v6 fixtures of the record format stay valid', () => {
  assert.equal(runRecordProblem(sampleRecord()), undefined);
  for (const format of ['typecv-run-record-v4', 'typecv-run-record-v5'])
    assert.equal(runRecordProblem(olderRecord(format)), undefined, format);
  const {
    chunks: _chunks,
    budgetMs: _budget,
    mainThread: _main,
    ...v3Timing
  } = olderRecord('typecv-run-record-v3').timing;
  assert.equal(
    runRecordProblem(olderRecord('typecv-run-record-v3', { timing: v3Timing })),
    undefined,
  );
});
