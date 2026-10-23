// Verifies: docs/dense-mvs.md step 6 (section 5.3, the worker's dense MVS), with a fake MVS in
// place of `reconstructDenseGpu`.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DENSE_LAB_FUSION,
  DENSE_LAB_OPTIONS,
  blockColors,
  createDenseJob,
  denseCleanDefault,
  denseFailure,
  denseWorkingBytes,
  endDense,
  keepFrame,
  runDense,
} from './src/lib/dense-run.ts';

const OPTIONS = {
  maxSources: 6,
  bestSources: 3,
  windowRadius: 5,
  windowStep: 2,
  photometricIterations: 8,
  geometricIterations: 4,
  minConsistentViews: 3,
};
const W = 8,
  H = 4;

test('the lab applies planar completion only to TSDF and records the completed pixel count', async () => {
  for (const fusion of ['tsdf', 'pixel']) {
    const dense = createDenseJob(false, false, fusion);
    keepFrame(dense, frame(0));
    let asked;
    const worker = context(dense, async (_scene, _images, options) => {
      asked = options;
      return {
        points: { count: 0 },
        views: [],
        ...(options.planarDepth ? { planar: { filled: 36 } } : {}),
      };
    });
    worker.options = DENSE_LAB_OPTIONS;
    await runDense(worker, { views: [view(0)], landmarks: [] });
    assert.equal(asked.planarDepth, fusion === 'tsdf');
    assert.equal(asked.tsdf.visibility, 'view-weighted');
    const result = worker.posted.at(-1).message;
    assert.equal(result.options.planarDepth, fusion === 'tsdf');
    assert.equal(result.planarFilled, fusion === 'tsdf' ? 36 : undefined);
    assert.equal(result.failure, null);
  }
});
const frame = (id) => ({ id, width: W, height: H, pixels: new Uint8ClampedArray(4 * W * H) });
const view = (id) => ({
  id,
  pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [id, 0, 0] },
  camera: { fx: 10, fy: 10, cx: 4, cy: 2 },
  image: 'data:image/jpeg;base64,x',
});

/** The plan a fake MVS reports (Amendment 3, `planMvsResidency`). */
const RESIDENCY = {
  budgetBytes: 1 << 20,
  bindingBytes: 1 << 20,
  photometric: { mode: 'window', slots: 7, elements: 0, bytes: 3000, largestArray: 900 },
  geometric: { mode: 'window', slots: 7, elements: 0, bytes: 4100, largestArray: 900 },
  fusion: { mode: 'streamed', slots: 3, chunk: 64, bytes: 3500, largestArray: 400 },
  peakBytes: 4100,
};

/** A worker context around a fake MVS: `mvs(scene, images, options)` plays the run. */
function context(dense, mvs) {
  const posted = [];
  const artifacts = [];
  let bytes = { liveBytes: 1000, pooledBytes: 400 };
  return {
    posted,
    artifacts,
    setBytes: (next) => (bytes = next),
    dense,
    options: OPTIONS,
    statistics: () => bytes,
    reconstruct: mvs,
    post: (message, transfer = []) =>
      (message.type === 'denseArtifact' ? artifacts : posted).push({ message, transfer }),
    finalFields: () => ({ kernels: ['mvsPropagate'], chunks: { mvsPropagate: { chunks: 8 } } }),
    lastKernel: () => 'mvsPropagate',
  };
}

/** A fake MVS over the scene's views: the progress of 5.3, then the depth and point events. */
function fakeMvs(run, { failAt, waitAt } = {}) {
  return async (scene, images, options) => {
    run.scene = scene;
    run.images = images;
    run.options = options;
    const ids = scene.views.map((v) => v.id);
    const N = ids.length;
    // Amendment 5: a depth preview of each reference before its progress event, in the
    // photometric and the geometric stage.
    const preview = (name) => async (v) =>
      options.onView({
        type: 'depth',
        stage: name,
        viewId: ids[v],
        width: 2,
        height: 1,
        depth: new Float32Array([1, 0]),
      });
    const stage = async (name, each) => {
      for (let v = 0; v < N; v++) {
        if (failAt === name + ' ' + v) throw new Error('kernel failed');
        if (waitAt === name + ' ' + v)
          await new Promise((_, reject) =>
            options.signal.addEventListener('abort', () => reject(options.signal.reason)),
          );
        options.signal.throwIfAborted();
        await each?.(v);
        options.onProgress({ stage: name, viewId: ids[v], done: v + 1, total: N });
      }
    };
    run.setBytes({ liveBytes: 3500, pooledBytes: 400 });
    await stage('photometric', preview('photometric'));
    run.setBytes({ liveBytes: 5000, pooledBytes: 400 });
    await stage('geometric', preview('geometric'));
    run.setBytes({ liveBytes: 4000, pooledBytes: 400 });
    await stage('fusion', (v) =>
      options.onView({
        type: 'points',
        viewId: ids[v],
        count: v + 1,
        positions: new Float32Array(3 * (v + 1)),
        colors: new Uint8Array(3 * (v + 1)),
        normals: new Int16Array(2 * (v + 1)),
      }),
    );
    return {
      points: { count: (N * (N + 1)) / 2 },
      residency: RESIDENCY,
      views: ids.map((id, v) => ({
        id,
        status: 'done',
        sources: ids.filter((other) => other !== id),
        depthPixels: 10,
        firstPoint: 0,
        points: v + 1,
      })),
    };
  };
}

test('the kept RGBA has no view limit, and the budget comes from the adapter (Amendment 3)', () => {
  const dense = createDenseJob();
  // More keyframes than any all-resident layout held: 61 was the most at 640 × 427 in the lab.
  for (let id = 0; id < 80; id++) keepFrame(dense, frame(id));
  assert.equal(dense.rgba.size, 80);
  assert.equal(dense.rgba.get(1).width, W);
  assert.equal(dense.skipped, '');
  assert.equal('limit' in dense, false);
  // Half of the device's `maxBufferSize` when that is larger than the SfM's budget.
  const MiB = 1024 ** 2;
  assert.equal(denseWorkingBytes(256 * MiB, 2048 * MiB), 1024 * MiB);
  assert.equal(denseWorkingBytes(256 * MiB, 256 * MiB), 256 * MiB);
  assert.equal(denseWorkingBytes(64 * MiB, 256 * MiB), 128 * MiB);
  assert.equal(denseWorkingBytes(64 * MiB, undefined), 64 * MiB);
});

test('a run with more keyframes than the all-resident layout held runs the MVS on all of them', async () => {
  // 25 views, as in the owner's run of 2026-10-21 that the old limit of 17 skipped.
  const ids = Array.from({ length: 25 }, (_, id) => id);
  const dense = createDenseJob();
  for (const id of ids) keepFrame(dense, frame(id));
  const run = {};
  const worker = context(dense, fakeMvs(run));
  run.setBytes = worker.setBytes;
  await runDense(worker, { views: ids.map(view), landmarks: [{ xyz: [0, 0, 1] }] });
  const result = worker.posted.at(-1).message;
  assert.equal(result.type, 'denseResult');
  assert.equal(result.skipped, '');
  assert.equal(run.images.size, 25);
  assert.equal(result.points, (25 * 26) / 2);
  assert.deepEqual(result.residency, RESIDENCY);
});

test('a dense run posts progress, depth maps, points and one result, with arrays transferred', async () => {
  const dense = createDenseJob();
  for (const id of [0, 1, 2]) keepFrame(dense, frame(id));
  const run = {};
  const worker = context(dense, fakeMvs(run));
  run.setBytes = worker.setBytes;
  // View 5 registered without a kept image, and view 2 kept an image without registering.
  const model = { views: [view(0), view(1), view(5)], landmarks: [{ xyz: [0, 0, 1] }] };
  await runDense(worker, model);
  // Only registered views with an image reach the MVS, as `MvsScene` views.
  assert.deepEqual(
    run.scene.views.map((v) => Object.keys(v)),
    [
      ['id', 'pose', 'camera'],
      ['id', 'pose', 'camera'],
    ],
  );
  assert.deepEqual([...run.images.keys()], [0, 1]);
  assert.equal(run.scene.landmarks, model.landmarks);
  assert.equal(run.options.windowRadius, 5);
  // docs/dense-mvs.md 10.4 (step 3 record): the lab's default fusion is the TSDF fusion.
  assert.equal(run.options.fusion, 'tsdf');
  assert.equal(run.options.signal, dense.abort.signal);
  const numerical = worker.posted.filter(({ message }) => message.type !== 'denseActivity');
  const activity = worker.posted.filter(({ message }) => message.type === 'denseActivity');
  assert.equal(activity[0].message.phase, 'photometric');
  assert.equal(activity[0].message.done, 0);
  assert.equal(activity.at(-1).message.phase, 'fusion');
  assert.equal(activity.at(-1).message.done, 0);
  assert.ok(
    activity.some(({ message }) => message.phase === 'geometric' && message.done === 0),
    'the next depth stage is visible before its first completed reference',
  );
  const types = numerical.map(({ message }) => message.type);
  assert.deepEqual(types, [
    'denseDepth',
    'denseProgress',
    'denseDepth',
    'denseProgress',
    'denseDepth',
    'denseProgress',
    'denseDepth',
    'denseProgress',
    'densePoints',
    'denseProgress',
    'densePoints',
    'denseProgress',
    'denseResult',
  ]);
  assert.deepEqual(numerical[1].message, {
    type: 'denseProgress',
    stage: 'photometric',
    viewId: 0,
    done: 1,
    total: 2,
  });
  // Amendment 5, rule 3: the stage and the block colors of the kept image, transferred.
  const depth = numerical[0];
  assert.deepEqual(Object.keys(depth.message), [
    'type',
    'viewId',
    'stage',
    'width',
    'height',
    'depth',
    'rgb',
  ]);
  assert.equal(depth.message.stage, 'photometric');
  assert.equal(depth.message.rgb.length, 2 * 1 * 3);
  assert.deepEqual(depth.transfer, [depth.message.depth.buffer, depth.message.rgb.buffer]);
  assert.equal(numerical[4].message.stage, 'geometric');
  const points = numerical[10];
  assert.deepEqual(Object.keys(points.message), [
    'type',
    'viewId',
    'positions',
    'colors',
    'normals',
  ]);
  assert.equal(points.message.viewId, 1);
  assert.deepEqual(points.transfer, [
    points.message.positions.buffer,
    points.message.colors.buffer,
    points.message.normals.buffer,
  ]);
  const result = worker.posted.at(-1).message;
  assert.equal(result.cancelled, false);
  assert.equal(result.failure, null);
  assert.equal(result.skipped, '');
  assert.equal(result.points, 3);
  assert.deepEqual(
    result.views.map((v) => [v.id, v.status, v.points]),
    [
      [0, 'done', 1],
      [1, 'done', 2],
    ],
  );
  assert.deepEqual(result.options, { ...OPTIONS, fusion: 'tsdf', denseClean: true });
  assert.deepEqual(result.kernels, ['mvsPropagate']);
  // 4.3, step 6: the device's bytes before the MVS, at the depth stage and at the fusion stage.
  assert.deepEqual(result.memory, {
    liveBytesBefore: 1000,
    pooledBytesBefore: 400,
    depthStageBytes: 4600,
    // Amendment 3: the plan's bytes of the depth stage, the larger of its two passes.
    depthStageFormulaBytes: 4100,
    fusionStageBytes: 3600,
  });
  assert.deepEqual(Object.keys(result.timing.stageMs), ['photometric', 'geometric', 'fusion']);
  assert.ok(result.timing.totalMs >= 0);
  // The kept images are released at the end.
  assert.equal(dense.rgba.size, 0);
  assert.equal(dense.running, false);
});

test('a cancel ends the MVS with cancelled: true and the points posted before it', async () => {
  const dense = createDenseJob();
  for (const id of [0, 1]) keepFrame(dense, frame(id));
  const run = {};
  const worker = context(dense, fakeMvs(run, { waitAt: 'fusion 1' }));
  run.setBytes = worker.setBytes;
  const done = runDense(worker, { views: [view(0), view(1)], landmarks: [] });
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(dense.running, true);
  // The worker's `cancelDense` handler aborts the job's controller.
  dense.abort.abort();
  await done;
  const results = worker.posted.filter(({ message }) => message.type === 'denseResult');
  assert.equal(results.length, 1);
  const result = results[0].message;
  assert.equal(result.cancelled, true);
  assert.equal(result.failure, null);
  assert.equal(result.points, 1);
  assert.deepEqual(result.views, []);
  assert.equal(result.memory.liveBytesBefore, 1000);
  assert.ok(result.timing.stageMs.photometric >= 0);
  assert.equal(result.timing.stageMs.fusion, 0);
});

test('a failure names its stage and kernel; one from outside the run ends it once', async () => {
  const dense = createDenseJob();
  for (const id of [0, 1]) keepFrame(dense, frame(id));
  const run = {};
  const worker = context(dense, fakeMvs(run, { failAt: 'geometric 1' }));
  run.setBytes = worker.setBytes;
  await runDense(worker, { views: [view(0), view(1)], landmarks: [] });
  const failed = worker.posted.at(-1).message;
  assert.equal(failed.type, 'denseResult');
  assert.equal(failed.cancelled, false);
  assert.equal(failed.failure.stage, 'geometric');
  assert.equal(failed.failure.kernel, 'mvsPropagate');
  assert.match(failed.failure.message, /kernel failed/);

  // A device loss posts the result at once and stops the run, whose rejection posts nothing.
  const lost = createDenseJob();
  for (const id of [0, 1]) keepFrame(lost, frame(id));
  const lostRun = {};
  const lostWorker = context(lost, fakeMvs(lostRun, { waitAt: 'photometric 0' }));
  lostRun.setBytes = lostWorker.setBytes;
  const pending = runDense(lostWorker, { views: [view(0), view(1)], landmarks: [] });
  await new Promise((resolve) => setTimeout(resolve, 10));
  endDense(lostWorker, {
    failure: denseFailure(lost, 'WebGPU device lost', 'mvsInitialize', 'unknown'),
  });
  await pending;
  const results = lostWorker.posted.filter(({ message }) => message.type === 'denseResult');
  assert.equal(results.length, 1);
  assert.deepEqual(results[0].message.failure, {
    message: 'WebGPU device lost',
    stage: 'photometric',
    kernel: 'mvsInitialize',
    reason: 'unknown',
  });
  assert.equal(lost.abort.signal.aborted, true);
});

test('a run with no registered view that kept its image is skipped with its reason', async () => {
  const dense = createDenseJob();
  for (const id of [1, 2]) keepFrame(dense, frame(id));
  const worker = context(dense, async () => assert.fail('no MVS without images'));
  await runDense(worker, { views: [view(0)], landmarks: [] });
  assert.equal(worker.posted.length, 1);
  const result = worker.posted[0].message;
  assert.equal(result.type, 'denseResult');
  assert.equal(result.skipped, 'No registered view kept its image');
  assert.equal(result.memory.depthStageBytes, null);
});

/** A fake MVS that returns real arrays, for the cleanup of 3.12. */
function cloudMvs(count) {
  return async () => ({
    points: {
      count,
      positions: new Float32Array(3 * count),
      colors: new Uint8Array(3 * count),
      normals: new Int16Array(2 * count),
    },
    views: [{ id: 0, status: 'done', sources: [1], depthPixels: 1, firstPoint: 0, points: count }],
  });
}
const cleanReport = {
  tier: 'cpu',
  before: 10,
  afterOutliers: 9,
  afterVoxel: 6,
  after: 5,
  outlierThreshold: 1.5,
  voxelSize: 0.1,
  eps: 0.4,
  clusters: 2,
  keptClusters: 1,
  ms: { outliers: 1, voxel: 2, cluster: 3 },
};

test('the cleanup follows the fusion, replaces the cloud and records its counts (3.12)', async () => {
  const dense = createDenseJob();
  keepFrame(dense, frame(0));
  const worker = context(dense, cloudMvs(10));
  const seen = [];
  worker.clean = async (points) => {
    seen.push(points.count);
    return {
      cloud: {
        count: 5,
        positions: new Float32Array(15),
        colors: new Uint8Array(15),
        normals: new Int16Array(10),
      },
      report: cleanReport,
    };
  };
  await runDense(worker, { views: [view(0)], landmarks: [] });
  assert.deepEqual(seen, [10]);
  const types = worker.posted.map(({ message }) => message.type);
  assert.deepEqual(types.slice(-2), ['denseClean', 'denseResult']);
  const clean = worker.posted.at(-2);
  assert.equal(clean.message.positions.length, 15);
  assert.deepEqual(clean.transfer, [
    clean.message.positions.buffer,
    clean.message.colors.buffer,
    clean.message.normals.buffer,
  ]);
  const result = worker.posted.at(-1).message;
  assert.equal(result.points, 10);
  assert.equal(result.finalPoints, 5);
  assert.deepEqual(result.clean, cleanReport);
  assert.equal(result.cleanError, '');
  assert.equal(result.options.denseClean, true);
});

test('with the cleanup off the cloud stays raw and the options say so (3.12)', async () => {
  const dense = createDenseJob(false);
  keepFrame(dense, frame(0));
  const worker = context(dense, cloudMvs(10));
  worker.clean = async () => assert.fail('the cleanup must not run');
  await runDense(worker, { views: [view(0)], landmarks: [] });
  assert.ok(!worker.posted.some(({ message }) => message.type === 'denseClean'));
  const result = worker.posted.at(-1).message;
  assert.equal(result.options.denseClean, false);
  assert.equal(result.clean, null);
  assert.equal(result.finalPoints, 10);
});

test('a failed cleanup keeps the raw cloud and says why (3.12)', async () => {
  const dense = createDenseJob();
  keepFrame(dense, frame(0));
  const worker = context(dense, cloudMvs(10));
  worker.clean = async () => {
    throw new Error('device lost');
  };
  await runDense(worker, { views: [view(0)], landmarks: [] });
  const result = worker.posted.at(-1).message;
  assert.equal(result.failure, null);
  assert.equal(result.cleanError, 'device lost');
  assert.equal(result.finalPoints, 10);
  assert.ok(!worker.posted.some(({ message }) => message.type === 'denseClean'));
});

const meshReport = {
  tier: 'webgpu',
  points: 5,
  voxelSize: 0.05,
  radius: 0.125,
  nodes: 100,
  cubes: 20,
  vertices: 12,
  faces: 20,
  ms: { lattice: 1, field: 2, cubes: 3, vertices: 4 },
};
const meshOf = () => ({
  positions: new Float32Array(36),
  colors: new Uint8Array(36),
  indices: new Uint32Array(60),
});

// Verifies: docs/dense-mvs.md 3.13 (the mesh follows the cleanup in the dense run).
test('the mesh follows the cleanup, on the cleaned cloud, and records its counts (3.13)', async () => {
  const dense = createDenseJob(true, true);
  keepFrame(dense, frame(0));
  const worker = context(dense, cloudMvs(10));
  worker.clean = async () => ({
    cloud: {
      count: 5,
      positions: new Float32Array(15),
      colors: new Uint8Array(15),
      normals: new Int16Array(10),
    },
    report: cleanReport,
  });
  // A transfer detaches its buffers, as postMessage does: the mesh must still read the cloud.
  const post = worker.post;
  worker.post = (message, transfer = []) => {
    post(message, transfer);
    if (message.type === 'denseClean') structuredClone(transfer, { transfer });
  };
  const seen = [];
  worker.mesh = async (points) => {
    seen.push(points.count, points.positions.length);
    return { mesh: meshOf(), report: meshReport };
  };
  await runDense(worker, { views: [view(0)], landmarks: [] });
  assert.deepEqual(seen, [5, 15]);
  const types = worker.posted.map(({ message }) => message.type);
  assert.deepEqual(types.slice(-4), ['denseClean', 'denseActivity', 'denseMesh', 'denseResult']);
  assert.equal(worker.posted.at(-3).message.phase, 'mesh');
  const posted = worker.posted.at(-2);
  assert.deepEqual(Object.keys(posted.message), [
    'type',
    'positions',
    'colors',
    'indices',
    'report',
  ]);
  assert.deepEqual(posted.transfer, [
    posted.message.positions.buffer,
    posted.message.colors.buffer,
    posted.message.indices.buffer,
  ]);
  const result = worker.posted.at(-1).message;
  assert.deepEqual(result.mesh, meshReport);
  assert.equal(result.meshError, '');
  assert.equal(result.options.denseMesh, true);
});

test('with the mesh off, or when it fails, the points stay and the record says why (3.13)', async () => {
  const off = createDenseJob();
  keepFrame(off, frame(0));
  const worker = context(off, cloudMvs(10));
  worker.mesh = async () => assert.fail('the mesh must not run');
  await runDense(worker, { views: [view(0)], landmarks: [] });
  assert.ok(!worker.posted.some(({ message }) => message.type === 'denseMesh'));
  const result = worker.posted.at(-1).message;
  assert.equal(result.options.denseMesh, false);
  assert.equal(result.mesh, null);

  const dense = createDenseJob(false, true);
  keepFrame(dense, frame(0));
  const failing = context(dense, cloudMvs(10));
  failing.mesh = async () => {
    throw new Error('device lost');
  };
  await runDense(failing, { views: [view(0)], landmarks: [] });
  const failed = failing.posted.at(-1).message;
  assert.equal(failed.failure, null);
  assert.equal(failed.meshError, 'device lost');
  assert.equal(failed.mesh, null);
  assert.ok(!failing.posted.some(({ message }) => message.type === 'denseMesh'));
});

// Verifies: docs/dense-mvs.md 10.4 and the step 3 record: the lab's fusion and cleanup defaults.
test('the lab fuses with the TSDF by default and cleans after the pixel fusion only', () => {
  assert.equal(DENSE_LAB_FUSION, 'tsdf');
  assert.equal(createDenseJob().fusion, 'tsdf');
  assert.equal(createDenseJob(true, false, 'pixel').fusion, 'pixel');
  assert.equal(denseCleanDefault('pixel'), true);
  assert.equal(denseCleanDefault('tsdf'), false);
});

test('blockColors averages the kept image over the preview grid (Amendment 5, rule 3)', () => {
  // A 4 × 2 image: the left half red, the right half blue; a 2 × 1 grid of blocks.
  const rgba = new Uint8ClampedArray(4 * 4 * 2);
  for (let y = 0; y < 2; y++)
    for (let x = 0; x < 4; x++)
      rgba.set(x < 2 ? [200, 0, 0, 255] : [0, 0, 100, 255], 4 * (y * 4 + x));
  assert.deepEqual([...blockColors({ rgba, width: 4, height: 2 }, 2, 1)], [200, 0, 0, 0, 0, 100]);
  // A grid finer than the image still gives every block a pixel.
  assert.deepEqual(
    [...blockColors({ rgba, width: 4, height: 2 }, 4, 4)].slice(0, 6),
    [200, 0, 0, 200, 0, 0],
  );
});

test('fusion progress reaches the page before final points, and previews do not count as final output', async () => {
  const dense = createDenseJob(false);
  keepFrame(dense, frame(0));
  const worker = context(dense, async (_scene, _images, options) => {
    const event = {
      phase: 'integrate',
      pass: 1,
      passes: 2,
      view: 0,
      views: 1,
      viewId: 0,
      done: 0,
      total: 100,
      points: 0,
    };
    options.onFusionProgress(event);
    for (let done = 1; done < 100; done++) options.onFusionProgress({ ...event, done });
    options.onFusionProgress({ ...event, done: 100 });
    assert.equal(dense.stage, 'fusion');
    await options.onFusionPreview({
      count: 2,
      positions: new Float32Array(6),
      colors: new Uint8Array(6),
      normals: new Int16Array(4),
    });
    assert.equal(worker.posted.at(-1).message.type, 'denseFusionPoints');
    assert.ok(!worker.posted.some((e) => e.message.type === 'denseResult'));
    return { points: { count: 3 }, views: [] };
  });
  await runDense(worker, { views: [view(0)], landmarks: [] });
  const activity = worker.posted.filter(
    (e) => e.message.type === 'denseActivity' && e.message.phase === 'integrate',
  );
  assert.equal(
    activity.length,
    2,
    'intra-chunk events are throttled, phase start and completion survive',
  );
  assert.equal(activity[0].message.viewId, 0);
  assert.equal(activity[1].message.done, 100);
  const preview = worker.posted.find((e) => e.message.type === 'denseFusionPoints');
  assert.deepEqual(preview.transfer, [
    preview.message.positions.buffer,
    preview.message.colors.buffer,
    preview.message.normals.buffer,
  ]);
  assert.equal(worker.posted.at(-1).message.points, 3);
});

test('changing cameras keeps the phase clock and forwards region-wide submitted work', async (t) => {
  let now = 1000;
  t.mock.method(performance, 'now', () => now);
  const dense = createDenseJob(false);
  keepFrame(dense, frame(0));
  const worker = context(dense, async (_scene, _images, options) => {
    const event = {
      phase: 'carve',
      pass: 1,
      passes: 2,
      view: 0,
      views: 2,
      viewId: 0,
      done: 100,
      total: 100,
      phaseDone: 100,
      phaseTotal: 300,
    };
    options.onFusionProgress(event);
    now += 1500;
    options.onFusionProgress({ ...event, view: 1, viewId: 1, phaseDone: 200 });
    now += 1000;
    options.onFusionProgress({ ...event, pass: 2 });
    return { points: { count: 0 }, views: [] };
  });
  await runDense(worker, { views: [view(0)], landmarks: [] });
  const activity = worker.posted
    .filter(({ message }) => message.phase === 'carve')
    .map(({ message }) => message);
  assert.deepEqual(
    activity.map((event) => event.phaseMs),
    [0, 1500, 0],
  );
  assert.deepEqual(
    activity.map((event) => event.phaseDone),
    [100, 200, 100],
  );
});

test('the worker archives full diagnostic stages before its final result, including confidence and source pixels', async () => {
  const dense = createDenseJob(false, false, 'tsdf');
  keepFrame(dense, frame(7));
  const ctx = context(dense, async (_scene, _images, options) => {
    for (const stage of ['photometric', 'geometric-unfiltered', 'geometric-filtered', 'planar']) {
      await options.onDepthDiagnostic({
        stage,
        viewId: 7,
        width: W,
        height: H,
        depth: new Float32Array(W * H).fill(2),
        ...(['geometric-filtered', 'planar'].includes(stage)
          ? { confidence: new Float32Array(W * H).fill(0.8) }
          : {}),
        depthScaleToModel: 3,
        camera: view(7).camera,
        pose: view(7).pose,
        sourceViewIds: [1, 2],
      });
    }
    return {
      points: { count: 0 },
      views: [{ id: 7, status: 'done', sources: [1, 2], depthPixels: W * H, points: 0 }],
    };
  });
  ctx.component = 2;
  ctx.options = DENSE_LAB_OPTIONS;
  await runDense(ctx, { views: [view(7)], landmarks: [] });
  const manifest = ctx.artifacts.find((f) => f.message.name.endsWith('.json'));
  assert.ok(manifest.message.name.includes('c002'));
  const data = JSON.parse(manifest.message.data);
  assert.equal(data.complete, true);
  assert.deepEqual(
    data.maps.map((m) => m.stage),
    ['photometric', 'geometric-unfiltered', 'geometric-filtered', 'planar'],
  );
  assert.ok(data.maps.every((m) => m.width === W && m.height === H));
  const archive = ctx.artifacts.find((f) => f.message.name.endsWith('.tar.gz'));
  assert.deepEqual(archive.transfer, [archive.message.data.buffer]);
  assert.equal(ctx.posted.at(-1).message.type, 'denseResult');
});
