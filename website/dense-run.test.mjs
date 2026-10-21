// Verifies: docs/dense-mvs.md step 6 (section 5.3, the worker's dense MVS), with a fake MVS in
// place of `reconstructDenseGpu`.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createDenseJob,
  denseFailure,
  denseViewLimit,
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
const frame = (id) => ({ id, width: W, height: H, pixels: new Uint8ClampedArray(4 * W * H) });
const view = (id) => ({
  id,
  pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [id, 0, 0] },
  camera: { fx: 10, fy: 10, cx: 4, cy: 2 },
  image: 'data:image/jpeg;base64,x',
});

/** A worker context around a fake MVS: `mvs(scene, images, options)` plays the run. */
function context(dense, mvs) {
  const posted = [];
  let bytes = { liveBytes: 1000, pooledBytes: 400 };
  return {
    posted,
    setBytes: (next) => (bytes = next),
    dense,
    options: OPTIONS,
    statistics: () => bytes,
    reconstruct: mvs,
    post: (message, transfer = []) => posted.push({ message, transfer }),
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
    await stage('photometric');
    await stage('geometric');
    run.setBytes({ liveBytes: 5000, pooledBytes: 400 });
    for (let v = 0; v < N; v++)
      await options.onView({
        type: 'depth',
        viewId: ids[v],
        width: 2,
        height: 1,
        depth: new Float32Array([1, 0]),
      });
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

test('the kept RGBA stays within the views the dense layout holds (4.2)', () => {
  // ⌊(workingBytes / P − 4) / 16⌋: 15 views at 640 × 427 and 13 at 640 × 480 with 64 MiB.
  assert.equal(denseViewLimit(64 * 1024 ** 2, 640 * 427), 15);
  assert.equal(denseViewLimit(64 * 1024 ** 2, 640 * 480), 13);
  assert.equal(denseViewLimit(256 * 1024 ** 2, 1280 * 960), 13);
  const dense = createDenseJob();
  const workingBytes = (16 * 3 + 4) * W * H;
  for (const id of [0, 1, 2]) keepFrame(dense, frame(id), workingBytes);
  assert.equal(dense.limit, 3);
  assert.deepEqual([...dense.rgba.keys()], [0, 1, 2]);
  assert.equal(dense.rgba.get(1).width, W);
  assert.equal(dense.skipped, '');
  // A fourth frame may be a candidate that is no keyframe: the map keeps it (off by one).
  keepFrame(dense, frame(3), workingBytes);
  assert.equal(dense.rgba.size, 4);
  assert.equal(dense.skipped, '');
  // The candidate is no keyframe: the worker deletes it, and the next frame fits.
  dense.rgba.delete(3);
  keepFrame(dense, frame(4), workingBytes);
  dense.rgba.delete(4);
  assert.equal(dense.skipped, '');
  // A fifth entry cannot be one candidate over the limit: the map is cleared and says why.
  keepFrame(dense, frame(3), workingBytes);
  keepFrame(dense, frame(4), workingBytes);
  assert.equal(dense.rgba.size, 0);
  assert.match(dense.skipped, /3 views the dense layout holds/);
  keepFrame(dense, frame(5), workingBytes);
  assert.equal(dense.rgba.size, 0);
});

test('a run reports skipped only when the final keyframes exceed the limit', async () => {
  const workingBytes = (16 * 3 + 4) * W * H;
  const model = { views: [0, 1, 2, 3].map(view), landmarks: [{ xyz: [0, 0, 1] }] };
  // Three keyframes and a fourth candidate that was no keyframe: the run is not skipped.
  const fits = createDenseJob();
  for (const id of [0, 1, 2, 3]) keepFrame(fits, frame(id), workingBytes);
  fits.rgba.delete(3);
  const run = {};
  const worker = context(fits, fakeMvs(run));
  run.setBytes = worker.setBytes;
  await runDense(worker, { views: model.views.slice(0, 3), landmarks: model.landmarks });
  const result = worker.posted.at(-1).message;
  assert.equal(result.type, 'denseResult');
  assert.equal(result.skipped, '');
  // Four keyframes at the limit of three: skipped, with the reason.
  const over = createDenseJob();
  for (const id of [0, 1, 2, 3]) keepFrame(over, frame(id), workingBytes);
  const worker2 = context(over, fakeMvs({}));
  await runDense(worker2, model);
  assert.match(worker2.posted.at(-1).message.skipped, /3 views the dense layout holds/);
});

test('a dense run posts progress, depth maps, points and one result, with arrays transferred', async () => {
  const dense = createDenseJob();
  for (const id of [0, 1, 2]) keepFrame(dense, frame(id), 1e9);
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
  assert.equal(run.options.signal, dense.abort.signal);
  const types = worker.posted.map(({ message }) => message.type);
  assert.deepEqual(types, [
    'denseProgress',
    'denseProgress',
    'denseProgress',
    'denseProgress',
    'denseDepth',
    'denseDepth',
    'densePoints',
    'denseProgress',
    'densePoints',
    'denseProgress',
    'denseResult',
  ]);
  assert.deepEqual(worker.posted[0].message, {
    type: 'denseProgress',
    stage: 'photometric',
    viewId: 0,
    done: 1,
    total: 2,
  });
  const depth = worker.posted[4];
  assert.deepEqual(Object.keys(depth.message), ['type', 'viewId', 'width', 'height', 'depth']);
  assert.deepEqual(depth.transfer, [depth.message.depth.buffer]);
  const points = worker.posted[8];
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
  assert.deepEqual(result.options, { ...OPTIONS, denseClean: true });
  assert.deepEqual(result.kernels, ['mvsPropagate']);
  // 4.3, step 6: the device's bytes before the MVS, at the depth stage and at the fusion stage.
  assert.deepEqual(result.memory, {
    liveBytesBefore: 1000,
    pooledBytesBefore: 400,
    depthStageBytes: 4600,
    depthStageFormulaBytes: (16 * 2 + 4) * W * H,
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
  for (const id of [0, 1]) keepFrame(dense, frame(id), 1e9);
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
  for (const id of [0, 1]) keepFrame(dense, frame(id), 1e9);
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
  for (const id of [0, 1]) keepFrame(lost, frame(id), 1e9);
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

test('a run whose images were dropped is skipped with its reason', async () => {
  const dense = createDenseJob();
  for (const id of [0, 1, 2]) keepFrame(dense, frame(id), (16 + 4) * W * H);
  const worker = context(dense, async () => assert.fail('no MVS without images'));
  await runDense(worker, { views: [view(0)], landmarks: [] });
  assert.equal(worker.posted.length, 1);
  const result = worker.posted[0].message;
  assert.equal(result.type, 'denseResult');
  assert.match(result.skipped, /1 views the dense layout holds/);
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
  keepFrame(dense, frame(0), 1e9);
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
  keepFrame(dense, frame(0), 1e9);
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
  keepFrame(dense, frame(0), 1e9);
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
  keepFrame(dense, frame(0), 1e9);
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
  const seen = [];
  worker.mesh = async (points) => {
    seen.push(points.count);
    return { mesh: meshOf(), report: meshReport };
  };
  await runDense(worker, { views: [view(0)], landmarks: [] });
  assert.deepEqual(seen, [5]);
  const types = worker.posted.map(({ message }) => message.type);
  assert.deepEqual(types.slice(-3), ['denseClean', 'denseMesh', 'denseResult']);
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
  keepFrame(off, frame(0), 1e9);
  const worker = context(off, cloudMvs(10));
  worker.mesh = async () => assert.fail('the mesh must not run');
  await runDense(worker, { views: [view(0)], landmarks: [] });
  assert.ok(!worker.posted.some(({ message }) => message.type === 'denseMesh'));
  const result = worker.posted.at(-1).message;
  assert.equal(result.options.denseMesh, false);
  assert.equal(result.mesh, null);

  const dense = createDenseJob(false, true);
  keepFrame(dense, frame(0), 1e9);
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
