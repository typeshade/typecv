// Verifies: docs/multi-video.md 6.2 (the `multiVideo` worker job), 2.2 (the cap per video), 6.3
// (tags and the matrix) and 7.1 (`?canonical=0`), with a fake feature front end and the real
// view graph and mapper on the CPU tier of the device (`dist/internal/gpu.js`, `npm run build`).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  reconstructMultiVideoGpu,
  reconstructMultiVideoOwnedGpu,
} from '../dist/internal/gpu.js';
import { quantizeDescriptors } from '../src/reconstruction/view-graph.ts';
import { pathScene } from '../tests/view-graph-scene.ts';
import { MultiVideoJob } from './src/lib/multi-video-job.ts';
import { multiVideoQuery } from './src/lib/multi-video.ts';

const scene = pathScene(
  [
    { label: 'orbit', cloud: 0, views: 6, from: -30, to: 30, radius: 9, height: -1 },
    { label: 'ascent', cloud: 0, views: 6, from: 5, to: 40, radius: 9, height: -3 },
    { label: 'elsewhere', cloud: 1, views: 6, from: -30, to: 30, radius: 9, height: -1 },
  ],
  { points: 50 },
);
const LABELS = ['orbit', 'ascent', 'elsewhere'];
const SIDE = 64;
const CAMERA = { fx: 850, fy: 850, cx: 480, cy: 270, k1: 0, k2: 0 };

/** The fake front end: the first byte of a frame says which scene keyframe its features are. */
const api = {
  async extractSiftGpuResident(_device, rgba) {
    const k = scene.keyframes[rgba[0]];
    return {
      count: k.keypoints.length / 2,
      readKeypoints: async () => k.keypoints,
      dispose: async () => {},
      index: rgba[0],
    };
  },
  async describeKeyframeGpu(_device, sift) {
    const k = scene.keyframes[sift.index];
    return { descriptors: quantizeDescriptors(k.descriptors), global: k.global };
  },
  reconstructMultiVideoGpu,
  reconstructMultiVideoOwnedGpu,
};

function frames(label) {
  const first = LABELS.indexOf(label) * 6;
  return Array.from({ length: 6 }, (_, i) => {
    const pixels = new Uint8ClampedArray(SIDE * SIDE * 4);
    for (let p = 0; p < pixels.length; p++) pixels[p] = ((p * 31 + (first + i) * 97) >>> 3) & 255;
    pixels[0] = first + i;
    return {
      id: i,
      width: SIDE,
      height: SIDE,
      pixels,
      time: i,
      camera: CAMERA,
      image: 'data:image/jpeg;base64,' + label + i,
      last: i === 5,
    };
  });
}

async function run(order, { search = '', skipExtra = false, snapshotIntervalMs } = {}) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  const posted = [];
  try {
    const query = multiVideoQuery(search);
    const job = new MultiVideoJob(
      api,
      device,
      new Float32Array(32 * 128),
      {
        query,
        maxFeatures: 4096,
        firstOctave: 0,
        snapshotIntervalMs,
        mapper: { seed: 7, ransacIterations: 100, finalBundleIterations: 4 },
        graph: {
          retrievalK: 3,
          temporalNeighbours: 2,
          exclusionWindow: 2,
          reservePerVideoPair: 2,
          verification: { seed: 7, ransacIterations: 100 },
        },
      },
      (m) => posted.push(m),
    );
    for (const label of order) {
      const videoId = scene.videoIds.get(label);
      job.video({ videoId, label, durationSeconds: 6, focalRatio: 0.8 });
      const list = frames(label);
      for (const frame of list) await job.frame(videoId, frame);
      if (skipExtra) await job.frame(videoId, { ...list[5], id: 6, time: 5.1, last: false });
      job.endVideo(videoId);
    }
    await job.finish();
    return posted;
  } finally {
    await device.dispose();
  }
}
const of = (posted, type) => posted.filter((m) => m.type === type);

test('a run posts the frames, the graph, one model for each component and the result', async () => {
  const posted = await run(LABELS, { skipExtra: true });
  assert.equal(of(posted, 'frameComplete').filter((m) => m.keyframe).length, 18);
  // The frame 0.1 s after a keyframe is skipped by the phase gate, with its reason.
  const skipped = of(posted, 'frameComplete').filter((m) => m.skipped);
  assert.equal(skipped.length, 3);
  assert.match(skipped[0].skipped, /minimum gap/);
  assert.equal(of(posted, 'features').length, 18);
  assert.deepEqual(
    of(posted, 'videoComplete').map((m) => [m.keyframes, m.dropped]),
    [
      [6, 0],
      [6, 0],
      [6, 0],
    ],
  );
  const [graph] = of(posted, 'graph');
  const activities = of(posted, 'graphActivity');
  assert.ok(activities.some((message) => message.stage === 'matching'));
  assert.ok(activities.some((message) => message.stage === 'verification'));
  const activity = activities.at(-1);
  assert.equal(activity.stage, 'components');
  assert.ok(
    posted.indexOf(activity) < posted.indexOf(of(posted, 'model')[0]),
    'the graph arrives before any reconstructed map',
  );
  const displayInliers = activity.links.reduce((sum, link) => sum + link.inliers, 0);
  assert.equal(graph.canonical, true);
  assert.equal(graph.stats.keyframes, 18);
  assert.ok(graph.stats.candidatePairs < graph.stats.exhaustivePairs);
  assert.deepEqual(
    graph.components.map((c) => c.keyframes),
    [12, 6],
  );
  // The matrix of 6.3: symmetric, orbit and ascent join, the third video joins no one.
  const { order, cells } = graph.matrix;
  const at = (a, b) =>
    cells[order.indexOf(scene.videoIds.get(a))][order.indexOf(scene.videoIds.get(b))];
  assert.ok(at('orbit', 'ascent') > 0);
  assert.equal(at('orbit', 'ascent'), at('ascent', 'orbit'));
  assert.equal(at('orbit', 'elsewhere'), 0);
  assert.equal(at('ascent', 'elsewhere'), 0);
  assert.equal(
    displayInliers,
    cells.reduce((sum, row) => sum + row.reduce((a, b) => a + b, 0), 0) / 2,
    'throttled snapshots retain every accepted cross-video inlier',
  );
  const models = of(posted, 'model').filter((m) => !m.partial);
  assert.deepEqual(
    models.map((m) => m.component),
    [0, 1],
  );
  assert.equal(models[0].reconstruction.views.length, 12);
  const view = models[0].reconstruction.views[0];
  assert.ok(view.image.startsWith('data:image/jpeg') && view.videoId && 'time' in view);
  assert.ok(models[0].reconstruction.landmarks.every((l) => l.color.length === 3));
  const [result] = of(posted, 'result');
  const tags = Object.fromEntries(result.summary.videos.map((v) => [v.label, v.tags]));
  assert.deepEqual(tags, { orbit: ['component'], ascent: ['component'], elsewhere: ['alone'] });
  assert.deepEqual(result.summary.notConnected, []);
  assert.equal(result.summary.viewGraph.nodes, 18);
  assert.equal(result.summary.viewGraph.exhaustivePairs, 153);
  assert.ok(result.summary.crossVideoEdges.edges > 0);
  assert.deepEqual(
    result.summary.components.map((c) => [c.keyframes, c.videos]),
    [
      [12, 2],
      [6, 1],
    ],
  );
  for (const v of result.summary.videos) assert.ok(v.registeredViews <= v.keyframes);
  // The record lists the frames each video gave (kept or not) and their analysis size.
  for (const v of result.summary.videos) {
    assert.ok(v.frames >= v.keyframes && v.keyframes > 0, v.label);
    assert.deepEqual([v.width, v.height], [SIDE, SIDE]);
  }
  assert.equal(
    result.summary.videos.reduce((n, v) => n + v.frames, 0),
    of(posted, 'frameComplete').length,
  );
  for (const name of [
    'features',
    'globalDescriptor',
    'retrieval',
    'pairMatching',
    'pairVerification',
    'components',
  ])
    assert.ok(result.stages[name].elapsedMs >= 0, name);
  assert.equal(result.keyframeCount, 18);
});

test('the order of the videos decides nothing, and ?canonical=0 reaches the result', async () => {
  const a = of(await run(LABELS), 'result')[0];
  const b = of(await run([...LABELS].reverse()), 'result')[0];
  const names = (r) =>
    r.components.map((c) =>
      c.viewIds.map((v) => r.keyframes.find((k) => k.viewId === v).frameKey).sort(),
    );
  assert.deepEqual(names(b), names(a));
  assert.equal(a.canonical, true);
  const c = of(await run(LABELS, { search: '?canonical=0' }), 'result')[0];
  assert.equal(c.canonical, false);
  assert.deepEqual(
    c.keyframes.map((k) => k.viewId),
    Array.from({ length: 18 }, (_, i) => i),
  );
});

test('the cap per video keeps the first and the last keyframe (2.2)', async () => {
  const posted = await run(LABELS, { search: '?maxKeyframes=4' });
  assert.deepEqual(
    of(posted, 'videoComplete').map((m) => [m.keyframes, m.dropped]),
    [
      [4, 2],
      [4, 2],
      [4, 2],
    ],
  );
  const [graph] = of(posted, 'graph');
  assert.equal(graph.stats.keyframes, 12);
  // The third video has 4 keyframes, under minComponentViews: no map, tag "not connected".
  const result = of(posted, 'result')[0];
  const tag = result.summary.videos.find((v) => v.label === 'elsewhere').tags;
  assert.deepEqual(tag, ['not connected']);
  assert.equal(result.summary.notConnected.length, 1);
});

test('the job refuses a video twice and a frame of an unknown video', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const job = new MultiVideoJob(
      api,
      device,
      new Float32Array(0),
      { query: multiVideoQuery(''), maxFeatures: 1, firstOctave: 0, mapper: {} },
      () => {},
    );
    job.video({ videoId: 'aaaaaaaaaaaaaaaa', label: null, durationSeconds: 1, focalRatio: 0.8 });
    assert.throws(() =>
      job.video({ videoId: 'aaaaaaaaaaaaaaaa', label: null, durationSeconds: 1, focalRatio: 0.8 }),
    );
    await assert.rejects(() => job.frame('bbbbbbbbbbbbbbbb', frames('orbit')[0]));
  } finally {
    await device.dispose();
  }
});

test('a run names its stages and streams the growing maps (Amendment 8)', async () => {
  const posted = await run(LABELS, { snapshotIntervalMs: 0 });
  const stages = of(posted, 'multiStage');
  // The stages come in order. The counts of one stage never go down.
  const order = ['candidates', 'matching', 'verification', 'components', 'reconstruction'];
  let at = 0;
  for (const m of stages) {
    const i = order.indexOf(m.stage);
    assert.ok(i >= at, 'stage order: ' + m.stage);
    at = i;
  }
  for (const name of ['matching', 'verification', 'reconstruction']) {
    const done = stages.filter((m) => m.stage === name).map((m) => m.done);
    assert.ok(done.length > 1, name);
    assert.deepEqual(
      done,
      [...done].sort((a, b) => a - b),
      name + ' counts rise',
    );
  }
  const matching = stages.filter((m) => m.stage === 'matching').at(-1);
  assert.equal(matching.done, matching.total);
  const verification = stages.filter((m) => m.stage === 'verification').at(-1);
  assert.equal(verification.done, verification.total);
  // Both components are planned: 12 + 6 views. The last count is all of them.
  const last = stages.filter((m) => m.stage === 'reconstruction').at(-1);
  assert.deepEqual([last.done, last.total], [18, 18]);
  // Partial maps come before the final map of their component, and grow.
  const models = of(posted, 'model');
  for (const component of [0, 1]) {
    const mine = models.filter((m) => m.component === component);
    const final = mine.filter((m) => !m.partial);
    assert.equal(final.length, 1);
    assert.equal(mine.at(-1), final[0], 'the final map is the last message of its component');
    const partial = mine.filter((m) => m.partial);
    assert.ok(partial.length >= 2, 'partial maps of component ' + component);
    const sizes = partial.map((m) => m.reconstruction.views.length);
    assert.deepEqual(
      sizes,
      [...sizes].sort((a, b) => a - b),
    );
    assert.ok(sizes.at(-1) <= final[0].reconstruction.views.length);
    // The landmarks of a growing map go as flat typed arrays, not as objects.
    assert.ok(
      partial.every((m) => {
        const { packed } = m.reconstruction;
        return (
          packed.xyz.length === 3 * packed.count &&
          packed.rgb.length === 3 * packed.count &&
          packed.trackIds.length === packed.count &&
          m.reconstruction.landmarks === undefined
        );
      }),
    );
  }
});

test('the snapshot throttle sends the latest map once in an interval', async () => {
  const { Snapshots } = await import('./src/lib/multi-video-job.ts');
  const sent = [];
  const throttle = new Snapshots(40, (s) => sent.push(s.registered));
  for (let i = 1; i <= 5; i++) throttle.offer({ component: 0, registered: i, reconstruction: {} });
  assert.deepEqual(sent, [1], 'the first goes at once');
  await new Promise((resolve) => setTimeout(resolve, 80));
  assert.deepEqual(sent, [1, 5], 'then only the latest');
  const second = [];
  const other = new Snapshots(40, (s) => second.push(s.registered));
  other.offer({ component: 0, registered: 1, reconstruction: {} });
  other.offer({ component: 0, registered: 2, reconstruction: {} });
  other.cancel();
  await new Promise((resolve) => setTimeout(resolve, 80));
  assert.deepEqual(second, [1], 'cancel drops a waiting map');
});

// Verifies: docs/multi-video.md Amendment 9 (the dense MVS of a multi-video run), with a fake MVS
// in place of `reconstructDenseGpu` and the real view graph and mapper on the CPU tier.
const DENSE_OPTIONS = {
  maxSources: 6,
  bestSources: 3,
  windowRadius: 5,
  windowStep: 2,
  photometricIterations: 8,
  geometricIterations: 4,
  minConsistentViews: 3,
};

/** A fake MVS over the scene's views: progress, one points event per view and the result. */
function fakeDense(calls, { failOn } = {}) {
  return {
    ...api,
    async reconstructDenseGpu(_device, sceneArg, images, options) {
      calls.push({ views: sceneArg.views.map((v) => v.id), images, options });
      const ids = sceneArg.views.map((v) => v.id);
      if (failOn === calls.length) throw new Error('kernel failed');
      for (const [i, viewId] of ids.entries()) {
        options.signal.throwIfAborted();
        options.onProgress({ stage: 'photometric', viewId, done: i + 1, total: ids.length });
        const image = images.get(viewId),
          view = sceneArg.views.find((v) => v.id === viewId);
        for (const stage of ['photometric', 'geometric-unfiltered', 'geometric-filtered'])
          await options.onDepthDiagnostic({
            viewId,
            stage,
            width: image.width,
            height: image.height,
            depth: new Float32Array(image.width * image.height).fill(2),
            ...(stage === 'geometric-filtered'
              ? { confidence: new Float32Array(image.width * image.height).fill(0.8) }
              : {}),
            depthScaleToModel: 1,
            camera: view.camera,
            pose: view.pose,
            sourceViewIds: ids.filter((id) => id !== viewId),
          });
      }
      for (const viewId of ids)
        await options.onView({
          type: 'points',
          viewId,
          count: 2,
          positions: new Float32Array(6),
          colors: new Uint8Array(6),
          normals: new Int16Array(4),
        });
      options.signal.throwIfAborted();
      return {
        points: { count: 2 * ids.length },
        views: ids.map((id) => ({ id, status: 'done', sources: [], depthPixels: 4, points: 2 })),
      };
    },
  };
}

async function runWithDense(denseApi, { cancelAfter, denseDevice } = {}) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  const posted = [];
  try {
    const job = new MultiVideoJob(
      denseApi,
      device,
      new Float32Array(32 * 128),
      {
        query: multiVideoQuery(''),
        maxFeatures: 4096,
        firstOctave: 0,
        mapper: { seed: 7, ransacIterations: 100, finalBundleIterations: 4 },
        graph: {
          retrievalK: 3,
          temporalNeighbours: 2,
          exclusionWindow: 2,
          reservePerVideoPair: 2,
          verification: { seed: 7, ransacIterations: 100 },
        },
        dense: { clean: false, mesh: false, options: DENSE_OPTIONS, lastKernel: () => 'k' },
      },
      (m, transfer) => {
        posted.push({ ...m, transferred: transfer?.length ?? 0 });
        if (cancelAfter && m.type === 'densePoints') job.cancelDense();
      },
    );
    for (const label of LABELS) {
      const videoId = scene.videoIds.get(label);
      job.video({ videoId, label, durationSeconds: 6, focalRatio: 0.8 });
      for (const frame of frames(label)) await job.frame(videoId, frame);
      job.endVideo(videoId);
    }
    await job.finish();
    const afterSparse = posted.length;
    await job.runDense(denseDevice ?? device);
    return { posted, afterSparse };
  } finally {
    await device.dispose();
  }
}

test('the dense stage runs on each component after the sparse result (Amendment 9)', async () => {
  const calls = [];
  const { posted, afterSparse } = await runWithDense(fakeDense(calls));
  // No dense message before the sparse result is posted.
  assert.ok(posted.slice(0, afterSparse).every((m) => !m.type.startsWith('dense')));
  assert.equal(posted[afterSparse - 1].type, 'result');
  const results = posted.filter((m) => m.type === 'denseResult');
  // Both components: the first of 12 keyframes, the second of 6. Each gets its own result.
  assert.deepEqual(
    results.map((m) => m.component),
    [0, 1],
  );
  assert.equal(calls.length, 2);
  // The images are the keyframes' RGBA at the analysis size, keyed by the registered view ids.
  assert.deepEqual(
    calls.map((c) => [...c.images.keys()].sort((a, b) => a - b)),
    calls.map((c) => [...c.views].sort((a, b) => a - b)),
  );
  for (const c of calls)
    for (const image of c.images.values()) {
      assert.equal(image.width, SIDE);
      assert.equal(image.rgba.length, SIDE * SIDE * 4);
    }
  // The two components share no view.
  assert.equal(
    new Set([...calls[0].views, ...calls[1].views]).size,
    calls[0].views.length + calls[1].views.length,
  );
  for (const r of results) {
    assert.equal(r.failure, null);
    assert.equal(r.skipped, '');
    assert.ok(r.points > 0 && r.views.length > 0);
    assert.deepEqual(r.options.minConsistentViews, 3);
  }
  // Points and progress carry their component; points are transferred, depth maps are not sent.
  const points = posted.filter((m) => m.type === 'densePoints');
  assert.ok(points.length > 0 && points.every((m) => m.transferred === 3));
  assert.ok(posted.filter((m) => m.type === 'denseProgress').every((m) => 'component' in m));
  const activity = posted.filter((m) => m.type === 'denseActivity');
  assert.deepEqual([...new Set(activity.map((m) => m.component))], [0, 1]);
  assert.ok(activity.some((m) => m.phase === 'photometric' && m.done === 0));
  assert.equal(posted.filter((m) => m.type === 'denseDepth').length, 0);
  const artifacts = posted.filter((m) => m.type === 'denseArtifact');
  assert.deepEqual([...new Set(artifacts.map((m) => m.component))], [0, 1]);
  const manifests = artifacts
    .filter((m) => m.name.endsWith('.json'))
    .map((m) => JSON.parse(m.data));
  assert.equal(manifests.length, 2);
  assert.ok(manifests.every((m) => m.complete && m.maps.length === 3 * m.expectedViews.length));
  assert.ok(artifacts.filter((m) => m.name.endsWith('.tar.gz')).every((m) => m.transferred === 1));

  // The stage ends once, after the last component.
  assert.equal(posted.at(-1).type, 'denseDone');
  assert.equal(posted.filter((m) => m.type === 'denseDone').length, 1);
});

test('denseDone reports the separate Dense device once after all components', async () => {
  const calls = [];
  const denseDevice = {
    statistics: { liveBytes: 4096, pooledBytes: 0 },
    get kernelStatistics() {
      return {
        rows: [
          {
            name: 'tsdfIntegrate',
            groups: calls.length,
            dispatches: calls.length,
            timedGroups: calls.length,
            gpuNs: calls.length * 1_000_000,
          },
        ],
        dropped: 1,
      };
    },
  };
  const { posted } = await runWithDense(fakeDense(calls), { denseDevice });
  assert.equal(calls.length, 2);
  const done = posted.filter((message) => message.type === 'denseDone');
  assert.equal(done.length, 1);
  assert.equal(done[0].gpuDropped, 1);
  assert.deepEqual(done[0].gpuKernels, [
    { name: 'tsdfIntegrate', groups: 2, dispatches: 2, timedGroups: 2, gpuNs: 2_000_000 },
  ]);
  assert.ok(
    posted
      .find((message) => message.type === 'result')
      .gpuKernels.every((row) => row.name !== 'tsdfIntegrate'),
    'the sparse table belongs to the original device',
  );
});

test('a cancel keeps the points fused so far and starts no later component', async () => {
  const calls = [];
  const { posted } = await runWithDense(fakeDense(calls), { cancelAfter: true });
  const results = posted.filter((m) => m.type === 'denseResult');
  assert.equal(results.length, 2);
  assert.ok(results.every((m) => m.cancelled === true));
  assert.equal(calls.length, 1);
  assert.equal(results[1].points, 0);
});

test('a failed component does not stop the next one', async () => {
  const calls = [];
  const { posted } = await runWithDense(fakeDense(calls, { failOn: 1 }));
  const results = posted.filter((m) => m.type === 'denseResult');
  assert.match(results[0].failure.message, /kernel failed/);
  assert.equal(results[1].failure, null);
  assert.ok(results[1].points > 0);
});

test('a run without the dense option keeps no pixels and posts no dense message', async () => {
  const posted = await run(LABELS);
  assert.ok(posted.every((m) => !String(m.type).startsWith('dense')));
});
