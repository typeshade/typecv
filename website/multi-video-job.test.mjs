// Verifies: docs/multi-video.md 6.2 (the `multiVideo` worker job), 2.2 (the cap per video), 6.3
// (tags and the matrix) and 7.1 (`?canonical=0`), with a fake feature front end and the real
// view graph and mapper on the CPU tier of the device (`dist/internal/gpu.js`, `npm run build`).
import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, reconstructMultiVideoGpu } from '../dist/internal/gpu.js';
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

async function run(order, { search = '', skipExtra = false } = {}) {
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
  const models = of(posted, 'model');
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
