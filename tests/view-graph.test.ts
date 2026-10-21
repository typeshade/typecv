import test from 'node:test';
import assert from 'node:assert/strict';
import { seededRandom } from '../src/geometry/numerics.ts';
import {
  DescriptorStore,
  buildViewGraph,
  connectedComponents,
  cpuViewGraphBackend,
  dequantizeDescriptors,
  matchDescriptorsCpu,
  quantizeDescriptors,
  selectCandidates,
  type ViewGraph,
  type ViewGraphKeyframe,
} from '../src/reconstruction/view-graph.ts';
import { pathScene, type PathSpec } from './view-graph-scene.ts';

// Verifies: docs/multi-video.md 3.3 (candidates), 3.4 (verification), 3.5 (8-bit store), 4.1 (components).

const THREE_PATHS: PathSpec[] = [
  { label: 'orbit', cloud: 0, views: 10, from: -30, to: 30, radius: 9, height: -1 },
  { label: 'ascent', cloud: 0, views: 10, from: 5, to: 40, radius: 9, height: -3 },
  { label: 'elsewhere', cloud: 1, views: 10, from: -30, to: 30, radius: 9, height: -1 },
];
const OPTIONS = {
  retrievalK: 4,
  temporalNeighbours: 2,
  exclusionWindow: 3,
  reservePerVideoPair: 2,
};

const scene = pathScene(THREE_PATHS);
const videoOfPath = (label: string) => scene.videoIds.get(label)!;

let cached: ViewGraph | undefined;
async function graph(): Promise<ViewGraph> {
  return (cached ??= await buildViewGraph(scene.keyframes, cpuViewGraphBackend(), OPTIONS));
}

test('the synthetic scene verifies N*k pairs instead of N^2 and finds the components', async () => {
  const g = await graph();
  assert.equal(g.stats.keyframes, 30);
  assert.equal(g.stats.exhaustivePairs, 435);
  assert.ok(g.stats.candidatePairs < g.stats.exhaustivePairs / 2, `${g.stats.candidatePairs}`);
  assert.equal(
    g.stats.candidatePairs,
    g.stats.temporalPairs + g.stats.retrievalPairs + g.stats.reservePairs,
  );
  assert.ok(g.stats.crossVideoEdges > 0);
  // The two paths around cloud 0 join. The path around cloud 1 stands alone.
  assert.equal(g.components.length, 2);
  const [big, small] = g.components;
  assert.equal(big!.viewIds.length, 20);
  assert.deepEqual(big!.videoIds, [videoOfPath('orbit'), videoOfPath('ascent')].sort());
  assert.equal(small!.viewIds.length, 10);
  assert.deepEqual(small!.videoIds, [videoOfPath('elsewhere')]);
  assert.ok(big!.reconstructed && small!.reconstructed);
  const tag = (label: string) => g.videos.find((v) => v.videoId === videoOfPath(label))!.tag;
  assert.equal(tag('orbit'), 'connected');
  assert.equal(tag('ascent'), 'connected');
  assert.equal(tag('elsewhere'), 'alone');
  // No edge joins two clouds, and every edge has the inliers the record asks for.
  const cloudOf = new Map(scene.keyframes.map((k, i) => [k.viewId, scene.truth[i]!.cloud]));
  for (const e of g.edges) {
    assert.equal(cloudOf.get(e.first), cloudOf.get(e.second));
    assert.ok(e.first < e.second && e.inliers >= 25);
    assert.equal(e.matches.length / 2, e.inliers);
  }
});

test('the graph does not depend on the order of the input', async () => {
  const g = await graph();
  const rng = seededRandom(9);
  const shuffled = [...scene.keyframes].sort(() => rng() - 0.5);
  const h = await buildViewGraph(shuffled, cpuViewGraphBackend(), OPTIONS);
  const summary = (x: ViewGraph) => ({
    candidates: x.candidates,
    edges: x.edges.map((e) => [e.first, e.second, e.inliers, e.source]),
    components: x.components.map((c) => c.viewIds),
    videos: x.videos,
  });
  assert.deepEqual(summary(h), summary(g));
});

test('expansion only adds edges, once, and stays off by default', async () => {
  const g = await graph();
  assert.equal(g.stats.expansionPairs, 0);
  const x = await buildViewGraph(scene.keyframes, cpuViewGraphBackend(), {
    ...OPTIONS,
    expansion: true,
  });
  const first = new Set(g.edges.map((e) => `${e.first}:${e.second}`));
  for (const key of first) assert.ok(x.edges.some((e) => `${e.first}:${e.second}` === key));
  assert.equal(
    x.edges.filter((e) => e.source === 'expansion').length,
    x.edges.length - g.edges.length,
  );
  assert.deepEqual(
    x.components.map((c) => c.viewIds.length),
    [20, 10],
  );
});

test('8-bit descriptors change the inlier counts little', async () => {
  const g = await graph();
  const f = await buildViewGraph(scene.keyframes, cpuViewGraphBackend(), {
    ...OPTIONS,
    descriptorStorage: 'float32',
  });
  const key = (e: { first: number; second: number }) => `${e.first}:${e.second}`;
  const fl = new Map(f.edges.map((e) => [key(e), e.inliers]));
  let both = 0,
    sum = 0;
  for (const e of g.edges) {
    const v = fl.get(key(e));
    if (v === undefined) continue;
    both++;
    sum += Math.abs(v - e.inliers) / v;
  }
  assert.ok(both >= 0.9 * f.edges.length);
  assert.ok(sum / both < 0.05, `mean relative change ${sum / both}`);
  assert.deepEqual(
    g.components.map((c) => c.viewIds),
    f.components.map((c) => c.viewIds),
  );
});

test('8-bit store: round(512 v) clamped to 255, a quarter of the float bytes', () => {
  const q = quantizeDescriptors(Float32Array.of(0, 0.001, 0.5, 0.9, 1, 0.2567));
  assert.deepEqual([...q], [0, 1, 255, 255, 255, 131]);
  assert.ok(Math.abs(dequantizeDescriptors(Uint8Array.of(131))[0]! - 0.2559) < 1e-3);
  const rows = new Float32Array(256).fill(0.05);
  const bits = new DescriptorStore('uint8'),
    floats = new DescriptorStore('float32');
  bits.add(1, rows);
  floats.add(1, rows);
  assert.equal(bits.bytes * 4, floats.bytes);
  assert.equal(bits.view(1).count, 2);
  assert.throws(() => bits.add(1, rows), RangeError);
  assert.throws(() => bits.add(2, new Float32Array(100)), RangeError);
});

test('CPU matching keeps mutual ratio matches in source order', () => {
  const rng = seededRandom(3);
  const a = Float32Array.from({ length: 128 * 6 }, () => rng());
  const b = new Float32Array(128 * 5);
  // b row 0 is a row 4, b row 1 is a row 1 (with a small change), the rest is far.
  b.set(a.subarray(4 * 128, 5 * 128), 0);
  b.set(a.subarray(128, 256), 128);
  b[130] += 0.01;
  for (let i = 256; i < b.length; i++) b[i] = rng() + 5;
  const m = matchDescriptorsCpu(a, b, 0.8);
  assert.deepEqual([...m.matches], [1, 1, 4, 0]);
  assert.ok(m.distances[1]! === 0 && m.distances[0]! > 0);
});

function frame(
  viewId: number,
  videoId: string,
  time: number,
  global: number[],
  frameKey = `k${viewId}`,
): ViewGraphKeyframe {
  return {
    viewId,
    videoId,
    frameKey,
    time,
    camera: { fx: 1, fy: 1, cx: 0, cy: 0 },
    keypoints: new Float64Array(0),
    descriptors: new Float32Array(0),
    global: Float32Array.from(global),
  };
}

test('candidates: temporal neighbours, the exclusion window, the reserve and duplicates', () => {
  // Video a: 8 keyframes with one global descriptor. Video b: 6 keyframes, nearly orthogonal.
  const frames: ViewGraphKeyframe[] = [];
  for (let i = 0; i < 8; i++) frames.push(frame(i, 'a', i, [1, 0.01 * i, 0]));
  for (let i = 0; i < 6; i++) frames.push(frame(10 + i, 'b', i, [0, 0.01 * i, 1]));
  const set = selectCandidates(frames, {
    temporalNeighbours: 1,
    retrievalK: 2,
    exclusionWindow: 2,
    reservePerVideoPair: 2,
  });
  const has = (a: number, b: number) => set.source.has(`${a}:${b}`);
  // Temporal: i and i + 1 in a video.
  assert.equal(set.source.get('0:1'), 'temporal');
  assert.equal(set.source.get('10:11'), 'temporal');
  assert.ok(!has(0, 2) || set.source.get('0:2') !== 'retrieval');
  // Retrieval of keyframe 0 skips keyframes 1 and 2 (window 2): it keeps 3 and 4.
  assert.deepEqual(set.retrieved.get(0), [3, 4]);
  assert.ok(has(0, 3) && has(0, 4));
  // The reserve keeps the 2 best pairs between a and b although retrieval never ranks b.
  const reserve = [...set.source].filter(([, why]) => why === 'reserve');
  assert.equal(reserve.length, 2);
  for (const [key] of reserve) assert.ok(Number(key.split(':')[1]) >= 10);
  // Pairs are unique and sorted.
  const keys = set.pairs.map(([a, b]) => `${a}:${b}`);
  assert.equal(new Set(keys).size, keys.length);
  assert.ok(set.pairs.every(([a, b]) => a < b));
  // A pair with one frameKey is dropped, in the temporal and the retrieval step alike.
  const dup = frames.map((f) => (f.viewId === 11 ? { ...f, frameKey: 'k10' } : f));
  const without = selectCandidates(dup, {
    temporalNeighbours: 1,
    retrievalK: 2,
    exclusionWindow: 2,
  });
  assert.ok(!without.source.has('10:11'));
});

test('candidates: equal similarities break ties by the lower view id', () => {
  const frames = [0, 1, 2, 3, 4].map((i) => frame(i, `v${i}`, 0, [1, 0]));
  const set = selectCandidates(frames, { retrievalK: 2, reservePerVideoPair: 0 });
  assert.deepEqual(set.retrieved.get(4), [0, 1]);
  assert.deepEqual(set.retrieved.get(0), [1, 2]);
});

test('components: size rule, alone, connected and not connected tags', () => {
  const kf = (viewId: number, videoId: string) => ({ viewId, videoId });
  const frames = [
    // Component 0: videos a and b, 6 keyframes. Video b has 2 of 6 keyframes elsewhere.
    ...[0, 1, 2].map((i) => kf(i, 'a')),
    ...[3, 4, 5].map((i) => kf(i, 'b')),
    // Video b keeps three more keyframes in a pair component too small to reconstruct.
    ...[6, 7, 8].map((i) => kf(i, 'b')),
    // Video c: alone, 5 keyframes in one component.
    ...[9, 10, 11, 12, 13].map((i) => kf(i, 'c')),
    // Video d: two keyframes only.
    kf(14, 'd'),
    kf(15, 'd'),
  ];
  const ring = (ids: number[]) => ids.slice(1).map((v, i) => ({ first: ids[i]!, second: v }));
  const edges = [
    ...ring([0, 1, 2, 3, 4, 5]),
    ...ring([6, 7, 8]),
    ...ring([9, 10, 11, 12, 13]),
    ...ring([14, 15]),
  ];
  const { components, videos } = connectedComponents(frames, edges);
  assert.deepEqual(
    components.map((c) => [c.index, c.viewIds.length, c.reconstructed]),
    [
      [0, 6, true],
      [1, 5, true],
      [2, 3, false],
      [3, 2, false],
    ],
  );
  const by = Object.fromEntries(videos.map((v) => [v.videoId, v]));
  assert.equal(by.a!.tag, 'connected');
  assert.equal(by.b!.tag, 'connected');
  assert.equal(by.b!.share, 0.5);
  assert.equal(by.c!.tag, 'alone');
  assert.equal(by.d!.tag, 'not connected');
  // Order of the edges does not change the components.
  const reversed = connectedComponents(frames, [...edges].reverse());
  assert.deepEqual(reversed.components, components);
});
