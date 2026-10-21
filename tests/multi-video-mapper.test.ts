import test from 'node:test';
import assert from 'node:assert/strict';
import { cameraProjection } from '../src/geometry/camera.ts';
import { transformPoint } from '../src/geometry/pose.ts';
import { bundleAdjust } from '../src/reconstruction/bundle-adjustment.ts';
import {
  executeReconstructionJob,
  incrementalSteps,
  type IncrementalResult,
  type ReconstructionJob,
} from '../src/reconstruction/mapper.ts';
import {
  componentInput,
  focalGroupsOf,
  reconstructComponent,
} from '../src/reconstruction/component-mapper.ts';
import {
  buildViewGraph,
  cpuViewGraphBackend,
  type ViewGraph,
  type ViewGraphKeyframe,
} from '../src/reconstruction/view-graph.ts';
import type { SparseReconstruction } from '../src/reconstruction/types.ts';
import { seededRandom } from '../src/geometry/numerics.ts';
import { pathScene, type PathSpec } from './view-graph-scene.ts';
import { scene } from './scene.ts';

// Verifies: docs/multi-video.md 3.4 (preVerified), 4.2 (a component), 4.4 (tie-breaks),
// 7.1 (CPU tier of the shuffle gate) and 2.3 item 4 (focal groups).

const PATHS: PathSpec[] = [
  { label: 'orbit', cloud: 0, views: 7, from: -30, to: 30, radius: 9, height: -1 },
  { label: 'ascent', cloud: 0, views: 7, from: 5, to: 40, radius: 9, height: -3 },
  { label: 'lateral', cloud: 0, views: 7, from: -45, to: -5, radius: 10, height: -2 },
  { label: 'elsewhere', cloud: 1, views: 7, from: -30, to: 30, radius: 9, height: -1 },
];
const GRAPH_OPTIONS = {
  retrievalK: 4,
  temporalNeighbours: 2,
  exclusionWindow: 3,
  reservePerVideoPair: 2,
  verification: { seed: 7, ransacIterations: 200 },
};
const MAPPER = { seed: 7, ransacIterations: 200, finalBundleIterations: 10 };
const world = pathScene(PATHS);

/** The graph of the keyframes in the given order. */
async function graphOf(keyframes: readonly ViewGraphKeyframe[]): Promise<ViewGraph> {
  return buildViewGraph(keyframes, cpuViewGraphBackend(), GRAPH_OPTIONS);
}

function countJobs(machine: Generator<ReconstructionJob, IncrementalResult, unknown>) {
  const counts: Record<string, number> = {};
  let step = machine.next();
  while (!step.done) {
    counts[step.value.kind] = (counts[step.value.kind] ?? 0) + 1;
    step = machine.next(executeReconstructionJob(step.value));
  }
  return { counts, result: step.value };
}

let base: { graph: ViewGraph; result: IncrementalResult } | undefined;
async function baseline() {
  if (base) return base;
  const graph = await graphOf(world.keyframes);
  return (base = { graph, result: reconstructComponent(graph, world.keyframes, 0, MAPPER) });
}

test('a component reconstructs from verified edges with no second verification', async () => {
  const { graph } = await baseline();
  const [big, small] = graph.components;
  assert.equal(big!.viewIds.length, 21);
  assert.equal(small!.viewIds.length, 7);
  const edges = graph.edges.filter((e) => big!.viewIds.includes(e.first));
  assert.ok(edges.length > 21);

  const input = componentInput(graph, world.keyframes, 0, MAPPER);
  const reused = countJobs(incrementalSteps(input.images, input.pairs, input.options));
  assert.equal(reused.counts.verifyPair ?? 0, 0);
  assert.equal(reused.result.reconstruction.views.length, 21);
  assert.equal(reused.result.diagnostics.pairVerification.length, edges.length);

  // The same edges as plain input pairs are verified again, once for each edge.
  const again = countJobs(
    incrementalSteps(input.images, input.pairs, { ...input.options, preVerified: undefined }),
  );
  assert.equal(again.counts.verifyPair, edges.length);
  assert.equal(again.result.reconstruction.views.length, 21);
  console.log(
    `verifyPair jobs: ${reused.counts.verifyPair ?? 0} reused, ${again.counts.verifyPair} plain`,
  );

  // The small component is its own call.
  const alone = reconstructComponent(graph, world.keyframes, 1, MAPPER);
  assert.equal(alone.reconstruction.views.length, 7);
});

test('a pre-verified entry that names no input pair, or repeats one, is refused', async () => {
  const { graph } = await baseline();
  const input = componentInput(graph, world.keyframes, 0, MAPPER);
  const rows = input.options.preVerified!;
  assert.throws(
    () => countJobs(incrementalSteps(input.images, input.pairs.slice(1), input.options)),
    RangeError,
  );
  assert.throws(
    () =>
      countJobs(
        incrementalSteps(input.images, input.pairs, {
          ...input.options,
          preVerified: [...rows, rows[0]!],
        }),
      ),
    RangeError,
  );
  const eligible = rows.findIndex((r) => r.verification.seedEligible);
  assert.ok(eligible >= 0);
  const stripped = rows.map((r, i) => (i === eligible ? { ...r, geometry: undefined } : r));
  assert.throws(
    () =>
      countJobs(
        incrementalSteps(input.images, input.pairs, { ...input.options, preVerified: stripped }),
      ),
    RangeError,
  );
});

test('the seed is the best pair over all edges of the component, ties by view id', async () => {
  const { graph, result } = await baseline();
  const seeds = graph.edges
    .filter(
      (e) =>
        e.seedEligible && e.geometry && result.reconstruction.views.some((v) => v.id === e.first),
    )
    .map((e) => ({
      pair: [e.first, e.second],
      score:
        e.geometry!.positiveDepthCount * Math.min(10, e.geometry!.diagnostics.medianParallaxDeg),
    }))
    .sort((a, b) => b.score - a.score || a.pair[0]! - b.pair[0]! || a.pair[1]! - b.pair[1]!);
  assert.deepEqual(result.diagnostics.initialPair, seeds[0]!.pair);
});

function reversed<T>(list: readonly T[]): T[] {
  return [...list].reverse();
}
function permuted<T>(list: readonly T[], seed: number): T[] {
  const rng = seededRandom(seed),
    out = [...list];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}
function summary(graph: ViewGraph, result: IncrementalResult) {
  return {
    components: graph.components.map((c) => [c.viewIds, c.videoIds]),
    registered: result.reconstruction.views.map((v) => v.id).sort((a, b) => a - b),
    order: result.diagnostics.registrationOrder,
    seed: result.diagnostics.initialPair,
    landmarks: result.reconstruction.landmarks.length,
    cameras: result.reconstruction.views.map((v) => [
      v.id,
      Array.from(v.pose.R),
      Array.from(v.pose.t),
    ]),
  };
}

test('shuffle gate, CPU tier: three input orders give one map', async () => {
  const { graph, result } = await baseline();
  const reference = summary(graph, result);
  assert.equal(reference.registered.length, 21);
  for (const keyframes of [reversed(world.keyframes), permuted(world.keyframes, 5)]) {
    const g = await graphOf(keyframes);
    const r = reconstructComponent(g, keyframes, 0, MAPPER);
    // Same keyframes, components, registered set, order, seed pair and cameras. Exactly equal.
    assert.deepEqual(summary(g, r), reference);
  }
});

test('the shuffle gate can fail: view ids in arrival order change the map', async () => {
  const { result } = await baseline();
  const reference = JSON.stringify(result.diagnostics.registrationOrder.map((id) => keyOf(id)));
  function keyOf(id: number): string {
    return world.keyframes.find((k) => k.viewId === id)!.frameKey;
  }
  let differs = false;
  for (const seed of [1, 2, 3]) {
    // Non-canonical: the id is the position in the arrival list.
    const arrival = permuted(world.keyframes, seed).map((k, i) => ({ ...k, viewId: i }));
    const g = await graphOf(arrival);
    const r = reconstructComponent(g, arrival, 0, MAPPER);
    const order = r.diagnostics.registrationOrder.map((id) => arrival[id]!.frameKey);
    if (JSON.stringify(order) !== reference) differs = true;
  }
  assert.ok(differs, 'arrival-order ids must change the registration order');
});

test('focal groups: one group for each video, by the rank of the videoId', () => {
  const groups = focalGroupsOf(reversed(world.keyframes));
  assert.equal(groups.size, world.keyframes.length);
  assert.equal(new Set(groups.values()).size, 4);
  assert.deepEqual(groups, focalGroupsOf(world.keyframes));
  const videos = [...new Set(world.keyframes.map((k) => k.videoId))].sort();
  for (const k of world.keyframes) assert.equal(groups.get(k.viewId), videos.indexOf(k.videoId));
});

/** Two cameras with different true focal lengths; the model starts with one too short and one too long. */
function twoCameraScene(focalGroups?: boolean) {
  const s = scene({ count: 80, views: 6, noise: 0, seed: 17 });
  const trueFx = (i: number) => (i < 3 ? 850 : 1000);
  const views = s.registered.map((v, i) => ({
    ...v,
    camera: { ...v.camera, fx: trueFx(i), fy: trueFx(i) },
  }));
  const landmarks = s.points.map((xyz, k) => ({
    trackId: k,
    xyz: xyz.slice(),
    observations: views.map((v) => ({
      viewId: v.id,
      featureId: k,
      pixel: cameraProjection(v.camera, transformPoint(v.pose, xyz))!.pixel as [number, number],
    })),
  }));
  const start: SparseReconstruction = {
    views: views.map((v, i) => {
      const wrong = i < 3 ? 0.93 : 1.06;
      return { ...v, camera: { ...v.camera, fx: v.camera.fx * wrong, fy: v.camera.fy * wrong } };
    }),
    landmarks,
  };
  const groups = new Map(views.map((v, i) => [v.id, i < 3 ? 0 : 1]));
  return {
    start,
    views,
    options: {
      iterations: 40,
      refineSharedFocal: true,
      focalGroups: focalGroups ? groups : undefined,
    },
  };
}

test('BA with focal groups refines one scale for each group', () => {
  const grouped = twoCameraScene(true);
  const result = bundleAdjust(grouped.start, grouped.options);
  result.reconstruction.views.forEach((v, i) => {
    assert.ok(Math.abs(v.camera.fx / grouped.views[i]!.camera.fx - 1) < 1e-3, `view ${i}`);
  });
  assert.ok(result.diagnostics.finalRmsePx < 1e-2);
  // One shared scale cannot give both cameras their focal length.
  const shared = twoCameraScene(false);
  const one = bundleAdjust(shared.start, shared.options);
  const worst = Math.max(
    ...one.reconstruction.views.map((v, i) =>
      Math.abs(v.camera.fx / shared.views[i]!.camera.fx - 1),
    ),
  );
  assert.ok(worst > 0.02, `${worst}`);
});

test('BA with one focal group equals BA without groups', () => {
  const plain = twoCameraScene(false);
  const one = new Map(plain.start.views.map((v) => [v.id, 7]));
  const a = bundleAdjust(plain.start, plain.options);
  const b = bundleAdjust(plain.start, { ...plain.options, focalGroups: one });
  assert.deepEqual(b, a);
  assert.throws(
    () =>
      bundleAdjust(plain.start, {
        ...plain.options,
        focalGroups: new Map([[plain.start.views[0]!.id, 0]]),
      }),
    RangeError,
  );
});

test('only the final BA of a component carries the per-video focal groups', async () => {
  const { graph } = await baseline();
  const input = componentInput(graph, world.keyframes, 0, { ...MAPPER, refineSharedFocal: true });
  assert.equal(new Set(input.options.focalGroups!.values()).size, 3);
  const machine = incrementalSteps(input.images, input.pairs, input.options);
  const bundles: (ReadonlyMap<number, number> | undefined)[] = [];
  let step = machine.next();
  while (!step.done) {
    if (step.value.kind === 'bundle') bundles.push(step.value.args[1]?.focalGroups);
    step = machine.next(executeReconstructionJob(step.value));
  }
  assert.ok(bundles.length > 1);
  assert.equal(bundles[bundles.length - 1], input.options.focalGroups);
  assert.ok(bundles.slice(0, -1).every((g) => g === undefined));
  assert.equal(
    componentInput(graph, world.keyframes, 0, { focalPerVideo: false }).options.focalGroups,
    undefined,
  );
});
