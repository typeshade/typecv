// Verifies: docs/dense-mvs.md section 10.5 (the visibility labeling: the min-cut, the graph, the
// terms and the component rule).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MinCutGraph,
  labelDenseCloud,
  landmarkSupport,
  nearestNeighbours,
  type LabelInput,
} from '../src/reconstruction/dense-label.ts';
import { TSDF_WEIGHT_SCALE, encodeTsdfNormal, tsdfTally } from '../src/reconstruction/tsdf.ts';

/** A small deterministic generator (mulberry32) so a failure reproduces. */
function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test('the min-cut equals the least energy of every labeling of small random graphs', () => {
  const rand = random(7);
  for (let trial = 0; trial < 300; trial++) {
    const n = 1 + Math.floor(rand() * 10);
    const source: number[] = [],
      sink: number[] = [];
    const edges: [number, number, number][] = [];
    const graph = new MinCutGraph(n);
    for (let i = 0; i < n; i++) {
      const s = rand() < 0.3 ? 0 : Math.floor(rand() * 8) / 2,
        t = rand() < 0.3 ? 0 : Math.floor(rand() * 8) / 2;
      source.push(s);
      sink.push(t);
      graph.addTerminal(i, s, t);
    }
    for (let e = 0; e < 2 * n; e++) {
      const i = Math.floor(rand() * n),
        j = Math.floor(rand() * n);
      if (i === j) continue;
      const c = Math.floor(rand() * 6) / 2;
      edges.push([i, j, c]);
      graph.addEdge(i, j, c);
    }
    // The energy of a labeling: a source-side node pays its sink capacity, a sink-side node its
    // source capacity, and an edge between the sides its capacity.
    const energy = (onSource: (i: number) => boolean) => {
      let E = 0;
      for (let i = 0; i < n; i++) E += onSource(i) ? sink[i]! : source[i]!;
      for (const [i, j, c] of edges) if (onSource(i) !== onSource(j)) E += c;
      return E;
    };
    let least = Infinity;
    for (let mask = 0; mask < 1 << n; mask++)
      least = Math.min(
        least,
        energy((i) => ((mask >> i) & 1) === 1),
      );
    const { flow, side } = graph.maxflow();
    assert.ok(Math.abs(flow - least) < 1e-9, `trial ${trial}: flow ${flow}, least ${least}`);
    assert.ok(Math.abs(energy((i) => side[i] === 1) - least) < 1e-9, `trial ${trial}: the cut`);
  }
});

test('the neighbours are the k nearest, ascending, ties to the lower index', () => {
  const rand = random(3);
  const n = 400;
  const P = Float32Array.from({ length: 3 * n }, () => rand());
  const k = 8;
  const { index, d2 } = nearestNeighbours(P, k);
  for (let i = 0; i < n; i += 7) {
    const all: [number, number][] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const q =
        (P[3 * i]! - P[3 * j]!) ** 2 +
        (P[3 * i + 1]! - P[3 * j + 1]!) ** 2 +
        (P[3 * i + 2]! - P[3 * j + 2]!) ** 2;
      all.push([q, j]);
    }
    all.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    for (let m = 0; m < k; m++) {
      assert.equal(index[i * k + m], all[m]![1]);
      assert.ok(Math.abs(d2[i * k + m]! - all[m]![0]) < 1e-12);
    }
  }
});

/** A plane z = 0 of `side` × `side` points, a floater patch above it, and an isolated cluster. */
function scene(options: { floaterFree: number; floaterWeight: number }) {
  const positions: number[] = [],
    normals: number[] = [],
    weights: number[] = [],
    free: number[] = [],
    tallies: number[] = [];
  const add = (p: number[], weight: number, samples: number, views: number, freeCount: number) => {
    positions.push(...p);
    normals.push(encodeTsdfNormal([0, 0, 1]));
    weights.push(Math.round(weight * TSDF_WEIGHT_SCALE));
    free.push(freeCount);
    let tally = 0;
    for (let s = 0; s < samples; s++) tally = tsdfTally(tally, s % views);
    tallies.push(tally);
  };
  const side = 30;
  for (let y = 0; y < side; y++)
    for (let x = 0; x < side; x++) add([x * 0.01, y * 0.01, 0], 4, 8, 4, 0);
  // A floater: 6 × 6 points 0.2 above the plane, with the given weight and free-space count.
  for (let y = 0; y < 6; y++)
    for (let x = 0; x < 6; x++)
      add([0.1 + x * 0.01, 0.1 + y * 0.01, 0.2], options.floaterWeight, 4, 2, options.floaterFree);
  // A small confident cluster far away: 3 × 3 points.
  for (let y = 0; y < 3; y++)
    for (let x = 0; x < 3; x++) add([2 + x * 0.01, y * 0.01, 0], 4, 8, 4, 0);
  const count = normals.length;
  const input: LabelInput = {
    count,
    positions: Float32Array.from(positions),
    normals: Uint32Array.from(normals),
    weights: Uint32Array.from(weights),
    free: Uint32Array.from(free),
    tallies: Uint32Array.from(tallies),
  };
  return { input, plane: side * side, floater: 36, cluster: 9 };
}

test('a carved floater is cut, a small component goes unless a landmark holds it', () => {
  const { input, plane, floater, cluster } = scene({ floaterFree: 40, floaterWeight: 2 });
  const { keep, report } = labelDenseCloud(input, [], { minComponent: 20 });
  for (let i = 0; i < plane; i++) assert.equal(keep[i], 1, `plane point ${i}`);
  for (let i = plane; i < plane + floater; i++) assert.equal(keep[i], 0, `floater point ${i}`);
  for (let i = plane + floater; i < plane + floater + cluster; i++) assert.equal(keep[i], 0);
  assert.equal(report.cutByVisibility, floater);
  assert.equal(report.removedByComponents, cluster);
  assert.equal(report.kept, plane);
  // A landmark at the cluster keeps it.
  const held = labelDenseCloud(input, [{ xyz: [2.01, 0.01, 0], radius: 0.005 }], {
    minComponent: 20,
  });
  for (let i = plane + floater; i < plane + floater + cluster; i++) assert.equal(held.keep[i], 1);
  assert.equal(held.report.supportingLandmarks, 1);
  assert.equal(held.report.kept, plane + cluster);
});

test('an uncarved confident floater stays: the terms, not the position, decide', () => {
  const { input, plane, floater } = scene({ floaterFree: 0, floaterWeight: 4 });
  const { keep } = labelDenseCloud(input, [], { minComponent: 20 });
  for (let i = 0; i < plane + floater; i++) assert.equal(keep[i], 1);
});

test('the labels do not depend on the order of the points', () => {
  const { input } = scene({ floaterFree: 40, floaterWeight: 2 });
  const a = labelDenseCloud(input, [], { minComponent: 20 });
  const n = input.count;
  const perm = Array.from({ length: n }, (_, i) => (i * 37) % n);
  assert.equal(new Set(perm).size, n);
  const shuffled: LabelInput = {
    count: n,
    positions: Float32Array.from(
      perm.flatMap((i) => [...input.positions.subarray(3 * i, 3 * i + 3)]),
    ),
    normals: Uint32Array.from(perm, (i) => input.normals[i]!),
    weights: Uint32Array.from(perm, (i) => input.weights[i]!),
    free: Uint32Array.from(perm, (i) => input.free[i]!),
    tallies: Uint32Array.from(perm, (i) => input.tallies[i]!),
  };
  const b = labelDenseCloud(shuffled, [], { minComponent: 20 });
  perm.forEach((i, s) => assert.equal(b.keep[s], a.keep[i]));
});

test('landmark support marks the points within each radius', () => {
  const P = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 0.5, 0]);
  const { supported, landmarks } = landmarkSupport(P, [
    { xyz: [0.05, 0, 0], radius: 0.1 },
    { xyz: [5, 5, 5], radius: 0.1 },
  ]);
  assert.deepEqual([...supported], [1, 0, 0]);
  assert.equal(landmarks, 1);
});

test('the options are checked', () => {
  const { input } = scene({ floaterFree: 0, floaterWeight: 4 });
  assert.throws(() => labelDenseCloud(input, [], { k: 0 }), RangeError);
  assert.throws(() => labelDenseCloud(input, [], { minComponent: 0 }), RangeError);
  assert.throws(() => labelDenseCloud(input, [], { confidenceScale: 0 }), RangeError);
});

/** The maximum flow by Edmonds-Karp on a dense capacity matrix: source n, sink n + 1. */
function edmondsKarp(n: number, cap: Float64Array): number {
  const N = n + 2,
    s = n,
    t = n + 1;
  let flow = 0;
  for (;;) {
    const prev = new Int32Array(N).fill(-1);
    prev[s] = s;
    const queue = [s];
    for (let q = 0; q < queue.length && prev[t] < 0; q++) {
      const u = queue[q]!;
      for (let v = 0; v < N; v++)
        if (prev[v]! < 0 && cap[u * N + v]! > 1e-12) {
          prev[v] = u;
          queue.push(v);
        }
    }
    if (prev[t]! < 0) return flow;
    let b = Infinity;
    for (let v = t; v !== s; v = prev[v]!) b = Math.min(b, cap[prev[v]! * N + v]!);
    for (let v = t; v !== s; v = prev[v]!) {
      cap[prev[v]! * N + v]! -= b;
      cap[v * N + prev[v]!]! += b;
    }
    flow += b;
  }
}

test('the min-cut equals Edmonds-Karp on random graphs of 200 nodes', () => {
  const rand = random(11);
  for (let trial = 0; trial < 10; trial++) {
    const n = 200,
      N = n + 2;
    const cap = new Float64Array(N * N);
    const graph = new MinCutGraph(n);
    for (let i = 0; i < n; i++) {
      const a = rand() * 2,
        b = rand() * 2;
      graph.addTerminal(i, a, b);
      cap[n * N + i] = a;
      cap[i * N + n + 1] = b;
    }
    for (let e = 0; e < 6 * n; e++) {
      const i = Math.floor(rand() * n),
        j = (i + 1 + Math.floor(rand() * 12)) % n;
      const c = rand();
      graph.addEdge(i, j, c);
      cap[i * N + j]! += c;
      cap[j * N + i]! += c;
    }
    const expected = edmondsKarp(n, cap);
    const { flow } = graph.maxflow();
    assert.ok(Math.abs(flow - expected) < 1e-6, `trial ${trial}: ${flow} against ${expected}`);
  }
});
