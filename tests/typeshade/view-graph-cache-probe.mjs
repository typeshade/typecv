import { ShadeDevice } from '../../src/gpu/device.ts';
import { gpuViewGraphBackend } from '../../src/gpu/view-graph.ts';
import {
  DescriptorStore,
  buildViewGraph,
  descriptorStoreEvidence,
} from '../../src/reconstruction/view-graph.ts';
import { createViewGraphDescriptorCache } from '../../src/gpu/view-graph-descriptors.ts';

function graphFrames() {
  const camera = { fx: 600, fy: 600, cx: 320, cy: 240 };
  return Array.from({ length: 4 }, (_, viewId) => ({
    viewId,
    videoId: 'a',
    frameKey: `frame-${viewId}`,
    time: viewId * 0.2,
    camera,
    keypoints: Float64Array.from({ length: 32 }, (_, k) => {
      const j = k >> 1,
        z = 4 + ((j * 7) % 11) * 0.2;
      return k % 2
        ? (camera.fy * (Math.floor(j / 4) - 1.5) * 0.3) / z + camera.cy
        : (camera.fx * (((j % 4) - 1.5) * 0.5 - viewId * 0.25)) / z + camera.cx;
    }),
    descriptors: descriptors(viewId, 16),
    global: Float32Array.of(1, 0),
  }));
}

function descriptors(view, count = 12) {
  return Float32Array.from(
    { length: count * 128 },
    (_, i) => (1 + ((i * 31 + (i % 128) * 7) % 127)) / 512 + (view % 3) / 4096,
  );
}
const plainMatches = (rows) =>
  rows.map((r) => ({ matches: Array.from(r.matches), distances: Array.from(r.distances) }));

export async function cacheWorkload({ backend = 'cpu', bytes = 24576 } = {}) {
  const d = await ShadeDevice.open({ backend, poolBytes: 0, workingBytes: 256 * 1024 });
  const cache = gpuViewGraphBackend(d, 4, 16, { descriptorCacheBytes: bytes });
  const store = new DescriptorStore();
  for (let i = 0; i < 8; i++) store.add(i, descriptors(i));
  const output = [];
  try {
    for (const [a, bs] of [
      [0, [1, 2]],
      [1, [2, 3]],
      [2, [3, 4]],
      [3, [4, 5]],
      [0, [1, 2]],
    ])
      output.push(
        ...plainMatches(
          await cache.match(
            store.view(a),
            bs.map((b) => store.view(b)),
            { ratio: 0.8 },
          ),
        ),
      );
    const beforeRelease = cache.descriptorCacheStatistics();
    await cache.releaseMatching();
    const afterRelease = cache.descriptorCacheStatistics();
    return { output, beforeRelease, afterRelease, device: d.statistics };
  } finally {
    await cache.releaseMatching();
    await d.dispose();
  }
}

export async function cacheOwnershipAndFailure({ backend = 'cpu' } = {}) {
  const d = await ShadeDevice.open({ backend, poolBytes: 0, workingBytes: 256 * 1024 });
  const cache = gpuViewGraphBackend(d, 4, 16, { descriptorCacheBytes: 24576 });
  const store = new DescriptorStore(),
    floats = new DescriptorStore('float32');
  for (let i = 0; i < 4; i++) {
    store.add(i, descriptors(i));
    floats.add(i, descriptors(i));
  }
  const evidence = descriptorStoreEvidence(store.view(0));
  const sameEvidence = evidence === descriptorStoreEvidence(store.view(0));
  const mutableExcluded = descriptorStoreEvidence(floats.view(0)) === undefined;
  const edited = store.view(0);
  edited.floatDescriptors = () => descriptors(2);
  const changedMethodExcluded = descriptorStoreEvidence(edited) === undefined;
  try {
    const a = store.view(0),
      b = store.view(1);
    const pending = [
      cache.match(a, [b], { ratio: 0.8 }),
      cache.match(a, [store.view(2)], { ratio: 0.8 }),
    ];
    cache.releaseMatching();
    const concurrent = (await Promise.all(pending)).flatMap(plainMatches);
    cache.releaseMatching();
    const afterConcurrent = d.statistics.liveBuffers;
    await cache.match(a, [b], { ratio: 0.8 });
    const allocate = d.allocate.bind(d);
    const sentinel = new Error('descriptor allocation failure');
    d.allocate = () => {
      throw sentinel;
    };
    let allocationFailure = false;
    try {
      await cache.match(a, [store.view(3)], { ratio: 0.8 });
    } catch (e) {
      allocationFailure = e === sentinel;
    } finally {
      d.allocate = allocate;
    }
    const afterAllocationFailure = d.statistics.liveBuffers;
    const validAgain = plainMatches(await cache.match(a, [b], { ratio: 0.8 }));
    const bad = {
      viewId: 99,
      count: 12,
      floatDescriptors: () => new Float32Array(12 * 128).fill(NaN),
    };
    let numericalFailure = false;
    try {
      await cache.match(a, [bad], { ratio: 0.8 });
    } catch {
      numericalFailure = true;
    }
    const afterNumericalFailure = d.statistics.liveBuffers;
    await cache.match(a, [b], { ratio: 0.8 });
    let emptyFailure = false;
    try {
      await cache.match(
        {
          viewId: 98,
          count: 0,
          floatDescriptors() {
            throw sentinel;
          },
        },
        [],
        { ratio: 0.8 },
      );
    } catch (e) {
      emptyFailure = e === sentinel;
    }
    const afterEmptyFailure = d.statistics.liveBuffers;
    const mutable = floats.view(0),
      other = floats.view(1);
    const beforeMutable = plainMatches(await cache.match(mutable, [other], { ratio: 0.8 }));
    mutable.floatDescriptors().fill(0);
    const afterMutable = plainMatches(await cache.match(mutable, [other], { ratio: 0.8 }));
    cache.releaseMatching();
    return {
      sameEvidence,
      mutableExcluded,
      changedMethodExcluded,
      concurrent,
      afterConcurrent,
      allocationFailure,
      afterAllocationFailure,
      validAgain,
      numericalFailure,
      afterNumericalFailure,
      emptyFailure,
      afterEmptyFailure,
      beforeMutable,
      afterMutable,
      final: d.statistics,
      cache: cache.descriptorCacheStatistics(),
    };
  } finally {
    cache.releaseMatching();
    await d.dispose();
  }
}

export async function cacheGraph({ backend = 'cpu', bytes = 32768, fail = false } = {}) {
  const d = await ShadeDevice.open({ backend, poolBytes: 0 });
  const cache = gpuViewGraphBackend(d, 4, 16, { descriptorCacheBytes: bytes });
  const sentinel = new Error('progress failure');
  let failure = false;
  try {
    let graph;
    try {
      graph = await buildViewGraph(graphFrames(), cache, {
        temporalNeighbours: 2,
        retrievalK: 0,
        reservePerVideoPair: 0,
        minMatches: 4,
        minEdgeInliers: 4,
        minComponentViews: 2,
        verification: { ransacIterations: 32, pairRefinementIterations: 0 },
        onActivity: fail
          ? (e) => {
              if (e.stage === 'matching' && e.phase === 'complete') throw sentinel;
            }
          : undefined,
      });
    } catch (e) {
      if (e !== sentinel) throw e;
      failure = true;
    }
    return {
      graph: graph && { ...graph, timing: undefined },
      failure,
      cache: cache.descriptorCacheStatistics(),
      device: d.statistics,
    };
  } finally {
    cache.releaseMatching();
    await d.dispose();
  }
}

export async function cacheWorkingPressure() {
  const d = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, workingBytes: 32768 });
  const cache = createViewGraphDescriptorCache(d, 8192),
    store = new DescriptorStore();
  for (let i = 0; i < 5; i++) store.add(i, descriptors(i, 4));
  try {
    for (let i = 0; i < 4; i++) cache.acquire(store.view(i)).release(true);
    const before = cache.statistics();
    const pressure = d.allocate(7168);
    const after = cache.statistics();
    d.drop(pressure);
    const a = cache.acquire(store.view(0)),
      b = cache.acquire(store.view(1));
    let protectedFailure = false;
    try {
      d.allocate(7680);
    } catch {
      protectedFailure = true;
    }
    cache.clear();
    const protectedBytes = cache.statistics().residentBytes;
    a.release(true);
    b.release(true);
    return {
      before,
      after,
      protectedFailure,
      protectedBytes,
      final: cache.statistics(),
      device: d.statistics,
    };
  } finally {
    cache.clear();
    await d.dispose();
  }
}
