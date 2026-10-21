import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, reconstructMultiVideoGpu } from '../../dist/internal/gpu.js';
import { quantizeDescriptors } from '../../src/reconstruction/view-graph.ts';
import { pathScene } from '../view-graph-scene.ts';

// Verifies: docs/multi-video.md 2.4 (ids), 4.1, 4.2 and step 6 (the pipeline on a device, the
// `canonical: false` switch of 7.1) on the CPU tier of the device.

const OPTIONS = {
  graph: {
    retrievalK: 3,
    temporalNeighbours: 2,
    exclusionWindow: 2,
    reservePerVideoPair: 2,
    verification: { seed: 7, ransacIterations: 100 },
  },
  mapper: { seed: 7, ransacIterations: 100, finalBundleIterations: 4 },
};

const scene = pathScene(
  [
    { label: 'orbit', cloud: 0, views: 6, from: -30, to: 30, radius: 9, height: -1 },
    { label: 'ascent', cloud: 0, views: 6, from: 5, to: 40, radius: 9, height: -3 },
    { label: 'elsewhere', cloud: 1, views: 6, from: -30, to: 30, radius: 9, height: -1 },
  ],
  { points: 50 },
);
// The worker's form of a keyframe: 8-bit descriptors, no view id.
const input = scene.keyframes.map(({ viewId: _id, descriptors, ...k }) => ({
  ...k,
  descriptors: quantizeDescriptors(descriptors),
}));
const permute = (list, seed) => {
  const out = [...list];
  let s = seed;
  for (let i = out.length - 1; i > 0; i--) {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    const j = s % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
};
const keyOf = (r, id) => r.keyframes.find((k) => k.viewId === id).frameKey;
const summary = (r) => ({
  partition: r.components.map((c) => c.viewIds.map((v) => keyOf(r, v)).sort()),
  registered: r.components.map((c) =>
    (c.reconstruction?.views ?? []).map((v) => keyOf(r, v.id)).sort(),
  ),
  order: r.components.map((c) => (c.registrationOrder ?? []).map((v) => keyOf(r, v))),
});

test('a run on the device gives components, and the input order decides nothing', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const stages = [];
    const a = await reconstructMultiVideoGpu(device, input, {
      ...OPTIONS,
      onProgress: (s) => stages.push(s),
    });
    assert.equal(a.canonical, true);
    assert.deepEqual(stages, ['viewGraph', 'component 0', 'component 1']);
    assert.equal(a.components.length, 2);
    assert.deepEqual(
      a.components.map((c) => c.viewIds.length),
      [12, 6],
    );
    for (const c of a.components) {
      assert.equal(c.error, undefined);
      assert.equal(c.reconstruction.views.length, c.viewIds.length);
    }
    assert.ok(a.graph.timing.pairMatchingMs > 0 && a.timing.reconstructionMs > 0);
    const b = await reconstructMultiVideoGpu(device, permute(input, 5), OPTIONS);
    assert.deepEqual(summary(b), summary(a));
    assert.equal(device.statistics.liveBuffers, 0);
  } finally {
    await device.dispose();
  }
});

test('canonical: false gives the position in the input list as the view id', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const shuffled = permute(input, 9);
    const r = await reconstructMultiVideoGpu(device, shuffled, { ...OPTIONS, canonical: false });
    assert.equal(r.canonical, false);
    assert.deepEqual(
      r.keyframes.map((k) => k.viewId),
      shuffled.map((_, i) => i),
    );
    assert.equal(r.components.length, 2);
  } finally {
    await device.dispose();
  }
});
