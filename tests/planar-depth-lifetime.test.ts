import test from 'node:test';
import assert from 'node:assert/strict';
import { OwnedPlanarMaps, planarLifetimePlan } from '../src/gpu/planar-depth-lifetime.ts';
import type { TsdfView } from '../src/reconstruction/tsdf.ts';

function views(count: number, pixels = 16): TsdfView[] {
  return Array.from({ length: count }, (_, index) => ({
    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    center: [index * 0.1, 0, 0],
    fx: 32,
    fy: 32,
    cx: 1.5,
    cy: 1.5,
    depth: new Float32Array(pixels).fill(index + 1),
    normal: new Uint32Array(pixels).fill(index),
    confidence: new Float32Array(pixels).fill(1),
    rgba: new Uint32Array(pixels).fill(0xff808080),
  }));
}
function completed(view: TsdfView): TsdfView {
  return {
    ...view,
    depth: view.depth.slice(),
    normal: view.normal.slice(),
    confidence: view.confidence?.slice(),
  };
}

test('last-reader ordering preserves every potential frozen reader and bounded paired-graph frontier', () => {
  const count = 240,
    pixels = 16;
  const sources = Array.from({ length: count }, (_, index) => [(index + count / 2) % count]);
  const input = views(count, pixels);
  const kept = input.slice();
  const store = new OwnedPlanarMaps(input, sources);
  assert.equal(store.plan.peakExtraViews, 2);
  assert.deepEqual(store.plan.order.slice(0, 6), [0, 120, 1, 121, 2, 122]);
  for (const reference of store.plan.order) {
    for (const source of store.plan.dependencies[reference]!)
      assert.equal(store.original(source), kept[source]);
    store.complete(reference, completed(store.original(reference)));
  }
  const memory = store.finish();
  assert.equal(memory.inputBytes, 12 * count * pixels);
  assert.equal(memory.peakRetainedBytes, 12 * (count + 2) * pixels);
  assert.equal(memory.peakExtraOriginalViews, 2);
  assert.equal(memory.retiredOriginalViews, count);
  assert.ok(memory.reorderedViews > 0);
  for (let index = 0; index < count; index++) {
    assert.deepEqual(input[index]!.depth, kept[index]!.depth);
    assert.notEqual(input[index]!.depth, kept[index]!.depth);
    assert.equal(kept[index]!.depth[0], index + 1);
    assert.throws(() => store.original(index), /retired/);
  }
});

test('backing-byte accounting deduplicates shared subviews and no-fill aliases', () => {
  const input = views(4, 16);
  const shared = new ArrayBuffer(3 * 16 * 4);
  for (const view of input) {
    Object.assign(view, {
      depth: new Float32Array(shared, 0, 16),
      normal: new Uint32Array(shared, 64, 16),
      confidence: new Float32Array(shared, 128, 16),
    });
  }
  const store = new OwnedPlanarMaps(input, [[1], [2], [3], [0]]);
  for (const reference of store.plan.order)
    store.complete(reference, { ...store.original(reference) });
  const memory = store.finish();
  assert.equal(memory.inputBytes, shared.byteLength);
  assert.equal(memory.peakRetainedBytes, shared.byteLength);
  assert.equal(memory.peakExtraOriginalViews, 0);
  assert.equal(memory.retiredOriginalViews, 4);
});

test('high-degree frozen evidence remains live until all readers finish', () => {
  const count = 9,
    pixels = 8;
  const store = new OwnedPlanarMaps(
    views(count, pixels),
    Array.from({ length: count }, () => Array.from({ length: count }, (_, i) => i)),
  );
  for (const reference of store.plan.order) {
    for (let source = 0; source < count; source++) assert.ok(store.original(source));
    store.complete(reference, completed(store.original(reference)));
  }
  const memory = store.finish();
  assert.equal(memory.peakExtraOriginalViews, count);
  assert.equal(memory.peakRetainedBytes, 24 * count * pixels);
});

test('planner rejects invalid dependencies and incomplete ownership cannot be published', () => {
  assert.throws(() => planarLifetimePlan([[1], [2]]), /Invalid planar lifetime/);
  assert.throws(() => new OwnedPlanarMaps(views(2), [[]]), /view count/);
  const store = new OwnedPlanarMaps(views(2), [[1], [0]]);
  store.complete(0, completed(store.original(0)));
  assert.throws(() => store.finish(), /did not consume/);
  assert.throws(() => store.complete(0, store.original(0)), /completed twice/);
});
