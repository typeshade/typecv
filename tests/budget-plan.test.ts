// Verifies: docs/gpu-budget.md sections 3.3 and 4: the plans cut the bundle adjustment and the
// triangulation to the budget, and never refuse a problem for its size above the floor.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  planBundleResidency,
  planTriangulationChunks,
  TRIANGULATION_OBSERVATION_BYTES,
  TRIANGULATION_TRACK_BYTES,
  type BundleResidencyInput,
} from '../src/gpu/budget-plan.ts';

const MiB = 1024 ** 2;

// The fixed-focal arrays of section 3.1, in bytes.
function bundleInput(
  C: number,
  P: number,
  trackLength: number,
  budgetBytes: number,
  bindingBytes = 2048 * MiB,
  maxKernelElements = 65535 * 64,
): BundleResidencyInput {
  const M = P * trackLength;
  const pointStarts = Uint32Array.from({ length: P + 1 }, (_, p) => p * trackLength);
  const common = [
    8 * (18 * C + 3 * P + 2 * M),
    8 * (12 * C + 3 * P),
    8 * (30 * C + 3 * P),
    8 * 26,
    8 * 3 * Math.ceil(M / 64),
    8 * (12 * C + 3 * P + 26),
    4 * (3 * M + P + 2 * C + 2),
  ];
  const sum = (xs: number[]) => xs.reduce((n, x) => n + x, 0);
  const residentArrays = [...common, 8 * (42 * M + 13 * P + 78 * C), 8 * 36 * C];
  const floorArrays = [...common, 8 * 78 * C, 8 * 36 * C];
  return {
    points: P,
    observations: M,
    pointStarts,
    longestTrack: trackLength,
    budgetBytes,
    bindingBytes,
    maxKernelElements,
    residentBytes: sum(residentArrays) + 8,
    residentArrays,
    floorBytes: sum(floorArrays) + 8,
    floorArrays,
    observationBytes: 336,
    pointBytes: 104,
    windowArrays: (m, p) => [8 * (42 * m + 13 * p + 78 * C)],
  };
}

test('run record d40f8950: 839 cameras, 53,323 points, 8 views each: resident on a 2 GiB adapter', () => {
  const plan = planBundleResidency(bundleInput(839, 53_323, 8, 1024 * MiB));
  assert.equal(plan.mode, 'resident');
  assert.equal(plan.windows.length, 1);
  // 167.6 MB: the table of section 3.2.
  assert.ok(Math.abs(plan.bytes - 167_570_020) < 1e5, String(plan.bytes));
});

test('the same problem streams within the fixed 64 MiB budget, and within 24 MiB', () => {
  for (const budget of [64 * MiB, 24 * MiB]) {
    const input = bundleInput(839, 53_323, 8, budget);
    const plan = planBundleResidency(input);
    assert.equal(plan.mode, 'streamed');
    assert.ok(plan.fits);
    assert.ok(plan.bytes <= budget, `${plan.bytes} <= ${budget}`);
    assert.ok(plan.windows.length > 1);
    // The windows tile the points and the observations in order, whole points each.
    let points = 0,
      observations = 0;
    for (const w of plan.windows) {
      assert.equal(w.pointBase, points);
      assert.equal(w.obsBase, observations);
      assert.equal(w.obsEnd - w.obsBase, 8 * (w.pointEnd - w.pointBase));
      assert.ok(w.obsEnd - w.obsBase <= plan.obsCapacity);
      assert.ok(w.pointEnd - w.pointBase <= plan.pointCapacity);
      points = w.pointEnd;
      observations = w.obsEnd;
    }
    assert.equal(points, 53_323);
    assert.equal(observations, 8 * 53_323);
  }
});

test('the default 128 MiB binding limit splits the blocks array of the d40f8950 problem', () => {
  const plan = planBundleResidency(bundleInput(839, 53_323, 8, 1024 * MiB, 128 * MiB));
  assert.equal(plan.mode, 'streamed');
  assert.ok(8 * (42 * plan.obsCapacity + 13 * plan.pointCapacity + 78 * 839) <= 128 * MiB);
});

test('a smaller budget gives a smaller window and more windows, down to the longest track', () => {
  const sizes = [32, 16, 10].map((mib) =>
    planBundleResidency(bundleInput(100, 20_000, 6, mib * MiB)),
  );
  for (const plan of sizes) assert.equal(plan.mode, 'streamed');
  assert.ok(sizes[0]!.obsCapacity > sizes[1]!.obsCapacity);
  assert.ok(sizes[1]!.obsCapacity > sizes[2]!.obsCapacity);
  const floor = bundleInput(100, 20_000, 6, 0).floorBytes;
  const least = planBundleResidency(bundleInput(100, 20_000, 6, floor + 6 * 336 + 104));
  assert.ok(least.fits);
  assert.equal(least.obsCapacity, 6);
  assert.equal(least.windows.length, 20_000);
});

test('a budget below the floor and one track does not fit; the plan says so', () => {
  const floor = bundleInput(100, 2_000, 6, 0).floorBytes;
  assert.equal(planBundleResidency(bundleInput(100, 2_000, 6, floor)).fits, false);
});

test('the dispatch limit caps the window and the cost windows', () => {
  const plan = planBundleResidency(bundleInput(10, 5_000, 4, 1024 * MiB, 2048 * MiB, 4096));
  assert.equal(plan.mode, 'streamed');
  assert.ok(plan.obsCapacity <= 4096);
  for (const w of plan.costWindows) {
    assert.ok(w.obsEnd - w.obsBase <= 4096);
    assert.equal(w.obsBase % 64, 0);
  }
});

test('triangulation chunks tile the tracks and fit the budget', () => {
  const counts = Array.from({ length: 1000 }, (_, i) => 2 + (i % 5));
  const bytes = (a: number, b: number) => {
    let n = 4;
    for (let i = a; i < b; i++)
      n += TRIANGULATION_OBSERVATION_BYTES * counts[i]! + TRIANGULATION_TRACK_BYTES;
    return n;
  };
  const whole = planTriangulationChunks(counts, {
    budgetBytes: 1e9,
    bindingBytes: 1e9,
    maxKernelElements: 1e6,
  });
  assert.equal(whole.length, 1);
  const budget = 20_000;
  const chunks = planTriangulationChunks(counts, {
    budgetBytes: budget,
    bindingBytes: 1e9,
    maxKernelElements: 1e6,
  });
  assert.ok(chunks.length > 1);
  let next = 0;
  for (const c of chunks) {
    assert.equal(c.first, next);
    assert.ok(c.end > c.first);
    assert.ok(bytes(c.first, c.end) <= budget);
    next = c.end;
  }
  assert.equal(next, 1000);
  // A budget below one track still gives one track a chunk.
  const tiny = planTriangulationChunks(counts, {
    budgetBytes: 1,
    bindingBytes: 1e9,
    maxKernelElements: 1e6,
  });
  assert.equal(tiny.length, 1000);
});
