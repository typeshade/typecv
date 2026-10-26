// Verifies: docs/dense-mvs.md Amendment 3 in section 8: the residency plan sizes every array from
// the budget, takes window mode when it fits, and stays within any budget above the floor.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MVS_PARTIAL_HYPOTHESES,
  mvsPartialLayout,
  planMvsResidency,
  type MvsResidencyInput,
} from '../src/gpu/mvs-residency.ts';

const MiB = 1024 ** 2;
const input = (width: number, height: number, N: number, budget: number, binding: number) =>
  ({
    width,
    height,
    maxSources: Math.min(6, N - 1),
    fusionSlots: Math.min(N, 51),
    seedFloats: 3 * 600,
    bestSources: 3,
    budgetBytes: budget,
    bindingBytes: binding,
  }) satisfies MvsResidencyInput;

test('independent perturbations fit the existing partial storage and preserve every other phase', () => {
  for (let best = 1; best <= 6; best++) {
    const perElement = MVS_PARTIAL_HYPOTHESES * best;
    for (const phase of [0, 1, 2, 3, 4, 5]) {
      const off = mvsPartialLayout(phase, best);
      const on = mvsPartialLayout(phase, best, true);
      assert.ok(off.stride <= perElement);
      assert.ok(on.stride <= perElement);
      if (phase === 3) {
        assert.deepEqual(off, { hypotheses: 1, stride: best });
        assert.deepEqual(on, { hypotheses: 3, stride: 3 * best });
      } else assert.deepEqual(on, off);
      if (phase === 2) assert.deepEqual(on, { hypotheses: 9, stride: 9 * best });
      if (phase === 5) assert.deepEqual(on, { hypotheses: 1, stride: best + 1 });
      // The host rounds a split chunk down to whole workgroups. Even the floor allocation
      // holds one: phase3 needs at most one third of the original phase2 storage per lane.
      const capacity = 64 * perElement;
      const lanes = Math.floor(Math.floor(capacity / on.stride) / 64) * 64;
      assert.ok(lanes >= 64);
      assert.ok(lanes * on.stride <= capacity);
    }
  }
});

test('the owner run of 2026-10-21: 25 views at 1280 × 720 on a 2 GiB adapter take window mode', () => {
  const P = 1280 * 720;
  const plan = planMvsResidency(input(1280, 720, 25, 1024 * MiB, 2048 * MiB));
  assert.equal(plan.photometric.mode, 'window');
  assert.equal(plan.geometric.mode, 'window');
  assert.equal(plan.fusion.mode, 'window');
  // (8 K + 28) bytes per pixel in the geometric pass, with the small arrays.
  const small = 4 * (24 * 7 + 7 + 3);
  assert.equal(plan.geometric.bytes, (8 * 6 + 28) * P + small);
  // (8 F + 36) bytes per pixel in the fusion, the scan's offsets and the small arrays.
  const offsets = 4 * 2 * Math.ceil(P / 64);
  assert.equal(plan.fusion.bytes, (8 * 25 + 36) * P + 4 * (24 * 25 + 25 + 2) + offsets);
  assert.equal(plan.peakBytes, plan.fusion.bytes);
  // The all-resident layout of 4.2 needed (16 N + 4) P = 355 MiB.
  assert.ok(plan.peakBytes < (16 * 25 + 4) * P);
});

test('the depth stage does not grow with N, and the fusion stops at 51 views', () => {
  const sizes = [12, 25, 50, 100].map((N) =>
    planMvsResidency(input(1280, 720, N, 1024 * MiB, 2048 * MiB)),
  );
  for (const plan of sizes) assert.equal(plan.geometric.bytes, sizes[0]!.geometric.bytes);
  assert.ok(sizes[3]!.fusion.bytes - sizes[2]!.fusion.bytes < 8 * 1280 * 720 + 1024);
});

test('the WebGPU defaults stream the fusion at 1280 × 720 and stay within the budget', () => {
  const plan = planMvsResidency(input(1280, 720, 50, 128 * MiB, 128 * MiB));
  assert.equal(plan.geometric.mode, 'window');
  assert.equal(plan.fusion.mode, 'streamed');
  assert.equal(plan.fusion.chunk, 1280 * 720);
  assert.ok(plan.fusion.bytes <= 128 * MiB);
  assert.ok(plan.fusion.largestArray <= 128 * MiB);
});

test('every budget above the floor gives a plan within it, slower as it shrinks', () => {
  const P = 640 * 360;
  let previous = Infinity;
  for (let budget = 64 * MiB; budget >= 36 * P + 8192; budget = Math.floor(budget * 0.8)) {
    const plan = planMvsResidency(input(640, 360, 30, budget, budget));
    for (const stage of [plan.photometric, plan.geometric, plan.fusion]) {
      assert.ok(stage.bytes <= budget, `${budget}: ${stage.bytes}`);
      assert.ok(stage.largestArray <= budget);
    }
    // The geometric pass's chunk of elements shrinks with the budget, never grows.
    const elements = plan.geometric.mode === 'window' ? Infinity : plan.geometric.elements;
    assert.ok(elements <= previous);
    previous = elements;
  }
  assert.ok(previous < Math.ceil(640 / 2) * 360, 'the sweep reached chunked streaming');
});

test('a budget below the floor is refused with the bytes needed and the budget', () => {
  assert.throws(() => planMvsResidency(input(640, 360, 30, 30 * 640 * 360, 1 << 30)), {
    name: 'RangeError',
    message: /needs at least \d+ B for its photometric stage at 640 × 360 .* budget is \d+ B/,
  });
});
