import test from 'node:test';
import assert from 'node:assert/strict';
import { BundleCgSchedule } from '../src/gpu/bundle-cg-schedule.ts';

test('a completed solve only predicts the next first chunk and never limits continuation', () => {
  const plan = new BundleCgSchedule(true);
  plan.begin();
  assert.equal(plan.next(96, 0), 96);
  plan.observe(21, false);
  plan.begin();
  assert.equal(plan.next(96, 0), 25);
  assert.equal(plan.next(71, 25), 71);
  plan.observe(20, false);
  plan.begin();
  assert.equal(plan.next(8, 0), 8, 'the device time budget remains the tighter cap');
});

test('a harder solve keeps its full remaining budget and cools down the following hint', () => {
  const plan = new BundleCgSchedule(true);
  plan.begin();
  plan.observe(9, false);
  plan.begin();
  assert.equal(plan.next(96, 0), 12);
  plan.observe(0, true);
  assert.equal(plan.next(84, 12), 84);
  plan.observe(42, false);
  plan.begin();
  assert.equal(plan.next(96, 0), 96, 'a miss restores the full device-selected first chunk');
  plan.observe(39, false);
  plan.begin();
  assert.equal(plan.next(96, 0), 48, 'an unhinted success can warm the schedule again');
});

test('zero-iteration convergence is encoded separately from unavailable or failed solves', () => {
  const plan = new BundleCgSchedule(true);
  plan.begin();
  plan.observe(1, false);
  plan.begin();
  assert.equal(plan.next(96, 0), 1);
  plan.observe(0, false);
  plan.begin();
  assert.equal(plan.next(96, 0), 96);
  for (const value of [undefined, -1, 1.5, NaN, Infinity, 258]) {
    plan.observe(value, false);
    plan.begin();
    assert.equal(plan.next(96, 0), 96);
  }
});

test('disabled CPU, bounded and cached policies retain their original chunk selection', () => {
  const plan = new BundleCgSchedule(false);
  plan.begin();
  plan.observe(1, false);
  plan.begin();
  assert.equal(plan.next(96, 0), 96);
  assert.equal(plan.next(7, 25), 7);
});
