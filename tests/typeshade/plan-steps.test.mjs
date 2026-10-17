import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice } from '../../dist/gpu.js';

test('bounded plan steps group dispatches and wait for read hooks before advancing', async () => {
  const events = [],
    trace = [],
    plans = [];
  const device = await ShadeDevice.open({
    backend: 'cpu',
    boundedSubmissions: true,
    maxDispatchesPerSubmission: 2,
    onProgress: (event) => events.push(`${event.kernel}:${event.phase}`),
  });
  let currentPlan;
  device.plan = async (body) => {
    currentPlan = [];
    body();
    plans.push(currentPlan);
    return {
      tier: 'cpu',
      dispatches: currentPlan.length,
      submissions: 1,
      groups: [],
    };
  };
  function* steps() {
    yield [
      'first',
      () => {
        trace.push('first');
        currentPlan.push('first');
      },
    ];
    yield [
      'second',
      () => {
        trace.push('second');
        currentPlan.push('second');
      },
    ];
    yield [
      'read',
      () => {
        trace.push('read-dispatch');
        currentPlan.push('read');
      },
      async () => {
        trace.push('readback');
      },
    ];
    assert.ok(trace.includes('readback'), 'generator advanced before readback completed');
    yield [
      'after-read',
      () => {
        trace.push('after-read');
        currentPlan.push('after-read');
      },
    ];
  }

  try {
    const report = await device.planSteps(steps());
    assert.deepEqual(plans, [['first', 'second'], ['read'], ['after-read']]);
    assert.deepEqual(trace, ['first', 'second', 'read-dispatch', 'readback', 'after-read']);
    assert.deepEqual(events, [
      'first:start',
      'second:start',
      'first:complete',
      'second:complete',
      'read:start',
      'read:complete',
      'after-read:start',
      'after-read:complete',
    ]);
    assert.equal(report.dispatches, 4);
    assert.equal(report.submissions, 3);
  } finally {
    await device.dispose();
  }
});

test('bounded submission cap defaults to one and validates its range', async () => {
  await assert.rejects(
    ShadeDevice.open({ backend: 'cpu', maxDispatchesPerSubmission: 0 }),
    RangeError,
  );
  await assert.rejects(
    ShadeDevice.open({ backend: 'cpu', maxDispatchesPerSubmission: 65 }),
    RangeError,
  );
  const events = [],
    device = await ShadeDevice.open({
      backend: 'cpu',
      boundedSubmissions: true,
      onProgress: (event) => events.push(`${event.kernel}:${event.phase}`),
    });
  let currentPlan;
  device.plan = async (body) => {
    currentPlan = [];
    body();
    return { tier: 'cpu', dispatches: currentPlan.length, submissions: 1, groups: [] };
  };
  try {
    assert.equal(device.maxDispatchesPerSubmission, 1);
    await device.planSteps([
      ['first', () => events.push('run:first')],
      ['second', () => events.push('run:second')],
    ]);
    assert.deepEqual(events, [
      'first:start',
      'run:first',
      'first:complete',
      'second:start',
      'run:second',
      'second:complete',
    ]);
  } finally {
    await device.dispose();
  }
});

test('unbounded solver checkpoints flush only at read hooks and keep unbounded submissions', async () => {
  const trace = [],
    device = await ShadeDevice.open({ backend: 'cpu', solverCheckpoints: true });
  let plans = 0;
  device.plan = async (body) => {
    plans++;
    body();
    return { tier: 'cpu', dispatches: 1, submissions: 1, groups: [] };
  };
  function* steps() {
    yield ['first', () => trace.push('first')];
    yield ['read', () => trace.push('read'), async () => trace.push('after-read')];
    assert.ok(trace.includes('after-read'), 'generator advanced before readback completed');
    yield ['last', () => trace.push('last')];
  }
  try {
    assert.equal(device.boundedSubmissions, false);
    assert.equal(device.solverCheckpoints, true);
    const report = await device.planSteps(steps());
    assert.equal(plans, 2);
    assert.deepEqual(trace, ['first', 'read', 'after-read', 'last']);
    assert.equal(report.submissions, 2);
  } finally {
    await device.dispose();
  }
});
