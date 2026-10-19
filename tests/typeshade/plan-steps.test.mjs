import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, fitChunk, learnRate } from '../../dist/gpu.js';

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

test('a chunk step is a plan of its own, timed, and the report carries its rate', async () => {
  const plans = [],
    device = await ShadeDevice.open({ backend: 'cpu', timeBudgetMs: 100 });
  let currentPlan;
  device.plan = async (body) => {
    currentPlan = [];
    body();
    plans.push(currentPlan);
    return { tier: 'cpu', dispatches: currentPlan.length, submissions: 1, groups: [] };
  };
  const dispatch = (name) => () => currentPlan.push(name);
  try {
    assert.equal(device.timeBudgetMs, 100);
    assert.deepEqual(device.chunkRates, {});
    const report = await device.planSteps([
      ['first', dispatch('first')],
      ['second', dispatch('second')],
      ['sweep', dispatch('sweep'), undefined, { key: 'sweep', units: 8 }],
      ['third', dispatch('third')],
      ['sweep', dispatch('sweep'), undefined, { key: 'sweep', units: 4 }],
      ['score', dispatch('score'), undefined, { key: 'score', units: 2 }],
      ['fourth', dispatch('fourth')],
    ]);
    assert.deepEqual(plans, [
      ['first', 'second'],
      ['sweep'],
      ['third'],
      ['sweep'],
      ['score'],
      ['fourth'],
    ]);
    assert.equal(report.submissions, 6);
    assert.deepEqual(Object.keys(report.chunks).sort(), ['score', 'sweep']);
    assert.equal(report.chunks.sweep.chunks, 2);
    assert.equal(report.chunks.sweep.lastUnits, 4);
    assert.equal(report.chunks.score.chunks, 1);
    assert.ok(report.chunks.sweep.msPerUnit > 0);
    assert.ok(report.chunks.sweep.longestMs >= 0);
    assert.deepEqual(device.chunkRates, report.chunks);
    // One plan of steps is one report in the history, however many submissions it took.
    assert.equal(device.reports.length, 1);
    assert.equal(device.reports[0].submissions, 6);
    // The CPU tier runs the whole work as one chunk: its results are those of the state machine
    // run once.
    assert.equal(device.chunkSize('sweep', 160), 160);
    assert.equal(device.chunkSize('unknown', 32, 4), 32);
  } finally {
    await device.dispose();
  }
});

test('the time budget is validated and defaults to 250 ms', async () => {
  for (const timeBudgetMs of [0, -1, 60001, Number.NaN, Number.POSITIVE_INFINITY, '250'])
    await assert.rejects(ShadeDevice.open({ backend: 'cpu', timeBudgetMs }), RangeError);
  const device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    assert.equal(device.timeBudgetMs, 250);
  } finally {
    await device.dispose();
  }
});

test('the chunk scheduler sizes the next chunk from the learned rate within the budget', () => {
  // The first chunk of a key is the minimum, capped at the maximum.
  assert.equal(fitChunk(undefined, 250, 160), 1);
  assert.equal(fitChunk(undefined, 250, 160, 4), 4);
  assert.equal(fitChunk(undefined, 250, 2, 4), 2);
  // A chunk of one unit that took 10 ms: 20 units fit 200 ms, but the next chunk grows at most
  // fourfold.
  const first = learnRate(undefined, { key: 'sweep', units: 1 }, 10);
  assert.deepEqual(first, { msPerUnit: 10, chunks: 1, longestMs: 10, lastUnits: 1 });
  assert.equal(fitChunk(first, 250, 160), 4);
  // Four units in 20 ms: the rate is the mean of 10 and 5 ms per unit, so 26 units fit, and the
  // chunk may grow to 16.
  const second = learnRate(first, { key: 'sweep', units: 4 }, 20);
  assert.deepEqual(second, { msPerUnit: 7.5, chunks: 2, longestMs: 20, lastUnits: 4 });
  assert.equal(fitChunk(second, 250, 160), 16);
  // Sixteen units in 400 ms, past the budget: the rate rises and the next chunk shrinks to what
  // fits.
  const third = learnRate(second, { key: 'sweep', units: 16 }, 400);
  assert.deepEqual(third, { msPerUnit: 16.25, chunks: 3, longestMs: 400, lastUnits: 16 });
  assert.equal(fitChunk(third, 250, 160), 12);
  // The maximum and the minimum bound the fit, and a chunk is at least one unit.
  assert.equal(fitChunk(third, 250, 8), 8);
  assert.equal(fitChunk(third, 250, 160, 20), 20);
  assert.equal(
    fitChunk({ msPerUnit: 1000, chunks: 1, longestMs: 1000, lastUnits: 1 }, 250, 160),
    1,
  );
  // A chunk that measured no time or no units is counted as one microsecond and one unit.
  const instant = learnRate(undefined, { key: 'k', units: 0 }, 0);
  assert.deepEqual(instant, { msPerUnit: 1e-3, chunks: 1, longestMs: 0, lastUnits: 1 });
});
