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

test('a chunk step is a plan of its own, timed, and the report carries its rate', async () => {
  const plans = [],
    chunked = [],
    device = await ShadeDevice.open({ backend: 'cpu', timeBudgetMs: 100 });
  // The stub hands the chunk to the TypeShade device as `plan` does, so the device fences and
  // times the plan and learns the rate of its key (TypeShade change 0074), and counts the
  // dispatches itself.
  let currentPlan;
  device.plan = async (body, chunk) => {
    currentPlan = [];
    const report = await device.shade.fuse(body, chunk && { chunk });
    plans.push(currentPlan);
    if (chunk) chunked.push({ ...chunk, elapsedMs: report.chunk?.elapsedMs });
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
    // Each chunked plan's report named its chunk and the time the device measured.
    assert.deepEqual(
      chunked.map((c) => [c.key, c.units]),
      [
        ['sweep', 8],
        ['sweep', 4],
        ['score', 2],
      ],
    );
    assert.ok(chunked.every((c) => typeof c.elapsedMs === 'number' && c.elapsedMs >= 0));
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

test('an untimed chunk step is a plan of its own that teaches no rate', async () => {
  // A scoring whose workgroup count the GPU wrote (TypeShade change 0075) may dispatch nothing,
  // so it runs alone, bounded as a chunk, but is not timed: `plan` gets no chunk for it, and the
  // TypeShade device learns its key's rate from the timed chunks alone.
  const plans = [],
    chunked = [],
    device = await ShadeDevice.open({ backend: 'cpu' });
  let currentPlan;
  device.plan = async (body, chunk) => {
    currentPlan = [];
    const report = await device.shade.fuse(body, chunk && { chunk });
    plans.push(currentPlan);
    chunked.push(chunk && [chunk.key, chunk.units, typeof report.chunk?.elapsedMs]);
    return { tier: 'cpu', dispatches: currentPlan.length, submissions: 1, groups: [] };
  };
  const dispatch = (name) => () => currentPlan.push(name);
  try {
    const report = await device.planSteps([
      ['first', dispatch('first')],
      ['score', dispatch('score'), undefined, { key: 'score', units: 32 }],
      ['second', dispatch('second')],
      ['score', dispatch('score'), undefined, { key: 'score', units: 128, timed: false }],
      ['third', dispatch('third')],
      ['score', dispatch('score'), undefined, { key: 'score', units: 128, timed: false }],
    ]);
    assert.deepEqual(plans, [['first'], ['score'], ['second'], ['score'], ['third'], ['score']]);
    assert.deepEqual(chunked, [
      undefined,
      ['score', 32, 'number'],
      undefined,
      undefined,
      undefined,
      undefined,
    ]);
    assert.equal(report.submissions, 6);
    assert.equal(report.chunks.score.chunks, 1);
    assert.equal(report.chunks.score.lastUnits, 32);
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

test('a chunk that is a plan stub alone leaves no rate: the TypeShade device learns it', async () => {
  // A test that replaces `plan` without calling through sees no chunk reach the device, so the
  // report carries no rate; the arithmetic itself is the compiler's (TypeShade change 0074,
  // `src/runtime/chunk.test.ts`).
  const device = await ShadeDevice.open({ backend: 'cpu' });
  device.plan = async () => ({ tier: 'cpu', dispatches: 0, submissions: 1, groups: [] });
  try {
    const report = await device.planSteps([
      ['sweep', () => {}, undefined, { key: 'sweep', units: 8 }],
    ]);
    assert.equal(report.submissions, 1);
    assert.equal(report.chunks, undefined);
    assert.deepEqual(device.chunkRates, {});
  } finally {
    await device.dispose();
  }
});
