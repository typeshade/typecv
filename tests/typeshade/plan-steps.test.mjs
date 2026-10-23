import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice } from '../../dist/gpu.js';

// A real plan consumes the runtime's report. Give it deterministic timestamps so the
// accounting contract can be checked without GPU timing noise or a timestamp-query device.
async function reportedDevice() {
  const device = await ShadeDevice.open({ backend: 'cpu', solverCheckpoints: true });
  const raw = [];
  let calls;
  device.shade.fuse = async (body) => {
    calls = [];
    body();
    const groups = calls.map((name) => ({
      calls: [name],
      fused: false,
      workgroupShapes: [[1, 1, 1]],
      ...(name === 'untimed' ? {} : { gpuTime: 2_000_000 }),
    }));
    const report = { tier: 'cpu', dispatches: groups.length, submissions: 1, groups };
    raw.push(report);
    return report;
  };
  return { device, raw, dispatch: (name) => () => calls.push(name) };
}

for (const many of [false, true]) {
  test(`kernel totals count runtime work once through ${many ? 'parallel' : 'serial'} step reports`, async () => {
    const { device, raw, dispatch } = await reportedDevice();
    const stream = () => [
      ['first', dispatch('first'), async () => {}],
      ['sweep', dispatch('sweep'), undefined, { key: 'sweep', units: 8 }],
      ['untimed', dispatch('untimed')],
    ];
    try {
      await device.plan(dispatch('direct'));
      const report = many
        ? await device.planStepsMany([stream(), stream()])
        : await device.planSteps(stream());
      const repeats = many ? 2 : 1;
      assert.equal(report.dispatches, 3 * repeats);
      assert.equal(device.reports.at(-1), report, 'history keeps the composite report');
      const expected = ['direct', 'first', 'sweep', 'untimed'].map((name) => ({
        name,
        groups: name === 'direct' ? 1 : repeats,
        dispatches: name === 'direct' ? 1 : repeats,
        timedGroups: name === 'untimed' ? 0 : name === 'direct' ? 1 : repeats,
        gpuNs: name === 'untimed' ? 0 : 2_000_000 * (name === 'direct' ? 1 : repeats),
      }));
      assert.deepEqual(device.kernelStatistics.rows, expected);
      assert.deepEqual(device.statistics.kernels, expected, 'run records see the same totals');
      assert.equal(
        expected.reduce((sum, row) => sum + row.dispatches, 0),
        raw.reduce((sum, item) => sum + item.dispatches, 0),
        'the collector counts actual runtime dispatches',
      );
      device.resetKernelStatistics();
      assert.deepEqual(device.kernelStatistics.rows, []);
      await device.plan(dispatch('next-run'));
      assert.deepEqual(device.kernelStatistics.rows, [
        { name: 'next-run', groups: 1, dispatches: 1, timedGroups: 1, gpuNs: 2_000_000 },
      ]);
    } finally {
      await device.dispose();
    }
  });
}

test('nested step reports do not recount work completed inside a read hook', async () => {
  const { device, dispatch } = await reportedDevice();
  try {
    await device.planSteps([
      [
        'outer',
        dispatch('outer'),
        async () => {
          await device.planSteps([['inner', dispatch('inner')]]);
        },
      ],
      ['outer', dispatch('outer')],
    ]);
    assert.deepEqual(device.kernelStatistics.rows, [
      { name: 'outer', groups: 2, dispatches: 2, timedGroups: 2, gpuNs: 4_000_000 },
      { name: 'inner', groups: 1, dispatches: 1, timedGroups: 1, gpuNs: 2_000_000 },
    ]);
  } finally {
    await device.dispose();
  }
});

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

/** A CPU device whose `plan` hands each chunk to the TypeShade device, as `plan` does, and records
 *  each plan's calls and the chunk it was timed under (undefined for an untimed plan). */
async function chunkRecorder(options) {
  const plans = [],
    device = await ShadeDevice.open({ backend: 'cpu', ...options });
  let current;
  device.plan = async (body, chunk) => {
    current = [];
    await device.shade.fuse(body, chunk && { chunk });
    plans.push({ calls: current, chunk });
    return { tier: 'cpu', dispatches: current.length, submissions: 1, groups: [] };
  };
  const dispatch = (name) => () => current.push(name);
  return { device, plans, dispatch };
}

test('a key stops being timed after eight chunks: the later chunks are untimed plans', async () => {
  // Each timed chunk costs two queue fences, a full drain of the GPU each (TypeShade change
  // 0074). Once eight chunks of a key are timed, its rate has settled: the next chunk of the key
  // is still a plan of its own, but it names no chunk, so the device neither fences nor times it.
  for (const value of [0, -1, 1.5, Number.NaN, Number.NEGATIVE_INFINITY, '8'])
    await assert.rejects(
      ShadeDevice.open({ backend: 'cpu', chunkTimingSamples: value }),
      RangeError,
    );
  const { device, plans, dispatch } = await chunkRecorder();
  try {
    assert.equal(device.chunkTimingSamples, 8);
    const steps = (n) =>
      Array.from({ length: n }, (_, i) => [
        'sweep',
        dispatch('sweep'),
        undefined,
        { key: 'sweep', units: 4 + i },
      ]);
    const first = await device.planSteps([['first', dispatch('first')], ...steps(10)]);
    assert.deepEqual(
      plans.map((p) => p.calls),
      [['first'], ...Array.from({ length: 10 }, () => ['sweep'])],
    );
    // The first eight chunks are timed, the ninth and the tenth are not.
    assert.deepEqual(
      plans.map((p) => p.chunk?.units),
      [undefined, 4, 5, 6, 7, 8, 9, 10, 11, undefined, undefined],
    );
    assert.equal(first.submissions, 11);
    assert.equal(device.chunkRates.sweep.chunks, 8);
    assert.equal(device.chunkRates.sweep.lastUnits, 11);
    // The report carries the key's rate, the rate of its timed chunks.
    assert.deepEqual(first.chunks, device.chunkRates);
    // A later call on the device times no chunk of the settled key, and its report still
    // carries the rate.
    plans.length = 0;
    const later = await device.planSteps(steps(2));
    assert.deepEqual(
      plans.map((p) => p.chunk),
      [undefined, undefined],
    );
    assert.equal(device.chunkRates.sweep.chunks, 8);
    assert.deepEqual(later.chunks, device.chunkRates);
    // Another key is timed from its first chunk.
    plans.length = 0;
    await device.planSteps([['score', dispatch('score'), undefined, { key: 'score', units: 2 }]]);
    assert.deepEqual(plans[0].chunk, { key: 'score', units: 2 });
    assert.equal(device.chunkRates.score.chunks, 1);
  } finally {
    await device.dispose();
  }
  // `chunkTimingSamples: Infinity` times every chunk, as before.
  const forced = await chunkRecorder({ chunkTimingSamples: Infinity });
  try {
    await forced.device.planSteps(
      Array.from({ length: 12 }, () => [
        'sweep',
        forced.dispatch('sweep'),
        undefined,
        { key: 'sweep', units: 1 },
      ]),
    );
    assert.ok(forced.plans.every((p) => p.chunk?.key === 'sweep'));
    assert.equal(forced.device.chunkRates.sweep.chunks, 12);
  } finally {
    await forced.device.dispose();
  }
});

test('a settled key is sized by its rate alone, within min and max', async () => {
  // The fourfold cap over the last chunk would hold every later chunk of a settled key to four
  // times its last timed chunk; a settled key takes what fits eight tenths of the budget. The
  // CPU tier keeps running the whole work as one chunk.
  const { device, dispatch } = await chunkRecorder({ timeBudgetMs: 100, chunkTimingSamples: 2 });
  try {
    const chunk = (units) => ['sweep', dispatch('sweep'), undefined, { key: 'sweep', units }];
    await device.planSteps([chunk(1), chunk(1)]);
    assert.equal(device.chunkRates.sweep.chunks, 2);
    assert.equal(device.chunkSize('sweep', 1e9, 1), 1e9);
    // The tier is WebGPU's to the scheduler: the settled rate sizes the chunk.
    device.backend = 'webgpu';
    const fit = Math.floor(80 / device.chunkRates.sweep.msPerUnit);
    assert.equal(device.chunkSize('sweep', 1e9, 1), Math.max(1, fit));
    assert.equal(device.chunkSize('sweep', 3, 1), Math.min(3, Math.max(1, fit)));
    assert.equal(device.chunkSize('sweep', 1e9, fit + 5), fit + 5);
    // A key that has not settled is sized by the TypeShade device: its first chunk is `min`.
    assert.equal(device.chunkSize('other', 1e9, 7), 1e9);
  } finally {
    device.backend = 'cpu';
    await device.dispose();
  }
});

test('a WebGPU device warms a chunk key once, in an untimed plan before its first timed chunk', async () => {
  // The first timed chunk of a key on a fresh device would include the creation of its
  // pipelines (on SwiftShader about 5 s for `triangulateTracks`). The chunk's `warm` dispatches
  // its entries over zero workgroups, as a plan of its own before that chunk, once for each
  // key and program on the device. The CPU tier has no pipelines and runs no warm-up.
  for (const backend of ['webgpu', 'cpu']) {
    const { device, plans, dispatch } = await chunkRecorder();
    device.backend = backend;
    const warmed = [];
    const chunk = (key, id, extra = {}) => [
      key,
      dispatch(key),
      undefined,
      {
        key,
        units: 4,
        warm: {
          id,
          run: () => {
            warmed.push(`${key}/${id}`);
            dispatch(`warm ${key}`)();
          },
        },
        ...extra,
      },
    ];
    try {
      await device.planSteps([
        ['first', dispatch('first')],
        // An untimed chunk needs no warm-up: it is not measured.
        chunk('score', 'pair', { timed: false }),
        chunk('score', 'pair'),
        chunk('score', 'pair'),
        chunk('sweep', 'bundle'),
        ['second', dispatch('second')],
        chunk('sweep', 'bundle'),
        // Another program whose chunks share the key is warmed once too.
        chunk('sweep', 'joint-focal'),
        chunk('sweep', 'joint-focal'),
        // A chunk with no warm-up runs as before.
        ['third', dispatch('third'), undefined, { key: 'plain', units: 1 }],
      ]);
      // A second call on the device warms nothing again.
      await device.planSteps([chunk('score', 'pair'), chunk('sweep', 'bundle')]);
      const expected =
        backend === 'webgpu'
          ? [
              ['first'],
              ['score'],
              ['warm score'],
              ['score'],
              ['score'],
              ['warm sweep'],
              ['sweep'],
              ['second'],
              ['sweep'],
              ['warm sweep'],
              ['sweep'],
              ['sweep'],
              ['third'],
              ['score'],
              ['sweep'],
            ]
          : [
              ['first'],
              ['score'],
              ['score'],
              ['score'],
              ['sweep'],
              ['second'],
              ['sweep'],
              ['sweep'],
              ['sweep'],
              ['third'],
              ['score'],
              ['sweep'],
            ];
      assert.deepEqual(
        plans.map((p) => p.calls),
        expected,
        backend,
      );
      assert.deepEqual(
        warmed,
        backend === 'webgpu' ? ['score/pair', 'sweep/bundle', 'sweep/joint-focal'] : [],
      );
      // A warm-up plan is untimed: the rates hold the timed chunks alone.
      assert.ok(
        plans.every((p) => !p.calls[0].startsWith('warm') || p.chunk === undefined),
        backend,
      );
      assert.equal(device.chunkRates.score.chunks, 3);
      assert.equal(device.chunkRates.sweep.chunks, 5);
    } finally {
      device.backend = 'cpu';
      await device.dispose();
    }
  }
});
