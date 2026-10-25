import test from 'node:test';
import assert from 'node:assert/strict';
import { TsdfUploadWindow } from '../src/gpu/tsdf-upload-window.ts';

function deferred() {
  let resolve!: () => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<void>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

test('TSDF upload watermark blocks snapshots before exceeding the cap and resumes after drain', async () => {
  const gate = deferred();
  const events: string[] = [];
  const window = new TsdfUploadWindow(64, async () => {
    events.push('drain');
    await gate.promise;
    events.push('completed');
  });
  await window.run(32, () => events.push('first-write'));
  await window.run(32, () => events.push('second-write'));
  assert.equal(window.pendingBytes, 64);
  const next = window.run(16, () => events.push('third-write'));
  assert.deepEqual(events, ['first-write', 'second-write', 'drain']);
  assert.equal(window.pendingBytes, 64, 'blocked admission has not reserved another snapshot');
  assert.deepEqual(window.report, { limitBytes: 64, peakSubmittedBytes: 64, drainCount: 1 });
  await assert.rejects(
    window.run(1, () => events.push('racing-write')),
    /sequential/,
  );
  gate.resolve();
  await next;
  assert.deepEqual(events, ['first-write', 'second-write', 'drain', 'completed', 'third-write']);
  assert.equal(window.pendingBytes, 16);
  assert.equal(window.report.peakSubmittedBytes, 64);
});

test('confirmed TSDF completion clears the watermark without adding redundant fences', async () => {
  let drains = 0;
  const window = new TsdfUploadWindow(48, async () => {
    drains++;
  });
  for (let view = 0; view < 240; view++) {
    await window.run(48, () => {});
    window.completed();
  }
  assert.equal(drains, 0);
  assert.equal(window.pendingBytes, 0);
  assert.deepEqual(window.report, { limitBytes: 48, peakSubmittedBytes: 48, drainCount: 0 });
  const report = window.report as { limitBytes: number };
  report.limitBytes = 1;
  assert.equal(window.report.limitBytes, 48, 'reports do not expose mutable internal state');
});

test('a single oversized view expands the TSDF effective cap without dropping or splitting it', async () => {
  let drains = 0;
  let writes = 0;
  const window = new TsdfUploadWindow(16, async () => {
    drains++;
  });
  await window.run(8, () => {
    writes++;
  });
  await window.run(64, () => {
    writes++;
  });
  assert.equal(writes, 2);
  assert.equal(drains, 1, 'older payload drains before the indivisible large view');
  assert.equal(window.pendingBytes, 64);
  assert.deepEqual(window.report, { limitBytes: 64, peakSubmittedBytes: 64, drainCount: 1 });
  await window.run(0, () => {
    writes++;
  });
  assert.equal(writes, 3);
  assert.equal(drains, 1);
});

test('a failed TSDF drain admits no new writes and retains the prior watermark', async () => {
  const failure = new Error('synthetic GPU fence failure');
  let writes = 0;
  const window = new TsdfUploadWindow(32, async () => {
    throw failure;
  });
  await window.run(32, () => {
    writes++;
  });
  await assert.rejects(
    window.run(16, () => {
      writes++;
    }),
    (error) => error === failure,
  );
  assert.equal(writes, 1);
  assert.equal(window.pendingBytes, 32);
  assert.deepEqual(window.report, { limitBytes: 32, peakSubmittedBytes: 32, drainCount: 1 });
});

test('an upload callback failure retains its complete conservative reservation', async () => {
  const failure = new Error('second channel write failed');
  let drains = 0;
  const window = new TsdfUploadWindow(64, async () => {
    drains++;
  });
  await assert.rejects(
    window.run(48, () => {
      throw failure;
    }),
    (error) => error === failure,
  );
  assert.equal(window.pendingBytes, 48);
  await window.run(32, () => {});
  assert.equal(drains, 1);
  assert.equal(window.pendingBytes, 32);
  assert.equal(window.report.peakSubmittedBytes, 48);
});

test('TSDF admission rejects pre-abort and aborts a blocked fence without a later write', async () => {
  const controller = new AbortController();
  const gate = deferred();
  let writes = 0;
  const window = new TsdfUploadWindow(32, () => gate.promise, controller.signal);
  await window.run(32, () => {
    writes++;
  });
  const waiting = window.run(1, () => {
    writes++;
  });
  controller.abort();
  await assert.rejects(waiting, { name: 'AbortError' });
  assert.equal(writes, 1);
  assert.equal(window.pendingBytes, 32);
  await assert.rejects(
    window.run(1, () => {
      writes++;
    }),
    { name: 'AbortError' },
  );
  gate.reject(new Error('late GPU loss after abort'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(writes, 1, 'late fence settlement cannot invoke the cancelled write');
});

test('an abort raised by the fence callback is observed before its promise settles', async () => {
  const controller = new AbortController();
  const gate = deferred();
  const reason = new Error('cancelled while requesting drain');
  let writes = 0;
  const window = new TsdfUploadWindow(
    16,
    () => {
      controller.abort(reason);
      return gate.promise;
    },
    controller.signal,
  );
  await window.run(16, () => {
    writes++;
  });
  await assert.rejects(
    window.run(16, () => {
      writes++;
    }),
    (error) => error === reason,
  );
  gate.resolve();
  assert.equal(writes, 1);
});

test('TSDF upload admission validates payloads and handles safe-integer boundary arithmetic', async () => {
  for (const limit of [0, -1, NaN, Infinity, 1.5, Number.MAX_SAFE_INTEGER + 1])
    assert.throws(() => new TsdfUploadWindow(limit, async () => {}), RangeError);
  let drains = 0;
  const window = new TsdfUploadWindow(Number.MAX_SAFE_INTEGER, async () => {
    drains++;
  });
  for (const bytes of [-1, NaN, Infinity, 1.5, Number.MAX_SAFE_INTEGER + 1])
    await assert.rejects(
      window.run(bytes, () => {}),
      RangeError,
    );
  await window.run(Number.MAX_SAFE_INTEGER - 1, () => {});
  await window.run(1, () => {});
  assert.equal(drains, 0);
  await window.run(1, () => {});
  assert.equal(drains, 1);
  assert.equal(window.pendingBytes, 1);
  assert.equal(window.report.peakSubmittedBytes, Number.MAX_SAFE_INTEGER);
});
