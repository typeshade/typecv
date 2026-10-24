import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { waitForWorkersClosed } from './browser-workers.mjs';
import {
  chooseDecodePath,
  decodeOverride,
  FrameWindow,
  DECODE_WINDOW,
} from './src/lib/decode-protocol.ts';

const full = { worker: true, offscreenCanvas: true, offscreenContext2d: true };

test('the decode path follows feature detection', () => {
  assert.equal(chooseDecodePath(full), 'worker');
  assert.equal(chooseDecodePath({ ...full, offscreenCanvas: false }), 'main-thread');
  assert.equal(chooseDecodePath({ ...full, offscreenContext2d: false }), 'main-thread');
  assert.equal(chooseDecodePath({ ...full, worker: false }), 'main-thread');
});

test('?decode=main forces the page path and ?decode=worker forces nothing', () => {
  assert.equal(decodeOverride('?decode=main'), 'main-thread');
  assert.equal(decodeOverride('?x=1&decode=worker'), 'worker');
  assert.equal(decodeOverride('?decode=other'), undefined);
  assert.equal(chooseDecodePath(full, '?decode=main'), 'main-thread');
  assert.equal(chooseDecodePath({ ...full, worker: false }, '?decode=worker'), 'main-thread');
});

test('the frame window holds the worker to DECODE_WINDOW frames ahead', async () => {
  const window = new FrameWindow(DECODE_WINDOW);
  const log = [];
  const worker = (async () => {
    for (let i = 0; i < 5; i++) {
      await window.take();
      log.push(i);
    }
  })();
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.deepEqual(log, [0, 1]);
  window.release();
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.deepEqual(log, [0, 1, 2]);
  window.release();
  window.release();
  window.release();
  await worker;
  assert.deepEqual(log, [0, 1, 2, 3, 4]);
});

function workerPage() {
  const page = new EventEmitter();
  const workers = new Set();
  page.workers = () => [...workers];
  page.addWorker = (url) => {
    const worker = new EventEmitter();
    worker.url = () => url;
    worker.close = () => {
      workers.delete(worker);
      worker.emit('close');
    };
    workers.add(worker);
    page.emit('worker', worker);
    return worker;
  };
  return page;
}

test('worker cleanup waits for every worker, including arrivals during shutdown', async () => {
  const page = workerPage(),
    numerical = page.addWorker('sfm.worker.js'),
    decoder = page.addWorker('decode.worker.js');
  let finished = false;
  const closed = waitForWorkersClosed(page).then(() => {
    finished = true;
  });
  try {
    numerical.close();
    await Promise.resolve();
    assert.equal(finished, false, 'one worker closing does not finish all-worker cleanup');
    const child = page.addWorker('blob:decode-child');
    decoder.close();
    await Promise.resolve();
    assert.equal(finished, false, 'a worker arriving during shutdown must also close');
    child.close();
    await closed;
    assert.equal(finished, true);
    assert.equal(page.workers().length, 0);
    assert.equal(page.listenerCount('worker'), 0);
    assert.equal(child.listenerCount('close'), 0);
  } finally {
    for (const worker of page.workers()) worker.close();
    await closed;
  }
});

test('worker cleanup reports a retained worker without terminating it', async () => {
  const page = workerPage(),
    retained = page.addWorker('blob:retained-decoder');
  await assert.rejects(
    waitForWorkersClosed(page, 1),
    /Workers did not close: blob:retained-decoder/,
  );
  assert.deepEqual(page.workers(), [retained]);
  assert.equal(page.listenerCount('worker'), 0);
  assert.equal(retained.listenerCount('close'), 0);
  retained.close();
});
