import test from 'node:test';
import assert from 'node:assert/strict';
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
