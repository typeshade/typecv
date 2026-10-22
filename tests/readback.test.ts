import test from 'node:test';
import assert from 'node:assert/strict';
import { readTogether } from '../src/gpu/readback.ts';

test('a failed read waits for sibling reads before the caller can free their buffers', async () => {
  let finish!: () => void;
  let siblingDone = false;
  const sibling = new Promise<number>((resolve) => {
    finish = () => {
      siblingDone = true;
      resolve(7);
    };
  });
  const reason = new Error('readback failed');
  let returned = false;
  const result = readTogether([Promise.reject(reason), sibling]).catch((error: unknown) => {
    returned = true;
    assert.equal(siblingDone, true);
    assert.equal(error, reason);
  });
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(returned, false);
  finish();
  await result;
});

test('concurrent readbacks preserve input order and the scalar array types', async () => {
  const a = Float32Array.of(1.5),
    b = Uint32Array.of(9);
  assert.deepEqual(await readTogether([Promise.resolve(a), Promise.resolve(b)]), [a, b]);
});
