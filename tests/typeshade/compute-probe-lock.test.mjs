import test from 'node:test';
import assert from 'node:assert/strict';
import { fork, spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, readFile, rm, writeFile, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { withComputeProbeLock } from './compute-probe-lock.mjs';

const helper = new URL('./compute-probe-lock.mjs', import.meta.url).href;
test('different processes serialize fake compilation and release after a thrown operation', async () => {
  const root = await mkdtemp(join(tmpdir(), 'typecv-probe-lock-'));
  const children = [];
  try {
    const fixture = join(root, 'child.mjs');
    await writeFile(
      fixture,
      `
import { withComputeProbeLock } from ${JSON.stringify(helper)};
import { open, unlink } from 'node:fs/promises';
import { once } from 'node:events';
const [root, mode] = process.argv.slice(2);
process.send('ready');
try {
  await withComputeProbeLock(root, async () => {
    const guard = await open(root + '/operation-active', 'wx');
    process.send('entered');
    try { if (mode === 'throw') await once(process, 'message'); }
    finally { await guard.close(); await unlink(root + '/operation-active'); }
    if (mode === 'throw') throw new Error('expected fake compile failure');
  }, {timeoutMs: 5000});
  process.send('done');
} catch (error) {
  if (error.message !== 'expected fake compile failure') throw error;
  process.send('threw');
}
`,
    );
    const start = (mode) => {
      const child = fork(fixture, [root, mode], { stdio: ['ignore', 'ignore', 'inherit', 'ipc'] });
      children.push(child);
      const events = [];
      child.on('message', (message) => events.push(message));
      return { child, events, exited: once(child, 'exit') };
    };
    const first = start('throw');
    while (!first.events.includes('entered')) await once(first.child, 'message');
    const second = start('success');
    while (!second.events.includes('ready')) await once(second.child, 'message');
    await delay(150);
    assert.equal(
      second.events.includes('entered'),
      false,
      'the second compiler waits, without entering fake work',
    );
    first.child.send('release');
    assert.deepEqual(await first.exited, [0, null]);
    assert.deepEqual(await second.exited, [0, null]);
    assert.deepEqual(first.events, ['ready', 'entered', 'threw']);
    assert.deepEqual(second.events, ['ready', 'entered', 'done']);
    await assert.rejects(readFile(join(root, '.typecv-build/compute-probe.lock')), {
      code: 'ENOENT',
    });
  } finally {
    for (const child of children) if (child.exitCode === null) child.kill();
    await rm(root, { recursive: true, force: true });
  }
});

test('dead owners and a waiting deadline fail explicitly without stealing the lock', async () => {
  const root = await mkdtemp(join(tmpdir(), 'typecv-probe-stale-'));
  try {
    const directory = join(root, '.typecv-build');
    await mkdir(directory);
    const file = join(directory, 'compute-probe.lock');
    const departed = spawn(process.execPath, ['-e', '']);
    await once(departed, 'exit');
    const stale = JSON.stringify({ pid: departed.pid });
    await writeFile(file, stale);
    await assert.rejects(
      withComputeProbeLock(root, () => assert.fail('must not steal'), { timeoutMs: 100 }),
      /no longer running.*Confirm no compiler/,
    );
    assert.equal(await readFile(file, 'utf8'), stale);
    const active = JSON.stringify({ pid: process.pid });
    await writeFile(file, active);
    await assert.rejects(
      withComputeProbeLock(root, () => assert.fail('must not enter'), { timeoutMs: 20 }),
      /Timed out waiting.*active builds must finish/,
    );
    assert.equal(await readFile(file, 'utf8'), active);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
