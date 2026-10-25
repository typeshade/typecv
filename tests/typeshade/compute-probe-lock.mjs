// Probe builds share generated shader host views in src/gpu; only compilation/cleanup serialize.
import { mkdir, open, readFile, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

export async function withComputeProbeLock(root, operation, { timeoutMs = 600000 } = {}) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1)
    throw new RangeError('Probe compile lock timeout must be a positive safe integer');
  const directory = resolve(root, '.typecv-build');
  await mkdir(directory, { recursive: true });
  const path = resolve(directory, 'compute-probe.lock');
  const deadline = Date.now() + timeoutMs;
  let owner;
  while (!owner) {
    try {
      owner = await open(path, 'wx');
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
      let pid;
      try {
        pid = JSON.parse(await readFile(path, 'utf8'))?.pid;
      } catch (readError) {
        if (readError.code !== 'ENOENT' && !(readError instanceof SyntaxError)) throw readError;
      }
      if (Number.isSafeInteger(pid) && pid > 0 && pid <= 0x7fffffff) {
        try {
          process.kill(pid, 0);
        } catch (probeError) {
          if (probeError.code === 'ESRCH')
            throw new Error(
              `Probe compiler lock owner PID ${pid} is no longer running: ${path}. Confirm no compiler is running, then remove this lock and retry.`,
            );
          if (probeError.code !== 'EPERM') throw probeError;
        }
      }
      if (Date.now() >= deadline)
        throw new Error(
          `Timed out waiting for the probe compiler lock: ${path}. Check its owner before removing it; active builds must finish first.`,
        );
      await delay(Math.min(100, Math.max(1, deadline - Date.now())));
    }
  }
  try {
    await owner.writeFile(JSON.stringify({ pid: process.pid }));
    return await operation();
  } finally {
    try {
      await owner.close();
    } finally {
      await unlink(path);
    }
  }
}
