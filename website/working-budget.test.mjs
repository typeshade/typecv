// docs/gpu-budget.md, section 2: a worker's working budget comes from the adapter's limits.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { workingBytesFromAdapter } from './src/lib/working-budget.ts';

const MiB = 1024 ** 2;

test('the working budget is half of the adapter maxBufferSize when that is larger', () => {
  // The adapter of run record d40f8950: maxBufferSize 2 GiB, 640 px frames (64 MiB scale).
  assert.equal(workingBytesFromAdapter(64 * MiB, 2048 * MiB), 1024 * MiB);
  assert.equal(workingBytesFromAdapter(256 * MiB, 2048 * MiB), 1024 * MiB);
});

test('a small adapter, or none, keeps the scale of the analysis frame', () => {
  assert.equal(workingBytesFromAdapter(64 * MiB, 256 * MiB), 128 * MiB);
  assert.equal(workingBytesFromAdapter(64 * MiB, 64 * MiB), 64 * MiB);
  assert.equal(workingBytesFromAdapter(64 * MiB, undefined), 64 * MiB);
});

test('both workers open the device with the adapter limits and the derived budget', () => {
  for (const file of ['src/workers/sfm.worker.ts', 'src/workers/multi-video.ts']) {
    const source = readFileSync(new URL(file, import.meta.url), 'utf8');
    assert.match(source, /workingBytesFromAdapter\(/, file);
    assert.match(source, /adapterLimits: true/, file);
    assert.doesNotMatch(source, /workingBytes: Math\.round\(64 \* 1024 \*\* 2 \* scale\)/, file);
  }
});
