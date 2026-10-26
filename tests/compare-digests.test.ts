import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { compareDigests } from '../scripts/compare-digests.mjs';

// The rules of the compare job of CI (docs/library-structure.md, section 5, phase 0): a case whose
// digests differ fails; what the base could not run is reported and passes.

const record = (output: string, extra = {}) => ({
  output,
  parts: { reconstruction: output },
  trace: 't',
  requests: { pnp: 1 },
  ...extra,
});

function directories(base: Record<string, object>, head: Record<string, object>) {
  const root = mkdtempSync(join(tmpdir(), 'typecv-compare-'));
  for (const [side, files] of [
    ['base', base],
    ['head', head],
  ] as const) {
    mkdirSync(join(root, side));
    for (const [file, cases] of Object.entries(files))
      writeFileSync(join(root, side, file), JSON.stringify({ cases }));
  }
  return { root, base: join(root, 'base'), head: join(root, 'head') };
}

test('equal digests pass, and a case or a file that the base lacks is reported', () => {
  const d = directories(
    { 'a.json': { one: record('x') } },
    { 'a.json': { one: record('x'), two: record('y') }, 'b.json': { three: record('z') } },
  );
  try {
    const result = compareDigests(d.base, d.head);
    assert.deepEqual(result.differences, []);
    assert.deepEqual(result.notInBase, ['a.json: two', 'b.json: every case (1)']);
    assert.deepEqual(result.missingInHead, []);
    execFileSync(process.execPath, ['scripts/compare-digests.mjs', d.base, d.head]);
  } finally {
    rmSync(d.root, { recursive: true, force: true });
  }
});

test('a changed digest fails and names the case, the fields and the parts', () => {
  const d = directories(
    { 'a.json': { one: record('x', { plan: { digest: 'p' } }) } },
    { 'a.json': { one: record('w', { plan: { digest: 'q' } }) } },
  );
  try {
    const result = compareDigests(d.base, d.head);
    assert.deepEqual(result.differences, [
      'a.json: one: output, plan (output parts: reconstruction)',
    ]);
    assert.throws(() =>
      execFileSync(process.execPath, ['scripts/compare-digests.mjs', d.base, d.head], {
        stdio: 'pipe',
      }),
    );
  } finally {
    rmSync(d.root, { recursive: true, force: true });
  }
});

test('a file that only the base wrote fails, since the head runs every test', () => {
  const d = directories({ 'a.json': { one: record('x') } }, {});
  try {
    writeFileSync(join(d.head, 'b.json'), JSON.stringify({ cases: {} }));
    assert.deepEqual(compareDigests(d.base, d.head).missingInHead, ['a.json']);
  } finally {
    rmSync(d.root, { recursive: true, force: true });
  }
});

test('digest files that ran on two Node versions fail', () => {
  const d = directories({}, {});
  try {
    const at = (node: string) => ({ configuration: { node }, cases: { one: record('x') } });
    writeFileSync(join(d.base, 'a.json'), JSON.stringify(at('v22.22.0')));
    writeFileSync(join(d.head, 'a.json'), JSON.stringify(at('v24.1.0')));
    assert.deepEqual(compareDigests(d.base, d.head).mismatch, ['v22.22.0', 'v24.1.0']);
    assert.throws(() =>
      execFileSync(process.execPath, ['scripts/compare-digests.mjs', d.base, d.head], {
        stdio: 'pipe',
      }),
    );
  } finally {
    rmSync(d.root, { recursive: true, force: true });
  }
});

test('the configuration lines name the flags of a run where its digest files record them', () => {
  const d = directories({}, {});
  try {
    const at = (execArgv?: string[]) => ({
      configuration: { node: 'v24.21.0', v8: '13.6', platform: 'linux x64', commit: 'c', execArgv },
      cases: { one: record('x') },
    });
    writeFileSync(join(d.base, 'a.json'), JSON.stringify(at()));
    writeFileSync(join(d.head, 'a.json'), JSON.stringify(at(['--single-threaded'])));
    const out = execFileSync(process.execPath, ['scripts/compare-digests.mjs', d.base, d.head], {
      encoding: 'utf8',
    });
    assert.match(
      out,
      /^Head: Node v24\.21\.0 \(V8 13\.6, linux x64; execArgv --single-threaded\) at c$/m,
    );
    assert.match(out, /^Base: Node v24\.21\.0 \(V8 13\.6, linux x64\) at c$/m);
  } finally {
    rmSync(d.root, { recursive: true, force: true });
  }
});
