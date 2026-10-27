import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { compareDigests, readDeclaration } from '../scripts/compare-digests.mjs';

// The rules of the compare job of CI (docs/library-structure.md, section 5, phase 0): a case whose
// digests differ fails, unless the pull request declares it in tests/digests/expected-changes.json;
// what the base could not run is reported and passes.

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

const compare = (base: string, head: string, ...args: string[]) =>
  spawnSync(process.execPath, ['scripts/compare-digests.mjs', base, head, ...args], {
    encoding: 'utf8',
  });

test('a declared difference of this pull request is reported and passes; an undeclared one fails', () => {
  const d = directories(
    { 'a.json': { one: record('x'), two: record('y'), three: record('z') } },
    { 'a.json': { one: record('w'), two: record('v'), three: record('z') } },
  );
  try {
    const file = join(d.root, 'expected-changes.json');
    const declare = (cases: string[]) =>
      writeFileSync(file, JSON.stringify({ pullRequest: 267, reason: 'the focal score', cases }));
    declare(['a.json: one', 'a.json: three', 'a.json: four', 'a.json: one']);
    const declaration = readDeclaration(file);
    assert.deepEqual(declaration, {
      pullRequest: 267,
      reason: 'the focal score',
      cases: ['a.json: one', 'a.json: three', 'a.json: four'],
    });
    const result = compareDigests(d.base, d.head, declaration!.cases);
    assert.deepEqual(result.differences, [
      'a.json: one: output (output parts: reconstruction)',
      'a.json: two: output (output parts: reconstruction)',
    ]);
    assert.deepEqual(result.declared, ['a.json: one: output (output parts: reconstruction)']);
    assert.deepEqual(result.undeclared, ['a.json: two: output (output parts: reconstruction)']);
    assert.deepEqual(result.declaredEqual, ['a.json: three']);
    assert.deepEqual(result.declaredUnknown, ['a.json: four']);
    // The command fails on the undeclared case and reports the declared ones.
    const partial = compare(d.base, d.head, '--declared', file, '--pull-request', '267');
    assert.equal(partial.status, 1);
    assert.match(partial.stdout, /^Cases whose digests differ as declared \(reported\): 1$/m);
    assert.match(partial.stdout, /^Declared cases whose digests are equal \(reported\): 1$/m);
    assert.match(partial.stdout, /^Declared cases that the head did not run \(reported\): 1$/m);
    assert.match(partial.stdout, /^Cases whose digests differ and are not declared: 1$/m);
    // With every difference declared, the command passes.
    declare(['a.json: one', 'a.json: two']);
    const full = compare(d.base, d.head, '--declared', file, '--pull-request', '267');
    assert.equal(full.status, 0, full.stdout + full.stderr);
    assert.match(
      full.stdout,
      /^Declared changes of pull request #267 \(.*expected-changes\.json\): 2 cases; reason: the focal score$/m,
    );
    assert.match(full.stdout, /^Cases whose digests differ and are not declared: 0$/m);
    // A declaration of another pull request is ignored, and the differences fail.
    const other = compare(d.base, d.head, '--declared', file, '--pull-request', '268');
    assert.equal(other.status, 1);
    assert.match(
      other.stdout,
      /^Declaration of pull request #267 in .*expected-changes\.json ignored: this is pull request #268$/m,
    );
    assert.match(other.stdout, /^Cases whose digests differ between the base and the head: 2$/m);
    // Without a declaration the differences fail as before.
    assert.equal(compare(d.base, d.head).status, 1);
  } finally {
    rmSync(d.root, { recursive: true, force: true });
  }
});

test('a malformed declaration is an error, and a missing one is no declaration', () => {
  const d = directories({ 'a.json': { one: record('x') } }, { 'a.json': { one: record('x') } });
  try {
    const file = join(d.root, 'expected-changes.json');
    writeFileSync(file, JSON.stringify({ pullRequest: '267', reason: '', cases: [] }));
    assert.throws(
      () => readDeclaration(file),
      /pullRequest must be a positive integer; reason must be a non-empty string; cases must be a non-empty array/,
    );
    const malformed = compare(d.base, d.head, '--declared', file, '--pull-request', '267');
    assert.equal(malformed.status, 2);
    assert.match(malformed.stderr, /^Malformed declaration: /m);
    writeFileSync(file, JSON.stringify({ pullRequest: 267, reason: 'r', cases: ['one'] }));
    assert.throws(() => readDeclaration(file), /cases must be a non-empty array/);
    writeFileSync(file, '{');
    assert.throws(() => readDeclaration(file), /not JSON/);
    const none = join(d.root, 'none.json');
    assert.equal(readDeclaration(none), undefined);
    assert.equal(compare(d.base, d.head, '--declared', none, '--pull-request', '267').status, 0);
    // The two flags come together, with a positive integer.
    assert.equal(compare(d.base, d.head, '--declared', none).status, 2);
    assert.equal(compare(d.base, d.head, '--pull-request', '267').status, 2);
    assert.equal(compare(d.base, d.head, '--declared', none, '--pull-request', 'x').status, 2);
  } finally {
    rmSync(d.root, { recursive: true, force: true });
  }
});
