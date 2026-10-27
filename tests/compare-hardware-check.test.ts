import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  compareHardwareCheck,
  markdown,
  verdict,
  type Baseline,
  type ComparisonRow,
} from '../scripts/compare-hardware-check.mjs';

// The comparison of a hardware-check record (docs/lab-run-records.md, "Hardware check"). The
// fixture tests/fixtures/hardware-check/swiftshader.json is a record of the check's worker on
// SwiftShader (website/hardware-check-browser.test.mjs with TYPECV_CHECK_RECORD_OUT), stored and
// read back by website/worker.mjs, with the site fields of a deployed page. The tests change copies
// of it into the cases that must fail or warn.

const ROOT = join(import.meta.dirname, '..');
const SCRIPT = join(ROOT, 'scripts/compare-hardware-check.mjs');
const FIXTURE = join(ROOT, 'tests/fixtures/hardware-check/swiftshader.json');
const record = JSON.parse(readFileSync(FIXTURE, 'utf8'));
const copy = () => JSON.parse(JSON.stringify(record));
/** A baseline equal to the record's own CPU-tier digests. */
const baselineOf = (r: any): Baseline =>
  Object.fromEntries(
    r.cases.map((c: any) => [c.name, { input: c.digests.input, cpu: c.digests.cpu }]),
  );
const baseline = baselineOf(record);
const find = (rows: ComparisonRow[], check: string) => {
  const row = rows.find((r) => r.check === check);
  assert.ok(row, `no row ${check}`);
  return row;
};
const failing = (rows: ComparisonRow[]) =>
  rows.filter((r) => r.verdict === 'FAIL').map((r) => r.check);

test('the fixture is a complete check: every case within its bound on SwiftShader', () => {
  assert.deepEqual(
    record.cases.map((c: any) => [c.name, c.outcome, c.pass, c.repeatable]),
    [
      ['clahe', 'completed', true, true],
      ['planar-depth', 'completed', true, true],
      ['cloud-clean', 'completed', true, true],
      ['a1-triangulation', 'completed', true, true],
      ['kernel-functions', 'completed', true, true],
    ],
  );
  assert.equal(record.adapter.architecture, 'swiftshader');
});

test('a passing record has no failing or warning row, and the table names every row', () => {
  const rows = compareHardwareCheck(record, { baseline, against: copy() });
  assert.equal(verdict(rows), 0);
  assert.deepEqual(
    rows.filter((r) => r.verdict === 'FAIL' || r.verdict === 'WARN'),
    [],
  );
  for (const check of [
    "flavor, the lab's path",
    'flavor, the default path',
    "flavor, the page's arithmetic",
  ])
    assert.equal(find(rows, check).verdict, 'PASS', check);
  for (const c of record.cases) {
    for (const suffix of ['outcome', 'bound', 'second run', 'input', 'CPU tier'])
      assert.equal(find(rows, `${c.name}: ${suffix}`).verdict, 'PASS', `${c.name}: ${suffix}`);
    assert.equal(find(rows, `${c.name}: against the earlier record`).verdict, 'PASS');
    assert.equal(find(rows, `${c.name}: device bits`).verdict, 'INFO');
  }
  const table = markdown(rows).split('\n');
  assert.equal(table[0], '| Check | Expected | Observed | Verdict |');
  assert.equal(table.length, rows.length + 4);
  assert.match(table.at(-1)!, /^\d+ rows: /);
});

test('a lost case, a changed flavor, an answer out of bound, two answers and a changed answer fail', () => {
  const lost = copy();
  const triangulation = lost.cases.find((c: any) => c.name === 'a1-triangulation');
  Object.assign(triangulation, {
    outcome: 'lost',
    warmMs: null,
    repeatable: null,
    agreement: null,
    pass: null,
    digests: { ...triangulation.digests, device: null, deviceWarm: null, deviceParts: null },
    loss: {
      message: 'D3D12 device removed: DXGI_ERROR_DEVICE_HUNG (0x887A0006)',
      reason: 'unknown',
      lastKernel: 'triangulateTracks',
      uncaptured: [],
    },
  });
  lost.cases.find((c: any) => c.name === 'kernel-functions').outcome = 'skipped';
  lost.cases.find((c: any) => c.name === 'kernel-functions').skipped =
    'Not run after the device loss in a1-triangulation';
  lost.aborted = 'The device was lost or timed out in a1-triangulation';
  lost.resume = '?cases=kernel-functions';
  let rows = compareHardwareCheck(lost, { baseline });
  assert.equal(verdict(rows), 1);
  assert.deepEqual(failing(rows), ['a1-triangulation: outcome', 'check']);
  assert.match(find(rows, 'a1-triangulation: outcome').observed, /DXGI_ERROR_DEVICE_HUNG/);
  assert.equal(find(rows, 'kernel-functions').verdict, 'SKIP');

  const flavor = copy();
  Object.assign(flavor.flavor, { selected: 'integer', same: false });
  rows = compareHardwareCheck(flavor, { baseline });
  assert.deepEqual(failing(rows), ["flavor, the lab's path"]);
  // A page whose `same` disagrees with its own selections fails the arithmetic row.
  flavor.flavor.same = true;
  assert.deepEqual(failing(compareHardwareCheck(flavor, { baseline })), [
    "flavor, the lab's path",
    "flavor, the page's arithmetic",
  ]);

  const bad = copy();
  bad.cases[0].pass = false;
  bad.cases[1].repeatable = false;
  assert.deepEqual(failing(compareHardwareCheck(bad, { baseline })), [
    'clahe: bound',
    'planar-depth: second run',
  ]);

  const changed = copy();
  changed.cases[2].digests.device = '0'.repeat(64);
  assert.deepEqual(failing(compareHardwareCheck(changed, { baseline, against: record })), [
    'cloud-clean: against the earlier record',
  ]);
});

test("another input or another CPU tier warns, and another device's record is not compared", () => {
  const warned = compareHardwareCheck(record, {
    baseline: {
      ...baseline,
      clahe: { input: '1'.repeat(64), cpu: baseline.clahe!.cpu },
      'cloud-clean': { input: baseline['cloud-clean']!.input, cpu: '2'.repeat(64) },
    },
  });
  assert.equal(verdict(warned), 0);
  assert.equal(find(warned, 'clahe: input').verdict, 'WARN');
  assert.equal(find(warned, 'cloud-clean: CPU tier').verdict, 'WARN');
  assert.equal(find(compareHardwareCheck(record, {}), 'clahe: input').verdict, 'n.a.');

  const other = copy();
  other.adapter.architecture = 'blackwell';
  other.cases[0].digests.device = '3'.repeat(64);
  const rows = compareHardwareCheck(record, { baseline, against: other });
  assert.equal(verdict(rows), 0);
  assert.equal(find(rows, 'earlier record').verdict, 'n.a.');
  assert.equal(
    rows.some((r) => r.check.endsWith('against the earlier record')),
    false,
  );
});

test('a benchmark record of the same device reports its flavor, and a failed one fails', () => {
  const run = (overrides: object) => ({
    createdAt: '2026-10-28T01:05:00.000Z',
    server: { id: '20261028010500000-0000aaaa' },
    gpu: record.adapter,
    client: record.client,
    benchmark: { name: 'pyramid-8', pass: 'cold' },
    failure: null,
    timing: { totalMs: 26400, chunks: { triangulateTracks: { longestMs: 5.9 } } },
    result: { registeredViews: 8, landmarks: 653, meanReprojectionErrorPx: 0.581 },
    input: { frames: 8 },
    options: { fp64Flavor: 'float' },
    ...overrides,
  });
  let rows = compareHardwareCheck(record, { baseline, benchmarks: [run({}) as any] });
  assert.equal(verdict(rows), 0);
  assert.equal(find(rows, 'benchmark 20261028010500000-0000aaaa: flavor').verdict, 'PASS');
  assert.match(
    find(rows, 'benchmark 20261028010500000-0000aaaa').observed,
    /8 of 8 views, 653 landmarks/,
  );
  rows = compareHardwareCheck(record, {
    baseline,
    benchmarks: [
      run({
        failure: {
          message: 'WebGPU device lost',
          operation: 'triangulation #11',
          kernel: 'triangulateTracks',
        },
        options: { fp64Flavor: 'integer' },
      }) as any,
    ],
  });
  assert.deepEqual(failing(rows), ['benchmark 20261028010500000-0000aaaa']);
  assert.equal(find(rows, 'benchmark 20261028010500000-0000aaaa: flavor').verdict, 'WARN');
});

test('the command line exits 0 with no failing row, 1 with one, and 2 on a bad input', () => {
  const directory = mkdtempSync(join(tmpdir(), 'typecv-compare-check-'));
  try {
    const run = (...args: string[]) =>
      spawnSync(process.execPath, [SCRIPT, ...args], { cwd: ROOT, encoding: 'utf8' });
    const passed = run(
      '--record',
      FIXTURE,
      '--against',
      FIXTURE,
      '--json',
      join(directory, 'out.json'),
    );
    assert.equal(passed.status, 0, passed.stderr);
    assert.match(
      passed.stdout,
      /\| flavor, the lab's path \| the rule before change 0085: float \| float \| PASS \|/,
    );
    assert.equal(JSON.parse(readFileSync(join(directory, 'out.json'), 'utf8')).verdict, 0);
    const lost = copy();
    lost.cases[0].outcome = 'timeout';
    writeFileSync(join(directory, 'lost.json'), JSON.stringify(lost));
    assert.equal(run('--record', join(directory, 'lost.json')).status, 1);
    assert.equal(run('--record', join(directory, 'missing.json')).status, 2);
    writeFileSync(join(directory, 'run.json'), JSON.stringify({ format: 'typecv-run-record-v6' }));
    const wrong = run('--record', join(directory, 'run.json'));
    assert.equal(wrong.status, 2);
    assert.match(wrong.stderr, /is not a hardware-check record \(field format\)/);
    assert.equal(run().status, 2);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
