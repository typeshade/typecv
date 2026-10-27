import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  accountId,
  namespaceId,
  printed,
  recordIds,
  selectKeys,
  summaryLine,
} from '../scripts/export-run-records.mjs';

test('the export finds the account and the namespace, orders the keys and summarizes them', () => {
  assert.equal(
    accountId(readFileSync(new URL('../wrangler.jsonc', import.meta.url), 'utf8')),
    '7c029c2ac777ff3c45a4bcbe67918a1e',
  );
  assert.equal(
    namespaceId({
      result: [
        { id: 'b'.repeat(32), title: 'other' },
        { id: 'a'.repeat(32), title: 'typecv-site-typecv-records' },
      ],
    }),
    'a'.repeat(32),
  );
  assert.equal(namespaceId({ result: [] }), undefined);
  assert.equal(namespaceId(undefined), undefined);
  const keys = [
    { name: 'record/20261019090000000-aaaa', metadata: { gpu: 'nvidia', input: 'sample' } },
    { name: 'ratelimit/x', metadata: null },
    {
      name: 'record/20261019100000000-bbbb',
      metadata: {
        gpu: 'nvidia turing',
        input: 'video',
        frames: 12,
        registered: 5,
        totalMs: 8123,
        reusedDevice: false,
        fp64Flavor: 'integer',
        budgetMs: 250,
        longestChunkMs: 212,
        longestTaskMs: 2851,
        longestGapMs: 2854,
        benchmark: null,
        failure: 'bundleResidual',
      },
    },
    { name: 'record/20261019095959000-cccc', metadata: { benchmark: 'pyramid-8' } },
  ];
  assert.deepEqual(
    selectKeys(keys, 2).map((k) => k.name),
    ['record/20261019100000000-bbbb', 'record/20261019095959000-cccc'],
  );
  assert.equal(
    summaryLine(keys[2]!),
    '20261019100000000-bbbb | nvidia turing | video×12 | registered 5 | 8123 ms | cold | fp64 integer | longest chunk 212/250 ms | main thread task 2851 ms, gap 2854 ms | FAILED at bundleResidual',
  );
  assert.equal(
    summaryLine(keys[3]!),
    '20261019095959000-cccc | ? | ?×? | registered ? | ? ms | cold | benchmark pyramid-8',
  );
  assert.equal(printed({ failure: { kernel: 'x' } }, 'failures'), true);
  assert.equal(printed({ failure: null }, 'failures'), false);
  // v5: a run whose dense MVS failed is printed too, and its line names the dense status.
  assert.equal(printed({ failure: null, dense: { status: 'failed' } }, 'failures'), true);
  assert.equal(printed({ failure: null, dense: { status: 'cancelled' } }, 'failures'), false);
  assert.equal(
    summaryLine({
      name: 'record/20261020120000000-eeee',
      metadata: { gpu: 'apple', denseStatus: 'completed', densePoints: 123456 },
    }),
    '20261020120000000-eeee | apple | ?×? | registered ? | ? ms | cold | dense completed, 123456 points',
  );
  assert.equal(printed({ failure: null }, 'all'), true);
  assert.equal(printed({ failure: {} }, 'none'), false);
  // A probe record of `/lab/probe/`: its own line, printed in full when it found a hanging
  // track or aborted.
  assert.equal(
    summaryLine({
      name: 'record/20261020030000000-dddd',
      metadata: {
        kind: 'probe',
        gpu: 'nvidia blackwell',
        fp64Flavor: 'float',
        boundedSubmissions: false,
        variants: ['completed', 'completed', 'lost'],
        steps: 21,
        hangingTracks: 9,
        hanging: [200, 201, 202, 203, 204, 205, 206, 207],
        aborted: false,
      },
    }),
    '20261020030000000-dddd | nvidia blackwell | probe | fp64 float | unbounded | variants completed/completed/lost | hanging 9: 200, 201, 202, 203, 204, 205, 206, 207, … | 21 steps',
  );
  assert.equal(
    summaryLine({
      name: 'record/20261020030100000-eeee',
      metadata: {
        kind: 'probe',
        gpu: '',
        fp64Flavor: null,
        boundedSubmissions: true,
        variants: ['lost'],
        steps: 2,
        hangingTracks: 0,
        hanging: [],
        aborted: true,
      },
    }),
    '20261020030100000-eeee | ? | probe | bounded | variants lost | hanging none | 2 steps | ABORTED',
  );
  const probe = { format: 'typecv-triangulation-probe-v1', aborted: null, bisection: null };
  assert.equal(printed(probe, 'failures'), false);
  assert.equal(printed({ ...probe, aborted: 'No WebGPU device opened' }, 'failures'), true);
  assert.equal(
    printed({ ...probe, bisection: { hangingTracks: [{ index: 200 }] } }, 'failures'),
    true,
  );
  assert.equal(printed({ ...probe, bisection: { hangingTracks: [] } }, 'failures'), false);
  assert.equal(printed(probe, 'all'), true);
  assert.equal(printed({ ...probe, aborted: 'x' }, 'none'), false);
  // A hardware-check record of `/lab/check/`: its own line, printed in full when a case failed,
  // the flavor changed or the check stopped early.
  assert.equal(
    summaryLine({
      name: 'record/20261028010000000-ffff',
      metadata: {
        kind: 'check',
        gpu: 'blackwell',
        fp64Flavor: 'float',
        sameFlavor: true,
        cases: 5,
        failed: 0,
        failedCases: [],
        aborted: false,
        site: 'db7216a4d0d7',
      },
    }),
    '20261028010000000-ffff | blackwell | check | fp64 float | flavor as before 0085 | 5 cases | failed none | site db7216a4d0d7',
  );
  assert.equal(
    summaryLine({
      name: 'record/20261028010100000-abcd',
      metadata: {
        kind: 'check',
        gpu: '',
        fp64Flavor: 'integer',
        sameFlavor: false,
        cases: 5,
        failed: 2,
        failedCases: ['planar-depth', 'cloud-clean'],
        aborted: true,
        site: null,
      },
    }),
    '20261028010100000-abcd | ? | check | fp64 integer | FLAVOR CHANGED | 5 cases | FAILED 2: planar-depth, cloud-clean | ABORTED',
  );
  const passed = { outcome: 'completed', pass: true, repeatable: true, name: 'clahe' };
  const check = {
    format: 'typecv-hardware-check-v1',
    aborted: null,
    flavor: { same: true },
    cases: [passed, { ...passed, name: 'kernel-functions', outcome: 'skipped', pass: null }],
  };
  assert.equal(printed(check, 'failures'), false);
  assert.equal(printed({ ...check, cases: [{ ...passed, outcome: 'lost' }] }, 'failures'), true);
  assert.equal(printed({ ...check, cases: [{ ...passed, pass: false }] }, 'failures'), true);
  assert.equal(printed({ ...check, cases: [{ ...passed, repeatable: false }] }, 'failures'), true);
  assert.equal(printed({ ...check, flavor: { same: false } }, 'failures'), true);
  assert.equal(printed({ ...check, aborted: 'Device lost in clahe' }, 'failures'), true);
  assert.equal(printed(check, 'all'), true);
  // --record exports the records that a person names by their ids.
  assert.deepEqual(recordIds(''), []);
  assert.deepEqual(recordIds(undefined), []);
  assert.deepEqual(recordIds(' 20261028010000000-0a1b2c3d, 20261028010100000-ffffffff '), [
    '20261028010000000-0a1b2c3d',
    '20261028010100000-ffffffff',
  ]);
  assert.throws(() => recordIds('20261028-abcd'), /Not a record id: 20261028-abcd/);
});
