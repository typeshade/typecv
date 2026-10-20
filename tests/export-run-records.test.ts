import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  accountId,
  namespaceId,
  printed,
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
    '20261019100000000-bbbb | nvidia turing | video×12 | registered 5 | 8123 ms | cold | fp64 integer | longest chunk 212/250 ms | FAILED at bundleResidual',
  );
  assert.equal(
    summaryLine(keys[3]!),
    '20261019095959000-cccc | ? | ?×? | registered ? | ? ms | cold | benchmark pyramid-8',
  );
  assert.equal(printed({ failure: { kernel: 'x' } }, 'failures'), true);
  assert.equal(printed({ failure: null }, 'failures'), false);
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
});
