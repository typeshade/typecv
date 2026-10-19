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
    '20261019100000000-bbbb | nvidia turing | video×12 | registered 5 | 8123 ms | cold | fp64 integer | FAILED at bundleResidual',
  );
  assert.equal(
    summaryLine(keys[3]!),
    '20261019095959000-cccc | ? | ?×? | registered ? | ? ms | cold | benchmark pyramid-8',
  );
  assert.equal(printed({ failure: { kernel: 'x' } }, 'failures'), true);
  assert.equal(printed({ failure: null }, 'failures'), false);
  assert.equal(printed({ failure: null }, 'all'), true);
  assert.equal(printed({ failure: {} }, 'none'), false);
});
