// Verifies: docs/multi-video.md Amendment 9 (the reason a run has no dense cloud) and
// docs/lab-run-records.md (`dense.skipped` and `dense.failure` carry it).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DENSE_FEW_VIEWS,
  DENSE_NO_COMPONENT,
  DENSE_NO_IMAGE,
  denseReason,
  denseSwitchTitle,
} from './src/lib/dense-reason.ts';
import { denseRecord, runRecordProblem, sumDenseRecords } from './src/lib/run-record.ts';
import { sampleRecord } from './record-fixture.mjs';

const block = (over) => denseRecord({ skipped: '', points: 0, views: [], ...over });

test('the reason of each way a run has no dense cloud', () => {
  assert.equal(denseReason(undefined, false, false), 'Dense MVS was off in the run settings.');
  assert.match(denseReason(undefined, false, true), /꺼져/);
  assert.equal(denseReason(undefined, undefined, false), undefined);
  assert.equal(denseReason(undefined, true, false), undefined);

  const few = block({ skipped: DENSE_FEW_VIEWS });
  assert.equal(denseReason(few, true, false), 'Dense MVS skipped: ' + DENSE_FEW_VIEWS);
  assert.match(denseReason(few, true, true), /건너뜀: .*2개 미만/);
  assert.match(denseReason(block({ skipped: DENSE_NO_COMPONENT }), true, true), /묶음이 없습니다/);
  assert.match(denseReason(block({ skipped: DENSE_NO_IMAGE }), true, true), /이미지/);
  assert.match(denseReason(block({ skipped: DENSE_NO_COMPONENT }), true, false), /No component/);

  assert.equal(denseReason(block({ cancelled: true }), true, false), 'Dense MVS was cancelled.');
  const failed = block({ failure: { message: 'device lost', stage: 'fusion', kernel: 'k' } });
  assert.equal(denseReason(failed, true, false), 'Dense MVS failed: device lost');
  const long = block({ failure: { message: 'x'.repeat(500), stage: '', kernel: '' } });
  assert.ok(denseReason(long, true, false).length < 200);
  // A completed run has no reason to give.
  assert.equal(denseReason(block({ points: 10 }), true, false), undefined);
});

test('the Dense switch tooltip falls back to the generic hint', () => {
  assert.equal(
    denseSwitchTitle(undefined, undefined, false),
    'No dense cloud. Turn on Dense MVS in the run settings and run.',
  );
  assert.equal(denseSwitchTitle(undefined, false, false), 'Dense MVS was off in the run settings.');
  assert.match(
    denseSwitchTitle(block({ skipped: DENSE_NO_COMPONENT }), true, false),
    /No component/,
  );
});

test('a record with the no-component reason stays valid', () => {
  const none = block({ skipped: DENSE_NO_COMPONENT });
  assert.equal(none.status, 'skipped');
  assert.equal(none.skipped, DENSE_NO_COMPONENT);
  const sum = sumDenseRecords([none]);
  assert.equal(sum.skipped, DENSE_NO_COMPONENT);
  assert.equal(runRecordProblem(sampleRecord({ dense: sum })), undefined);
});
