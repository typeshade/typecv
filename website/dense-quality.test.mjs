import test from 'node:test';
import assert from 'node:assert/strict';
import { denseQualityOptions } from './src/lib/dense-quality.ts';
import { denseRecord, runRecordProblem } from './src/lib/run-record.ts';
import { sampleRecord } from './record-fixture.mjs';

test('dense quality experiments are explicit, independent, bounded and preserve legacy defaults', () => {
  assert.deepEqual(denseQualityOptions(''), {});
  assert.deepEqual(denseQualityOptions('?dense=1&denseBoundary=1'), { boundaryAware: true });
  assert.deepEqual(denseQualityOptions('?denseFiniteGain=0.02'), { minFiniteGain: 0.02 });
  assert.deepEqual(denseQualityOptions('?denseBoundary=0&denseFiniteGain=0'), {
    boundaryAware: false,
    minFiniteGain: 0,
  });
  for (const value of ['-1', 'Infinity', 'NaN', '3', '', ' '])
    assert.deepEqual(
      denseQualityOptions('?denseBoundary=invalid&denseFiniteGain=' + encodeURIComponent(value)),
      {},
    );
});

test('result records retain exact quality settings without inventing enabled defaults', () => {
  const result = (options) =>
    denseRecord({
      cancelled: false,
      failure: null,
      skipped: '',
      points: 1,
      views: [{ id: 0 }],
      options,
    });
  const legacy = result({});
  assert.equal(legacy.options.boundaryAware, undefined);
  assert.equal(legacy.options.minFiniteGain, undefined);
  const explicit = result(denseQualityOptions('?denseBoundary=1&denseFiniteGain=0.02'));
  assert.equal(explicit.options.boundaryAware, true);
  assert.equal(explicit.options.minFiniteGain, 0.02);
  assert.equal(runRecordProblem(sampleRecord({ dense: explicit })), undefined);
  for (const value of [-1, Infinity, NaN, 3]) {
    const invalid = structuredClone(explicit);
    invalid.options.minFiniteGain = value;
    assert.equal(runRecordProblem(sampleRecord({ dense: invalid })), 'dense');
  }
  const disabled = result({ boundaryAware: false, minFiniteGain: 0 });
  assert.equal(disabled.options.boundaryAware, false);
  assert.equal(disabled.options.minFiniteGain, 0);
  const invalid = result({ boundaryAware: 'yes', minFiniteGain: Infinity });
  assert.equal(invalid.options.boundaryAware, undefined);
  assert.equal(invalid.options.minFiniteGain, undefined);
});
