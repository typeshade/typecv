import test from 'node:test';
import assert from 'node:assert/strict';
import { LAB_MAPPER, LAB_MAPPER_NUMBERS, LAB_WINDOW_SIZE } from '../src/sfm/config/lab-preset.ts';

// The gate of phase 2 of docs/library-structure.md (section 5): the lab preset keeps the values
// that the lab, its run records and the scripts used before it was centralized (section 4, rule 6).
// The order of the keys is part of the check: the run record and the lab transcripts write the
// options in this order. The copy below is the preset of `LAB_OPTIONS` on main at 4022861.
const TODAY = {
  windowSize: 32,
  ransacIterations: 500,
  maxErrorPx: 3,
  minInitialAngleDeg: 1,
  minTriangulationAngleDeg: 0.5,
  bundleIterations: 4,
  finalBundleIterations: 8,
  refineSharedFocal: true,
};

test('the lab preset holds the values and the order of LAB_OPTIONS on 4022861', () => {
  const preset = { windowSize: LAB_WINDOW_SIZE, ...LAB_MAPPER };
  assert.equal(JSON.stringify(preset), JSON.stringify(TODAY));
});

test('the six mapper numbers of the preset are LAB of scripts/measure-focal.mjs on 4022861', () => {
  const { windowSize: _windowSize, refineSharedFocal: _refine, ...numbers } = TODAY;
  assert.equal(JSON.stringify(LAB_MAPPER_NUMBERS), JSON.stringify(numbers));
});
