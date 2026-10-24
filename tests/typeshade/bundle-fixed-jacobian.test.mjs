import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/gpu.js';
import {
  checkFixedJacobianCases,
  assertFixedJacobianCases,
} from './bundle-fixed-jacobian-cases.mjs';

test('fixed camera observations still optimize points and shared focal on the CPU tier', async () => {
  assertFixedJacobianCases(assert, await checkFixedJacobianCases(api, 'cpu'));
});
