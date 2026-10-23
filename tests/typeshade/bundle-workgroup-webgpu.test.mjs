import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { buildComputeProbe } from './compute-probe-build.mjs';

test('BA workgroup overrides preserve camera blocks, tails, fixed rows and continuation', async () => {
  const file = await buildComputeProbe('bundle-camera-probe');
  const { result, errors } = await inBrowser(
    async (hardware) => {
      const { measureCameraWorkgroups } = await import('/probe/camera.js');
      return measureCameraWorkgroups({
        counts: [131],
        samples: 1,
        hardware,
        modes: [{}, { first: 0, last: 0 }, { first: 0, last: 1 }, { active: 0 }],
      });
    },
    process.env.TYPECV_HARDWARE_GPU === '1',
    { timeoutMs: 180000, serve: { '/probe/camera.js': file } },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.validation, []);
  assert.equal(result.rows.length, 4);
  for (const row of result.rows)
    assert.deepEqual(
      row.variants.map((v) => [v.cooperative, v.size]),
      [
        [false, 8],
        [false, 16],
        [false, 32],
        [false, 64],
        [false, 128],
        [true, 32],
        [true, 64],
        [true, 128],
      ],
    );
  for (const row of result.rows)
    for (const variant of row.variants) {
      assert.deepEqual(variant.shape, [variant.size, 1, 1]);
      assert.equal(variant.cached, true);
      assert.deepEqual(variant.callShape, [variant.size, 1, 1]);
      assert.equal(variant.callSubmissions, 1);
      assert.ok(variant.callRelativeError < 2e-10);
      assert.ok(variant.maxRelativeError < 2e-10);
      assert.equal(variant.wallMs.length, 1);
      assert.equal(variant.gpuMs.length, result.timestampQuery ? 1 : 0);
    }
});

test(
  'integer fp64 camera assembly matches its CPU oracle on a native adapter',
  {
    skip: process.env.TYPECV_HARDWARE_GPU !== '1' && process.env.TYPECV_INTEGER_BA_GPU !== '1',
  },
  async () => {
    const file = await buildComputeProbe('bundle-camera-probe');
    const integer = await inBrowser(
      async (hardware) => {
        const { checkIntegerCameraAssembly } = await import('/probe/camera.js');
        return checkIntegerCameraAssembly(hardware);
      },
      process.env.TYPECV_HARDWARE_GPU === '1',
      { timeoutMs: 300000, serve: { '/probe/camera.js': file } },
    );
    assert.deepEqual(integer.errors, []);
    assert.deepEqual(integer.result.validation, []);
    assert.equal(integer.result.rows.length, 4);
    for (const row of integer.result.rows) {
      assert.equal(row.variants.length, 8);
      assert.equal(row.errors.length, 8);
      for (const error of row.errors)
        assert.ok(Number.isFinite(error) && error < 2e-10, JSON.stringify(row));
    }
  },
);
