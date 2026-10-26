import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { inBrowser } from './browser.mjs';
import { SAMPLES, sampleDeviceCase, sceneDeviceCases } from './sfm-digest-inputs.mjs';

// The device-tier characterization digests of docs/library-structure.md, section 5, phase 0
// (sfm-digest-cases.mjs) on WebGPU, which is SwiftShader in CI. Identity across machines is not
// established (section 4, rule 2; Rule 1.2), so this test records no fixture. It writes the
// digests to `$TYPECV_DIGEST_OUT/webgpu.json`, or else to .typecv-build/digests/webgpu.json, and
// the compare job of CI compares the files of a base and a head made on one runner. It checks
// that every case ran without a page error, a validation error or a thrown error.

const root = join(import.meta.dirname, '../..');
const cases = [...sceneDeviceCases(), ...SAMPLES.map(sampleDeviceCase)];

test('the device-tier characterization digests run on WebGPU and are written to a file', async () => {
  const { result, errors } = await inBrowser(
    async ({ cases }) => {
      const api = await import('/dist/gpu.js');
      const { runDeviceCase } = await import('/tests/typeshade/sfm-digest-cases.mjs');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter');
      const gpu = await adapter.requestDevice(),
        validation = [];
      gpu.addEventListener('uncapturederror', (event) => validation.push(event.error.message));
      const records = {};
      try {
        for (const data of cases) {
          const record = await runDeviceCase(
            api,
            (onProgress) => api.ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0, onProgress }),
            data,
          );
          // A library without an entry point of the case leaves the case out (a merge base).
          if (record !== undefined) records[data.name] = record;
        }
      } finally {
        gpu.destroy();
      }
      const info = adapter.info ?? {};
      return {
        records,
        validation,
        adapter: [info.vendor, info.architecture, info.description].filter(Boolean).join(' '),
      };
    },
    { cases },
    { timeoutMs: 3600000 },
  );
  const out = process.env.TYPECV_DIGEST_OUT ?? join(root, '.typecv-build/digests');
  mkdirSync(out, { recursive: true });
  const names = Object.keys(result.records).sort();
  writeFileSync(
    join(out, 'webgpu.json'),
    `${JSON.stringify(
      {
        adapter: result.adapter,
        cases: Object.fromEntries(names.map((name) => [name, result.records[name]])),
      },
      null,
      2,
    )}\n`,
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.validation, []);
  assert.deepEqual(names, cases.map((data) => data.name).sort());
  for (const name of names) {
    assert.equal(result.records[name].summary.error, undefined, `${name} threw`);
    assert.ok(result.records[name].summary.views >= 2, `${name} registered fewer than two views`);
  }
});
