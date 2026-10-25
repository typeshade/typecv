import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { inBrowser } from './browser.mjs';
import {
  nccGpuFixtures,
  legacyNccControl,
  NCC_PROBE,
  NCC_ROW_FLOATS,
} from './mvs-ncc-gpu-cases.mjs';

test('default NCC scalar/cache/aggregate match independent patches on actual f32 WebGPU', async () => {
  const root = resolve(import.meta.dirname, '../..'),
    temporary = await mkdtemp(join(tmpdir(), 'typecv-ncc-'));
  try {
    const bundled = await build({
        stdin: { contents: "export {compile,packModule} from 'typeshade';", resolveDir: root },
        bundle: true,
        platform: 'node',
        format: 'cjs',
        write: false,
        external: ['typescript'],
      }),
      compiler = { exports: {} };
    new Function('require', 'module', 'exports', bundled.outputFiles[0].text)(
      createRequire(import.meta.url),
      compiler,
      compiler.exports,
    );
    const source = await readFile(new URL('../../src/gpu/mvs.shade.ts', import.meta.url), 'utf8'),
      packs = {};
    for (const [label, text] of Object.entries({
      current: source,
      legacy: legacyNccControl(source),
    })) {
      const parsed = compiler.exports.compile(text + NCC_PROBE);
      assert.deepEqual(
        parsed.diagnostics.filter((d) => d.category === 'error'),
        [],
      );
      packs[label] = compiler.exports.packModule(parsed.module);
    }
    await writeFile(join(temporary, 'packs.json'), JSON.stringify(packs));
    await build({
      entryPoints: [fileURLToPath(new URL('./mvs-ncc-gpu-host.mjs', import.meta.url))],
      bundle: true,
      platform: 'browser',
      format: 'esm',
      outfile: join(temporary, 'host.mjs'),
    });
    const fixtures = nccGpuFixtures(),
      hardware = process.env.TYPECV_HARDWARE_GPU === '1',
      { result, errors } = await inBrowser(
        async ({ fixtures, hardware }) => {
          const { runNccGpu } = await import('/private-ncc/host.mjs');
          return runNccGpu({ packsUrl: '/private-ncc/packs.json', fixtures, hardware });
        },
        { fixtures, hardware },
        {
          timeoutMs: 180000,
          serve: {
            '/private-ncc/host.mjs': join(temporary, 'host.mjs'),
            '/private-ncc/packs.json': join(temporary, 'packs.json'),
          },
        },
      );
    assert.deepEqual(errors, []);
    assert.equal(result.backend, 'webgpu');
    assert.deepEqual(result.validation, []);
    assert.deepEqual(result.uncaptured, []);
    assert.deepEqual(result.lost, []);
    assert.equal(result.native.liveBuffers, 0);
    assert.equal(result.native.createdBuffers, result.native.destroyedBuffers);
    assert.ok(result.native.pipelines >= 2);
    assert.equal(result.native.dispatches, fixtures.length * 2);
    assert.equal(result.native.readbacks, fixtures.length * 2);
    let maxError = 0;
    for (const row of result.rows) {
      const fixture = fixtures.find((f) => f.id === row.id),
        values = row.variants.current.values,
        expected = fixture.expected;
      assert.equal(row.variants.current.error, undefined, JSON.stringify(row));
      assert.equal(row.variants.legacy.error, undefined, JSON.stringify(row));
      assert.equal(values.length, NCC_ROW_FLOATS);
      assert.ok(values.every(Number.isFinite), row.id);
      assert.equal(values[4], Number(expected.referenceOk), `${row.id}: reference variance domain`);
      // Reference-invalid inputs deliberately do not call source helpers; their domain is invalid.
      const expectedSlots = [
        [5, expected.aggregate],
        [6, expected.aggregate],
        ...expected.sourceCosts.flatMap((cost, s) => [
          [8 + s, cost],
          [14 + s, cost],
        ]),
      ];
      for (const [slot, oracleCost] of expectedSlots) {
        const cost = expected.referenceOk ? oracleCost : 2;
        const error = Math.abs(values[slot] - cost);
        maxError = Math.max(maxError, error);
        assert.ok(error < 0.001, `${row.id} slot${slot}: GPU${values[slot]} expected${cost}`);
        if (cost === 2) assert.equal(values[slot], 2, `${row.id}: invalid NCC must stay2`);
      }
    }
    const regression = result.rows.find((r) => r.id === 'near-floor-offset'),
      expected = fixtures.find((f) => f.id === regression.id).expected.cost,
      legacyError = Math.abs(regression.variants.legacy.values[5] - expected);
    // The recorded regression is an actual SwiftShader-f32 failure, not a CPU oracle comparison.
    // Physical drivers may fuse the old formula differently, so they need not reproduce the bug.
    if (!hardware) assert.ok(legacyError > 0.001, `legacy counterexample missing: ${legacyError}`);
    console.log(
      JSON.stringify({
        nccWebGPU: {
          fixtures: fixtures.length,
          maxError,
          legacyError,
          adapter: result.description,
          native: result.native,
        },
      }),
    );
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});
