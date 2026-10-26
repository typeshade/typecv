import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { build } from 'esbuild';
import { mvsReferencePatchCases } from './mvs-reference-patch-cases.mjs';

function emittedFunction(wgsl, name) {
  const start = wgsl.indexOf('fn ' + name + '(');
  assert.ok(start >= 0, `missing emitted ${name}`);
  const open = wgsl.indexOf('{', start);
  let depth = 1,
    end = open + 1;
  for (; end < wgsl.length && depth; end++) {
    if (wgsl[end] === '{') depth++;
    else if (wgsl[end] === '}') depth--;
  }
  assert.equal(depth, 0);
  return wgsl.slice(start, end);
}

test('invocation-local MVS reference patches preserve scalar/f32 costs, candidates and streamed phases', async () => {
  // Bundle the installed compiler's public API in memory for this CPU-only test. The app's
  // bundles and its production compiler/runtime boundary are untouched.
  const bundled = await build({
    stdin: {
      contents: "export { compile, compileModuleJs } from 'typeshade';",
      resolveDir: path.resolve(import.meta.dirname, '../..'),
      sourcefile: 'mvs-reference-patch-compiler.mjs',
    },
    bundle: true,
    platform: 'node',
    format: 'cjs',
    write: false,
    external: ['typescript'],
  });
  const module = { exports: {} };
  new Function('require', 'module', 'exports', bundled.outputFiles[0].text)(
    createRequire(import.meta.url),
    module,
    module.exports,
  );
  const source = await readFile(new URL('../../src/mvs/gpu/mvs.shade.ts', import.meta.url), 'utf8');
  // Inspect the uninstrumented shader. A pointer signature alone does not stop LICM from
  // materializing a whole patch field as a value before its source/plane guards.
  const compiled = module.exports.compile(source);
  assert.deepEqual(
    compiled.diagnostics.filter((d) => d.category === 'error'),
    [],
  );
  for (const name of ['sourceCostCached', 'referencePatchValue', 'referencePatchWeight']) {
    const body = emittedFunction(compiled.wgsl, name);
    assert.doesNotMatch(
      body,
      /\b(?:let|var)\b[^=]*=\s*(?:\(\*\w+\)|\*\w+)\s*(?:\.(?:values|weights))?\s*;/,
      `${name} must not copy a complete patch or 121-element field`,
    );
    assert.doesNotMatch(
      body.split('{')[0],
      /\w+:\s*(?:MvsReferencePatch|array<f32,\s*121>)\s*[,)]/,
      `${name} must not take a patch or field by value`,
    );
  }
  const rows = mvsReferencePatchCases(module.exports, source);
  assert.equal(rows.length, 2);
  for (const row of rows) {
    assert.equal(row.probes, 174);
    assert.ok(row.validSourceCosts > 0);
    assert.ok(row.invalidWindows > 0);
    assert.equal(row.propagationCalls, 84);
    assert.equal(row.invalidPropagationCalls, 32);
    assert.equal(row.invalidFallbackCalls, 16);
    assert.ok(row.coverage.invalidWindowsCulled >= 16);
    assert.equal(row.streamedCalls, 360);
    assert.ok(row.coverage.windowNeighborsEvaluated > 0);
    assert.ok(row.coverage.windowNeighborsAccepted > 0);
    assert.ok(row.coverage.streamedNeighborsEvaluated > 0);
    assert.ok(row.coverage.streamedNeighborsAccepted > 0);
    assert.ok(row.coverage.cachedSourceCalls > 0);
    assert.ok(row.coverage.uncachedSourceCalls > 0);
    assert.equal(row.coverage.invalidStatsCachedCalls, 0);
    assert.equal(row.coverage.geometricCachedCalls, 0);
    assert.equal(row.coverage.bilateralCachedCalls, 0);
    assert.equal(row.coverage.bilateralPropagationCalls, 28);
    assert.ok(row.coverage.photometricUnweightedCachedCalls > 0);
    assert.equal(row.coverage.streamedCachedCalls, 0);
  }
  console.log(JSON.stringify({ referencePatch: rows }));
});
