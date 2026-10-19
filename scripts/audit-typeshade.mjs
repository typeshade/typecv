import assert from 'node:assert/strict';
import { mkdir, readFile, rm } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';
import { inBrowser } from '../tests/typeshade/browser.mjs';

await mkdir('.typecv-build', { recursive: true });
await mkdir('dist', { recursive: true });
await build({
  entryPoints: ['tests/typeshade/audit-compiler.mjs'],
  outfile: '.typecv-build/audit-compiler.mjs',
  bundle: true,
  platform: 'node',
  format: 'esm',
  external: ['typescript'],
});
execFileSync(
  process.execPath,
  ['.typecv-build/audit-compiler.mjs', '.typecv-build/audit-packs.json'],
  { stdio: 'inherit' },
);
const data = JSON.parse(await readFile('.typecv-build/audit-packs.json', 'utf8'));
// The default runs the float flavor appropriate to SwiftShader. The integer
// lowering is also emitted; opt in to its separate device validation explicitly.
const flavors = process.env.TYPECV_AUDIT_INTEGER === '1' ? ['float', 'integer'] : ['float'];
const browserBundle = await build({
  entryPoints: ['tests/typeshade/audit-host.mjs'],
  outfile: 'dist/typeshade-audit.js',
  bundle: true,
  platform: 'browser',
  format: 'esm',
  metafile: true,
});
assert.ok(
  !Object.keys(browserBundle.metafile.inputs).some(
    (path) => /\/typeshade\/src\/compiler\//.test(path) || /\/typescript\//.test(path),
  ),
);
function coefficients(roots) {
  let out = [1];
  for (const root of roots) {
    const next = Array(out.length + 1).fill(0);
    out.forEach((v, i) => {
      next[i] -= root * v;
      next[i + 1] += v;
    });
    out = next;
  }
  return out;
}
const problems = [],
  expected = [];
for (const roots of [
  [1, 2, 3, 4],
  [0.5, 1, 1 + 2 ** -14, 2],
]) {
  const c = coefficients(roots);
  for (let i = 0; i < roots.length; i++) {
    const root = roots[i],
      radius = Math.min(
        0.125,
        i ? (root - roots[i - 1]) / 3 : 1,
        i + 1 < roots.length ? (roots[i + 1] - root) / 3 : 1,
      );
    problems.push(
      Object.fromEntries([
        ...c.map((v, j) => [`a${j}`, v]),
        ['lo', root - radius],
        ['hi', root + radius],
      ]),
    );
    expected.push(root);
  }
}
problems.push({ a0: 1, a1: 0, a2: 0, a3: 0, a4: 1, lo: -1, hi: 1 });
expected.push(null);
try {
  const { result, errors } = await inBrowser(
    async (input) => {
      const { runAudit } = await import('/dist/typeshade-audit.js');
      return runAudit(input);
    },
    { ...data, problems, flavors },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.errors, []);
  assert.equal(result.backend, 'webgpu');
  for (const flavor of flavors) {
    assert.deepEqual(result.results[`precision_${flavor}`].precise, [0.125, 0.125, 0.125]);
    assert.deepEqual(result.results[`precision_${flavor}`].ordinary, [0, 0, 0]);
    const solved = result.results[`quartic_${flavor}`];
    expected.forEach((root, i) => {
      assert.equal(solved.valid[i], root === null ? 0 : 1);
      if (root !== null)
        assert.ok(
          Math.abs(solved.roots[i] - root) < 1e-8,
          `${flavor} root ${i}: ${solved.roots[i]} vs ${root}`,
        );
    });
  }
  for (const group of result.results.workgroups) {
    const expected = Array.from({ length: Math.ceil(513 / group.width) }, (_, b) =>
      Array.from(
        { length: Math.min(group.width, 513 - b * group.width) },
        (_, j) => ((b * group.width + j) % 7) - 3,
      ).reduce((a, b) => a + b, 0),
    );
    assert.deepEqual(group.sums, expected);
    assert.equal(
      group.accepted[0],
      Array.from({ length: 513 }, (_, i) => (i % 7) - 3).filter((v) => v > 0).length,
    );
  }
  const analytic = data.at.map(([x, z, focal]) => (-focal * x) / (z * z));
  result.results.derivative.forEach((v, i) => assert.ok(Math.abs(v - analytic[i]) < 1e-5));
  assert.equal(result.dispatches, 2 * flavors.length + 3);
  console.log(
    JSON.stringify(
      {
        typeshade: '41a140a724f1c5a04de237ec1d5d34d73a910aee',
        mode: process.env.TYPECV_HARDWARE_GPU === '1' ? 'hardware-requested' : 'software-webgpu',
        executedFp64Flavors: flavors,
        ...result,
        expectedRoots: expected,
        analyticDerivative: analytic,
        cpuDerivative: data.cpuDerivative,
        forwardGradCheck: data.forward,
        reverseGradCheck: data.reverse,
        guardBindings: data.guards,
      },
      null,
      2,
    ),
  );
} finally {
  await rm('dist/typeshade-audit.js', { force: true });
}
