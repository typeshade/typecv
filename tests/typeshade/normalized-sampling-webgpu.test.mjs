import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdir, rm } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { inBrowser } from './browser.mjs';

test('normalized GPU sampling preserves every f32 bit without a dense normalized image', async () => {
  const root = resolve(import.meta.dirname, '../..');
  const buildDirectory = resolve(root, '.typecv-build');
  const compilerFile = resolve(buildDirectory, `normalized-sampling-compiler-${process.pid}.mjs`);
  const bundleName = `normalized-sampling-test-${process.pid}.js`;
  const browserFile = resolve(root, 'dist', bundleName);
  await mkdir(buildDirectory, { recursive: true });
  await mkdir(resolve(root, 'dist'), { recursive: true });
  try {
    // Compile actual source only in Node. The browser bundle contains generated
    // host wrappers and TypeShade runtime, with no compiler or TypeScript parser.
    await build({
      stdin: {
        contents: "export { hostFace } from 'typeshade/language-service';",
        resolveDir: root,
        sourcefile: 'normalized-sampling-compiler.mjs',
      },
      outfile: compilerFile,
      bundle: true,
      platform: 'node',
      format: 'esm',
      external: ['typescript'],
    });
    const { hostFace } = await import(pathToFileURL(compilerFile).href);
    const output = await build({
      entryPoints: [resolve(import.meta.dirname, 'normalized-sampling-cases.mjs')],
      outfile: browserFile,
      bundle: true,
      platform: 'browser',
      format: 'esm',
      metafile: true,
      plugins: [
        {
          name: 'normalized-sampling-test-shaders',
          setup(builder) {
            builder.onLoad({ filter: /\.shade\.ts$/ }, async ({ path }) => {
              const face = hostFace(await readFile(path, 'utf8'), {
                fileName: path,
                fuse: true,
                readDocument(fileName) {
                  try {
                    return readFileSync(fileName, 'utf8');
                  } catch {
                    return undefined;
                  }
                },
              });
              assert.ok(face.code !== undefined, JSON.stringify(face.diagnostics));
              return { contents: face.code, loader: 'js', resolveDir: dirname(path) };
            });
          },
        },
      ],
    });
    assert.ok(
      !Object.keys(output.metafile.inputs).some(
        (path) => /\/typeshade\/src\/compiler\//.test(path) || /\/typescript\//.test(path),
      ),
    );
    const { result, errors } = await inBrowser(
      async ({ hardware, url }) => {
        const { checkNormalizedSampling } = await import(url);
        return checkNormalizedSampling({ hardware });
      },
      { hardware: process.env.TYPECV_HARDWARE_GPU === '1', url: `/dist/${bundleName}` },
      { timeoutMs: 180000 },
    );
    assert.deepEqual(errors, []);
    assert.deepEqual(result.validationErrors, []);
    assert.equal(result.cases.length, 30);
    assert.equal(result.sampleCases.length, 18);
    assert.equal(result.batchCases.length, 8);
    assert.ok(result.batchCases.every((row) => row.batches === 2 && row.independentBatchValues));
    assert.ok(
      [...result.cases, ...result.sampleCases, ...result.batchCases].every(
        (row) => row.bitwiseEqual && row.allGroupsWebgpu,
      ),
    );
  } finally {
    await rm(browserFile, { force: true });
    await rm(compilerFile, { force: true });
  }
});
