// docs/library-structure.md, section 5, row 9 (phase 9, step 9a): the hosts import the constants
// that their kernel modules export, instead of copying them (section 3.6). This is the bundle
// check of the row's gate. esbuild bundles each host with every `.shade.ts` module replaced by its
// host face, the module that the TypeShade Vite plugin writes (surface §64). A host that imports a
// name the face does not export fails the bundle. The values are the copies that the hosts held
// before the step: `hough.ts` 16, `1 + 4 * 16` and `1e-5`, `tsdf.ts` `0xffffffff`.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { build, type Plugin } from 'esbuild';

const root = path.resolve(import.meta.dirname, '..');

type HostFace = (source: string, options: { fileName: string }) => { code: string };

/** Bundle `contents` as CommonJS and load it, as tests/typeshade/compiler-probes.test.mjs does. */
async function load(contents: string, resolveDir = root): Promise<Record<string, unknown>> {
  const bundle = await build({
    stdin: { contents, resolveDir },
    bundle: true,
    platform: 'node',
    format: 'cjs',
    write: false,
    external: ['typescript'],
  });
  const m = { exports: {} as Record<string, unknown> };
  new Function('require', 'module', 'exports', bundle.outputFiles[0]!.text)(
    createRequire(import.meta.url),
    m,
    m.exports,
  );
  return m.exports;
}

const hostFace = (async () =>
  (await load("export { hostFace } from 'typeshade/language-service';")).hostFace as HostFace)();

/** The host face of a kernel module, as source text. */
async function faceOf(file: string): Promise<string> {
  return (await hostFace)(await readFile(file, 'utf8'), { fileName: file }).code;
}

/** Load every `.shade.ts` module as its host face. */
const faces: Plugin = {
  name: 'host-faces',
  setup(b) {
    b.onLoad({ filter: /\.shade\.ts$/ }, async (args) => ({
      contents: await faceOf(args.path),
      loader: 'ts',
      resolveDir: path.dirname(args.path),
    }));
  },
};

const CASES = [
  {
    kernel: 'src/analysis/gpu/hough.shade.ts',
    host: 'src/analysis/gpu/hough.ts',
    values: { MAX_SEGMENTS: 16, SEGMENT_STRIDE: 1 + 4 * 16, TIE: 1e-5 },
  },
  {
    kernel: 'src/mvs/gpu/tsdf.shade.ts',
    host: 'src/mvs/gpu/tsdf.ts',
    values: { EMPTY: 0xffffffff },
  },
];

for (const { kernel, host, values } of CASES) {
  test(`${kernel} exports the constants that ${host} used to copy, with their values`, async () => {
    const face = await load(
      await faceOf(path.join(root, kernel)),
      path.join(root, path.dirname(kernel)),
    );
    const names = Object.keys(values) as (keyof typeof values)[];
    assert.deepEqual(Object.fromEntries(names.map((name) => [name, face[name]])), values);
  });

  test(`${host} imports those constants from ${kernel}, and its bundle resolves them`, async () => {
    const source = await readFile(path.join(root, host), 'utf8');
    const kernelName = path.basename(kernel);
    const imported = new RegExp(
      `import \\{([^}]*)\\} from '\\./${kernelName.replace('.', '\\.')}'`,
    );
    const names =
      imported
        .exec(source)?.[1]
        ?.split(',')
        .map((s) => s.trim()) ?? [];
    for (const name of Object.keys(values)) {
      assert.ok(names.includes(name), `${host} does not import ${name} from ./${kernelName}`);
      assert.doesNotMatch(source, new RegExp(`const ${name}\\b`), `${host} still copies ${name}`);
    }
    const result = await build({
      entryPoints: [path.join(root, host)],
      bundle: true,
      platform: 'neutral',
      format: 'esm',
      write: false,
      logLevel: 'silent',
      external: ['typeshade', 'typeshade/*'],
      plugins: [faces],
    });
    assert.deepEqual(result.errors, []);
    assert.deepEqual(result.warnings, []);
  });
}
