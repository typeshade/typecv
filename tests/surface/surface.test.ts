import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { kernelFaces } from '../../scripts/kernel-faces.mjs';
import { ENTRIES, SURFACE_FILE, apiSurface, emitDeclarations } from '../../scripts/api-surface.mjs';
import { DIGEST_FILE, kernelAliases, kernelDigests } from '../../scripts/kernel-digests.mjs';

// The compile-time gates of docs/library-structure.md, section 4, rule 2 (section 5, phase 0):
// the public declaration surface of typecv and typecv/gpu, and the WGSL and GLSL of every kernel
// module. `npm run test:surface` runs this file; the check job of CI runs it after `npm run build`.

const root = join(import.meta.dirname, '../..');
const faces = await kernelFaces(root);
const views = new Map(
  faces.map((face) => [join(root, face.path.replace(/\.ts$/, '.typeshade.ts')), face.view]),
);

/** A unified diff of two texts, by git when it is present, else the first differing line. */
function difference(name: string, recorded: string, current: string): string {
  const directory = join(root, '.typecv-build');
  mkdirSync(directory, { recursive: true });
  const fresh = join(directory, basename(name));
  writeFileSync(fresh, current);
  try {
    execFileSync('git', ['diff', '--no-index', '--no-color', '--', join(root, name), fresh], {
      encoding: 'utf8',
    });
  } catch (error) {
    const out = (error as { stdout?: string }).stdout;
    if (out) {
      const lines = out.split('\n');
      return lines.slice(0, 200).join('\n') + (lines.length > 200 ? '\n…' : '');
    }
  }
  const a = recorded.split('\n'),
    b = current.split('\n');
  const line = a.findIndex((text, i) => text !== b[i]);
  return `first difference at line ${line + 1}:\n- ${a[line]}\n+ ${b[line]}`;
}

test('the public declaration surface of typecv and typecv/gpu is the recorded one', async () => {
  const recorded = readFileSync(join(root, SURFACE_FILE), 'utf8');
  const current = await apiSurface(root, views);
  if (current !== recorded)
    assert.fail(
      `${SURFACE_FILE} differs from the declarations of the source.\n` +
        `A phase that names an addition updates the file: node scripts/api-surface.mjs --write\n\n` +
        difference(SURFACE_FILE, recorded, current),
    );
});

test('the surface reads the declarations that the build emits', (t) => {
  // The build writes dist/ (npm run build); the check job builds before this test.
  if (
    !existsSync(join(root, 'dist/types/gpu/index.d.ts')) ||
    !existsSync(join(root, 'dist/index.d.ts'))
  ) {
    t.skip('no build in dist/: run npm run build first');
    return;
  }
  for (const entry of ENTRIES) {
    const { outputs } = emitDeclarations(root, entry.config, views);
    const differing = [...outputs].filter(([file, text]) => {
      if (!existsSync(file)) return true;
      // tsconfig.build.json maps declarations; the map comment is not a declaration.
      return readFileSync(file, 'utf8').replace(/\n\/\/# sourceMappingURL=.*\n?$/, '\n') !== text;
    });
    assert.deepEqual(
      differing.map(([file]) => file),
      [],
      `${entry.config}: the declarations in memory differ from dist/ (run npm run build)`,
    );
  }
});

test('the WGSL and GLSL of every kernel module are the recorded ones', async () => {
  const recorded = JSON.parse(readFileSync(join(root, DIGEST_FILE), 'utf8'));
  const current = await kernelDigests(root, faces);
  assert.equal(current.compiler, recorded.compiler, 'the TypeShade pin of package.json changed');
  const names = [...new Set([...Object.keys(recorded.modules), ...Object.keys(current.modules)])];
  const changed = names.flatMap((name) => {
    const a = recorded.modules[name],
      b = current.modules[name];
    if (a === undefined) return [`${name}: new module`];
    if (b === undefined) return [`${name}: module removed`];
    const fields = [...new Set([...Object.keys(a), ...Object.keys(b)])].filter(
      (key) => a[key] !== b[key as keyof typeof b],
    );
    return fields.length > 0 ? [`${name}: ${fields.join(', ')}`] : [];
  });
  assert.deepEqual(
    changed,
    [],
    `The emitted shader text differs from ${DIGEST_FILE}. A phase that names a kernel updates the ` +
      'file: node scripts/kernel-digests.mjs --write',
  );
});

test('every kernel alias emits the WGSL and GLSL of its target', async () => {
  // docs/library-structure.md, section 4, rule 4 (amendment 1): an alias keeps an old kernel path
  // working only if it emits its target's shader text.
  const aliases = await kernelAliases(root, faces);
  assert.deepEqual(
    aliases.filter(({ equal }) => !equal).map(({ alias, target }) => `${alias} -> ${target}`),
    [],
    'A kernel alias emits other shader text than its target',
  );
});
