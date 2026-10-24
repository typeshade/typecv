import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { cp, mkdir, mkdtemp, readFile, rm, symlink, unlink, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { promisify } from 'node:util';

const exec = promisify(execFile);

test('website preparation preserves the internal runtime static and lazy module tree', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'typecv-runtime-assets-'));
  const esbuildLink = path.join(root, 'node_modules/esbuild');
  let linked = false;
  try {
    const internal = {
      'gpu.js': "export { result } from './chunks/shared.js';\n",
      'chunks/shared.js':
        "import { offset } from './nested/offset.js';\n" +
        'export async function result() {\n' +
        "  const { lazy } = await import('./lazy.js');\n" +
        '  return offset + lazy;\n' +
        '}\n',
      'chunks/nested/offset.js': 'export const offset = 17;\n',
      'chunks/lazy.js': 'export const lazy = 25;\n',
      'chunks/lazy.js.map': '{"version":3,"sources":[],"mappings":""}\n',
    };
    const files: Record<string, string> = {
      'package.json': '{"type":"module"}\n',
      'dist/index.js': 'export const core = true;\n',
      'dist/gpu.js': "export * from './gpu/index.js';\n",
      'dist/gpu/index.js': 'export const gpu = true;\n',
      'website/playground.mjs': 'export const playground = true;\n',
      'src/gpu/vocabulary/rootsift-32-v1.bin': 'fixture vocabulary',
      LICENSE: 'fixture TypeCV license\n',
      'node_modules/typeshade/LICENSE': 'fixture TypeShade license\n',
      'docs/README.md': '# Fixture\n',
      ...Object.fromEntries(
        Object.entries(internal).map(([name, contents]) => [`dist/internal/${name}`, contents]),
      ),
    };
    for (const [name, contents] of Object.entries(files)) {
      const destination = path.join(root, name);
      await mkdir(path.dirname(destination), { recursive: true });
      await writeFile(destination, contents);
    }
    const scripts = path.join(root, 'website/scripts');
    await mkdir(scripts, { recursive: true });
    for (const name of ['prepare.mjs', 'docs.mjs']) {
      await cp(new URL(`../website/scripts/${name}`, import.meta.url), path.join(scripts, name));
    }
    // Borrow only the installed build dependency; all inputs and outputs remain in the fixture.
    const require = createRequire(import.meta.url);
    await symlink(
      path.dirname(require.resolve('esbuild/package.json')),
      esbuildLink,
      process.platform === 'win32' ? 'junction' : 'dir',
    );
    linked = true;
    await exec('git', ['init', '--template='], { cwd: root });
    await exec(
      'git',
      [
        '-c',
        'user.name=Runtime fixture',
        '-c',
        'user.email=fixture@example.invalid',
        '-c',
        'commit.gpgsign=false',
        'commit',
        '--allow-empty',
        '-m',
        'runtime fixture',
      ],
      { cwd: root },
    );
    await exec(process.execPath, [path.join(scripts, 'prepare.mjs')], { cwd: root });

    const published = path.join(root, 'website/public/runtime/internal');
    for (const name of Object.keys(internal)) {
      assert.deepEqual(
        await readFile(path.join(published, name)),
        await readFile(path.join(root, 'dist/internal', name)),
        `internal runtime asset ${name} is copied byte-for-byte`,
      );
    }
    const runtime = await import(pathToFileURL(path.join(published, 'gpu.js')).href);
    assert.equal(await runtime.result(), 42);
  } finally {
    // Remove the dependency link explicitly before deleting the owned temporary project.
    try {
      if (linked) await unlink(esbuildLink);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  }
});
