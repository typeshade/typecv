import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';

// No file under dist names the machine that built it (issue #266): the TypeShade plugin at the
// pin wrote each kernel function's fuse record with the module's absolute path, so the bundle
// differed between two checkouts of one commit and the site served the build machine's path.
// `npm run test:surface` runs this file after `npm run build`.

const root = join(import.meta.dirname, '../..');
const dist = join(root, 'dist');

/** Every file under `directory`. */
function filesUnder(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesUnder(path) : [path];
  });
}

/** The build directory, in each spelling a bundle may hold, and the usual absolute roots. */
const forbidden: readonly { readonly name: string; readonly pattern: RegExp }[] = [
  ...new Set([root, root.replace(/\\/g, '/'), JSON.stringify(root).slice(1, -1)]),
].map((spelling) => ({
  name: `the build directory ${spelling}`,
  pattern: new RegExp(spelling.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
}));
const roots: readonly { readonly name: string; readonly pattern: RegExp }[] = [
  { name: 'an absolute path under /home/', pattern: /\/home\/[^/\s"'`]+\// },
  { name: 'an absolute path under /Users/', pattern: /\/Users\/[^/\s"'`]+\// },
  { name: 'an absolute path under /root/ or /tmp/', pattern: /["'`(]\/(root|tmp)\/[^\s"'`]+/ },
  { name: 'an absolute Windows path', pattern: /\b[A-Za-z]:(\\\\|\\|\/)(Users|home|a|work)\b/ },
  { name: 'a file: URL', pattern: /file:\/\/\// },
];

test('dist holds no absolute path of the build machine', () => {
  assert.ok(existsSync(dist), 'dist is missing: run `npm run build` first');
  const found: string[] = [];
  for (const file of filesUnder(dist)) {
    const text = readFileSync(file, 'utf8');
    for (const { name, pattern } of [...forbidden, ...roots]) {
      const match = pattern.exec(text);
      if (match === null) continue;
      const line = text.slice(0, match.index).split('\n').length;
      found.push(`${relative(root, file)}:${line}: ${name}: ${match[0].slice(0, 120)}`);
    }
  }
  assert.deepEqual(found, [], `dist names the build machine:\n${found.join('\n')}`);
});
