// The test files of one shard of a package.json test script. The CI job `webgpu` of
// .github/workflows/ci.yml runs `test:webgpu` and `test:website` on several runners at once, one
// shard each, and the job `webgpu` itself reports the result of every shard (docs/library-structure.md,
// section 5, the common gate).
//
//   node scripts/ci-shard.mjs <script> <shard> <of>
//
// Prints the test files of shard <shard> (1-based) of <of>, one per line. The files are assigned
// round-robin in the order of package.json, so every file of the script runs in exactly one shard
// (tests/ci-shard.test.ts). The flags of the script (`--test-concurrency=1`) are not printed; the
// job passes them itself.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/** The test files of the named script. */
export function scriptFiles(script, packageJson = readPackage()) {
  const command = packageJson.scripts?.[script];
  if (typeof command !== 'string') throw new Error(`package.json has no script ${script}`);
  return command.split(/\s+/).filter((part) => /\.(mjs|ts)$/.test(part));
}

/** The files of shard `shard` of `of`: every `of`-th file of the script, starting at `shard`. */
export function shardFiles(script, shard, of, packageJson = readPackage()) {
  if (!Number.isInteger(shard) || !Number.isInteger(of) || of < 1 || shard < 1 || shard > of)
    throw new RangeError(`shard ${shard} of ${of} is not a shard`);
  return scriptFiles(script, packageJson).filter((_, index) => index % of === shard - 1);
}

function readPackage() {
  const root = join(dirname(fileURLToPath(import.meta.url)), '..');
  return JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [script, shard, of] = process.argv.slice(2);
  process.stdout.write(shardFiles(script, Number(shard), Number(of)).join('\n') + '\n');
}
