import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { scriptFiles, shardFiles } from '../scripts/ci-shard.mjs';

// The shards of the CI job `webgpu` (.github/workflows/ci.yml): every test file of `test:webgpu`
// and of `test:website` runs in exactly one shard. The shard counts here are the counts of the
// job's matrix.
const SHARDS = { 'test:webgpu': 3, 'test:website': 2 } as const;

const packageJson = JSON.parse(readFileSync(join(import.meta.dirname, '../package.json'), 'utf8'));

for (const [script, of] of Object.entries(SHARDS))
  test(`the ${of} shards of ${script} hold each of its files once`, () => {
    const all = scriptFiles(script, packageJson);
    assert.ok(all.length > 0);
    const shards = Array.from({ length: of }, (_, i) => shardFiles(script, i + 1, of, packageJson));
    const union = shards.flat();
    assert.deepEqual([...union].sort(), [...all].sort());
    assert.equal(union.length, all.length);
    for (const shard of shards) assert.ok(shard.length >= Math.floor(all.length / of));
  });

test('a shard outside 1 to of, and an unknown script, are refused', () => {
  assert.throws(() => shardFiles('test:webgpu', 0, 3, packageJson), RangeError);
  assert.throws(() => shardFiles('test:webgpu', 4, 3, packageJson), RangeError);
  assert.throws(() => shardFiles('no-such-script', 1, 1, packageJson), /no script/);
});
