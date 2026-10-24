import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  bindNamespace,
  createdNamespaceId,
  findNamespaceId,
} from '../scripts/ensure-run-records.mjs';

const id = 'a'.repeat(32);

test('the deploy step finds, parses and binds the run-record namespace', () => {
  const list = `⛅️ wrangler 4.151.0\n[\n  { "id": "${id}", "title": "typecv-site-typecv-records" },\n  { "id": "${'b'.repeat(32)}", "title": "other" }\n]\n`;
  assert.equal(findNamespaceId(list), id);
  assert.equal(findNamespaceId('[]'), undefined);
  assert.equal(findNamespaceId('not json'), undefined);
  assert.equal(
    createdNamespaceId(
      `Add the following to your configuration file:\n{ "binding": "RECORDS", "id": "${id}" }`,
    ),
    id,
  );
  const config = readFileSync(new URL('../wrangler.jsonc', import.meta.url), 'utf8');
  const bound = bindNamespace(config, id);
  assert.ok(bound.includes(`"id": "${id}"`));
  assert.ok(!bound.includes('REPLACE_WITH_RECORDS_KV_ID'));
  assert.equal(bindNamespace(bound, 'c'.repeat(32)).includes(`"id": "${'c'.repeat(32)}"`), true);
  const unbound = bindNamespace(config, undefined);
  assert.ok(!unbound.includes('kv_namespaces'));
  assert.ok(unbound.includes('"run_worker_first"'));
});

for (const { name, newline } of [
  { name: 'LF', newline: '\n' },
  { name: 'CRLF', newline: '\r\n' },
]) {
  test(`the deploy step removes only the run-record binding with ${name} line endings`, () => {
    const before = ['{', '  "assets": { "run_worker_first": ["/api/*"] },'].join(newline),
      binding = `  "kv_namespaces": [{ "binding": "RECORDS", "id": "${id}" }],`,
      after = ['  "r2_buckets": [],', '}', ''].join(newline),
      config = [before, binding, after].join(newline);
    assert.equal(bindNamespace(config, undefined), [before, after].join(newline));
  });
}
