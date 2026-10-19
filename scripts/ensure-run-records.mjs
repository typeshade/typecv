#!/usr/bin/env node
// Before `wrangler deploy`: make sure the run-record collector's KV namespace exists and
// bind it in wrangler.jsonc, and store the read token when the deploy has one. Run from
// the repository root with CLOUDFLARE_API_TOKEN set. Without KV permission on the token
// the binding is removed and the site deploys without the collector (its routes answer 503).
//
//   node scripts/ensure-run-records.mjs            # ensure, bind, store the token if given
//   node scripts/ensure-run-records.mjs --check    # print what would change; no network
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';

export const NAMESPACE = 'typecv-records';
export const PLACEHOLDER = 'REPLACE_WITH_RECORDS_KV_ID';
const CONFIG = new URL('../wrangler.jsonc', import.meta.url);

/** The id of the namespace in `wrangler kv namespace list` output, or undefined. */
export function findNamespaceId(listOutput, worker = 'typecv-site') {
  const start = listOutput.indexOf('[');
  if (start < 0) return undefined;
  let entries;
  try {
    entries = JSON.parse(listOutput.slice(start, listOutput.lastIndexOf(']') + 1));
  } catch {
    return undefined;
  }
  const titles = new Set([NAMESPACE, `${worker}-${NAMESPACE}`]);
  const entry = entries.find((e) => titles.has(e.title) && /^[0-9a-f]{32}$/.test(e.id ?? ''));
  return entry?.id;
}
/** The id `wrangler kv namespace create` printed, or undefined. */
export function createdNamespaceId(createOutput) {
  return /"id":\s*"([0-9a-f]{32})"/.exec(createOutput)?.[1];
}
/** wrangler.jsonc with the RECORDS binding set to `id`, or without the binding when `id` is undefined. */
export function bindNamespace(config, id) {
  const binding = /^\s*"kv_namespaces":\s*\[\{[^\]]*\}\],?\n/m;
  if (id === undefined) return config.replace(binding, '');
  return config.replace(/"id":\s*"(?:[0-9a-f]{32}|REPLACE_WITH_RECORDS_KV_ID)"/, `"id": "${id}"`);
}

function wrangler(args, input) {
  const result = spawnSync('npx', ['wrangler', ...args], {
    encoding: 'utf8',
    input,
    env: process.env,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  return { ok: result.status === 0, out: `${result.stdout}\n${result.stderr}` };
}

function main() {
  const check = process.argv.includes('--check');
  const config = readFileSync(CONFIG, 'utf8');
  if (check) {
    console.log(bindNamespace(config, 'deadbeefdeadbeefdeadbeefdeadbeef'));
    return;
  }
  const listed = wrangler(['kv', 'namespace', 'list']);
  let id = listed.ok ? findNamespaceId(listed.out) : undefined;
  if (listed.ok && !id) {
    const created = wrangler(['kv', 'namespace', 'create', NAMESPACE]);
    id = created.ok ? createdNamespaceId(created.out) : undefined;
    if (!id) console.log(created.out);
  }
  if (!id) {
    console.log(listed.out);
    console.log(
      '::warning::Could not list or create the KV namespace. The CLOUDFLARE_API_TOKEN needs ' +
        '"Workers KV Storage: Edit". Deploying the site without the run-record collector.',
    );
    writeFileSync(CONFIG, bindNamespace(config, undefined));
    return;
  }
  writeFileSync(CONFIG, bindNamespace(config, id));
  console.log(`RECORDS bound to KV namespace ${id}`);
  const token = process.env.RECORDS_READ_TOKEN;
  if (token) {
    const stored = wrangler(['secret', 'put', 'RECORDS_READ_TOKEN'], token + '\n');
    console.log(stored.ok ? 'RECORDS_READ_TOKEN stored' : `::warning::${stored.out}`);
  } else console.log('No RECORDS_READ_TOKEN secret: the read routes stay closed.');
}

if (process.argv[1] && import.meta.url === new URL(process.argv[1], 'file://').href) main();
