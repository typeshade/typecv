#!/usr/bin/env node
// Export the lab's run records from the collector's KV namespace with the Cloudflare API
// token of the deploy (no read token needed). Run from the repository root with
// CLOUDFLARE_API_TOKEN set; the account id comes from wrangler.jsonc.
//
//   node scripts/export-run-records.mjs [--out lab-records] [--limit 100] [--print failures|all|none]
//
// Writes <out>/index.json (the listing with metadata, newest first) and <out>/<id>.json for
// each record, prints one summary line per record, and prints the JSON of the records
// selected by --print (default: the failed runs).
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export const NAMESPACE_TITLES = ['typecv-site-typecv-records', 'typecv-records'];
export const RECORD_PREFIX = 'record/';
const API = 'https://api.cloudflare.com/client/v4';

/** The account id of wrangler.jsonc. */
export function accountId(config) {
  return /"account_id":\s*"([0-9a-f]{32})"/.exec(config)?.[1];
}
/** The collector's namespace id in the namespaces listing, or undefined. */
export function namespaceId(listing) {
  const entry = (listing?.result ?? []).find(
    (e) => NAMESPACE_TITLES.includes(e.title) && /^[0-9a-f]{32}$/.test(e.id ?? ''),
  );
  return entry?.id;
}
/** The newest `limit` record keys: ids start with the receipt time, so the name orders them. */
export function selectKeys(keys, limit) {
  return keys
    .filter((key) => key.name.startsWith(RECORD_PREFIX))
    .sort((a, b) => (a.name < b.name ? 1 : a.name > b.name ? -1 : 0))
    .slice(0, limit);
}
/** One line per record for the job log. */
export function summaryLine(key) {
  const m = key.metadata ?? {};
  return [
    key.name.slice(RECORD_PREFIX.length),
    m.gpu ?? '?',
    `${m.input ?? '?'}×${m.frames ?? '?'}`,
    `registered ${m.registered ?? '?'}`,
    `${m.totalMs ?? '?'} ms`,
    m.reusedDevice ? 'warm' : 'cold',
    m.fp64Flavor ? `fp64 ${m.fp64Flavor}` : '',
    m.benchmark ? `benchmark ${m.benchmark}` : '',
    m.failure ? `FAILED at ${m.failure}` : '',
  ]
    .filter(Boolean)
    .join(' | ');
}
/** Whether a record is printed in full under `--print`. */
export function printed(record, mode) {
  return mode === 'all' || (mode === 'failures' && record.failure != null);
}

function option(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
async function api(path, token, raw = false) {
  const response = await fetch(API + path, { headers: { authorization: `Bearer ${token}` } });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status} ${await response.text()}`);
  return raw ? response.text() : response.json();
}
async function listKeys(account, namespace, token) {
  const keys = [];
  let cursor = '';
  do {
    const page = await api(
      `/accounts/${account}/storage/kv/namespaces/${namespace}/keys?prefix=${encodeURIComponent(RECORD_PREFIX)}&limit=1000${cursor ? `&cursor=${cursor}` : ''}`,
      token,
    );
    keys.push(...page.result);
    cursor = page.result_info?.cursor ?? '';
  } while (cursor);
  return keys;
}
async function main() {
  const token = process.env.CLOUDFLARE_API_TOKEN;
  if (!token) throw new Error('CLOUDFLARE_API_TOKEN is not set');
  const account = accountId(readFileSync(new URL('../wrangler.jsonc', import.meta.url), 'utf8'));
  if (!account) throw new Error('No account_id in wrangler.jsonc');
  const out = option('--out', 'lab-records'),
    limit = Math.max(1, Number(option('--limit', '100')) || 100),
    mode = option('--print', 'failures');
  const namespace = namespaceId(await api(`/accounts/${account}/storage/kv/namespaces`, token));
  if (!namespace) throw new Error('No run-record namespace in this account');
  const keys = selectKeys(await listKeys(account, namespace, token), limit);
  mkdirSync(out, { recursive: true });
  const index = [];
  for (const key of keys) {
    const text = await api(
      `/accounts/${account}/storage/kv/namespaces/${namespace}/values/${encodeURIComponent(key.name)}`,
      token,
      true,
    );
    const id = key.name.slice(RECORD_PREFIX.length);
    writeFileSync(join(out, `${id}.json`), text);
    index.push({ id, metadata: key.metadata ?? null });
    console.log(summaryLine(key));
    let record;
    try {
      record = JSON.parse(text);
    } catch {
      record = null;
    }
    if (record && printed(record, mode)) {
      console.log(`::group::${id}`);
      console.log(JSON.stringify(record, null, 2));
      console.log('::endgroup::');
    }
  }
  writeFileSync(join(out, 'index.json'), JSON.stringify(index, null, 2));
  console.log(`${keys.length} record(s) written to ${out}/`);
}
if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href)
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
