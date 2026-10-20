#!/usr/bin/env node
// Before `wrangler deploy`: make sure the R2 bucket of the lab's video samples exists. Run from
// the repository root with CLOUDFLARE_API_TOKEN set. Without R2 permission on the token the
// SAMPLES binding is removed from wrangler.jsonc for this deploy, and the site deploys without
// the remote samples (their cards show as not published).
//
//   node scripts/ensure-samples-bucket.mjs           # ensure the bucket, or drop the binding
//   node scripts/ensure-samples-bucket.mjs --check   # print the config without the binding
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';

export const BUCKET = 'typecv-samples';
const CONFIG = new URL('../wrangler.jsonc', import.meta.url);

/** Whether `wrangler r2 bucket list` output names the bucket. */
export function listsBucket(listOutput, bucket = BUCKET) {
  return new RegExp(`^name:\\s*${bucket}\\s*$`, 'm').test(listOutput);
}
/** wrangler.jsonc without the R2 binding. */
export function withoutBucket(config) {
  return config.replace(/^\s*"r2_buckets":\s*\[\{[^\]]*\}\],?\n/m, '');
}

function wrangler(args) {
  const result = spawnSync('npx', ['wrangler', ...args], {
    encoding: 'utf8',
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return { ok: result.status === 0, out: `${result.stdout}\n${result.stderr}` };
}

function main() {
  const config = readFileSync(CONFIG, 'utf8');
  if (process.argv.includes('--check')) {
    console.log(withoutBucket(config));
    return;
  }
  const listed = wrangler(['r2', 'bucket', 'list']);
  let ready = listed.ok && listsBucket(listed.out);
  if (listed.ok && !ready) {
    const created = wrangler(['r2', 'bucket', 'create', BUCKET]);
    ready = created.ok;
    if (!ready) console.log(created.out);
  }
  if (!ready) {
    console.log(listed.out);
    console.log(
      `::warning::Could not list or create the R2 bucket ${BUCKET}. The CLOUDFLARE_API_TOKEN ` +
        'needs "Workers R2 Storage: Edit", and R2 must be enabled on the account. Deploying ' +
        'the site without the remote lab samples.',
    );
    writeFileSync(CONFIG, withoutBucket(config));
    return;
  }
  console.log(`SAMPLES bound to the R2 bucket ${BUCKET}`);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
