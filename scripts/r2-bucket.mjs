// Shared by the scripts that ensure an R2 bucket before `wrangler deploy`
// (ensure-samples-bucket.mjs, ensure-record-files-bucket.mjs). Run from the repository root with
// CLOUDFLARE_API_TOKEN set. Without R2 permission on the token the bucket's binding is removed
// from wrangler.jsonc for this deploy.
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';

const CONFIG = new URL('../wrangler.jsonc', import.meta.url);

/** Whether `wrangler r2 bucket list` output names the bucket. */
export function listsBucket(listOutput, bucket) {
  return new RegExp(`^name:\\s*${bucket}\\s*$`, 'm').test(listOutput);
}

/**
 * wrangler.jsonc without the R2 binding of one bucket. Each binding stands on its own line. The
 * `r2_buckets` array goes away when no binding is left.
 */
export function withoutBucket(config, bucket) {
  const lines = config
    .split('\n')
    .filter((line) => !(line.includes('"bucket_name"') && line.includes(`"${bucket}"`)));
  const text = lines.join('\n');
  return text.replace(/^\s*"r2_buckets":\s*\[\s*\],?\n/m, '');
}

function wrangler(args) {
  const result = spawnSync('npx', ['wrangler', ...args], {
    encoding: 'utf8',
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return { ok: result.status === 0, out: `${result.stdout}\n${result.stderr}` };
}

/** Ensure the bucket exists, or drop its binding; `--check` prints the config without it. */
export function ensureBucket(bucket, binding, consequence) {
  const config = readFileSync(CONFIG, 'utf8');
  if (process.argv.includes('--check')) {
    console.log(withoutBucket(config, bucket));
    return;
  }
  const listed = wrangler(['r2', 'bucket', 'list']);
  let ready = listed.ok && listsBucket(listed.out, bucket);
  if (listed.ok && !ready) {
    const created = wrangler(['r2', 'bucket', 'create', bucket]);
    ready = created.ok;
    if (!ready) console.log(created.out);
  }
  if (!ready) {
    console.log(listed.out);
    console.log(
      `::warning::Could not list or create the R2 bucket ${bucket}. The CLOUDFLARE_API_TOKEN ` +
        `needs "Workers R2 Storage: Edit", and R2 must be enabled on the account. ${consequence}`,
    );
    writeFileSync(CONFIG, withoutBucket(config, bucket));
    return;
  }
  console.log(`${binding} bound to the R2 bucket ${bucket}`);
}
