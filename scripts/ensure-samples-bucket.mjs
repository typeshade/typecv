#!/usr/bin/env node
// Before `wrangler deploy`: make sure the R2 bucket of the lab's video samples exists. Without R2
// permission on the token the SAMPLES binding is removed from wrangler.jsonc for this deploy, and
// the site deploys without the remote samples (their cards show as not published).
//
//   node scripts/ensure-samples-bucket.mjs           # ensure the bucket, or drop the binding
//   node scripts/ensure-samples-bucket.mjs --check   # print the config without the binding
import {
  ensureBucket,
  listsBucket as listsNamed,
  withoutBucket as withoutNamed,
} from './r2-bucket.mjs';

export const BUCKET = 'typecv-samples';

export const listsBucket = (listOutput, bucket = BUCKET) => listsNamed(listOutput, bucket);
export const withoutBucket = (config, bucket = BUCKET) => withoutNamed(config, bucket);

if (import.meta.url === `file://${process.argv[1]}`)
  ensureBucket(BUCKET, 'SAMPLES', 'Deploying the site without the remote lab samples.');
