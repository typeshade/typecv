#!/usr/bin/env node
// Before `wrangler deploy`: make sure the private R2 bucket of the result files that the lab can
// send with a run record exists. Without R2 permission on the token the RECORD_FILES binding is
// removed from wrangler.jsonc for this deploy, and the lab disables "Also send the result files".
//
//   node scripts/ensure-record-files-bucket.mjs           # ensure the bucket, or drop the binding
//   node scripts/ensure-record-files-bucket.mjs --check   # print the config without the binding
import { ensureBucket } from './r2-bucket.mjs';

export const BUCKET = 'typecv-records';

if (import.meta.url === `file://${process.argv[1]}`)
  ensureBucket(BUCKET, 'RECORD_FILES', 'Deploying the site without result-file uploads.');
