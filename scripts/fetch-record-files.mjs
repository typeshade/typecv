#!/usr/bin/env node
// Download the result files that a lab run record declares from the private R2 bucket of
// wrangler.jsonc (binding RECORD_FILES), where website/worker.mjs stores each one under the key
// records/<id>/<name>, and check each file against the size and SHA-256 that the record declares.
//
//   node scripts/fetch-record-files.mjs --record lab-records/<id>.json --out files-<id>
//   node scripts/fetch-record-files.mjs --record lab-records/<id>.json --out files-<id> \
//     --only typecv-reconstruction,typecv-dense
//   node scripts/fetch-record-files.mjs --record lab-records/<id>.json --out files-<id> --verify
//
// --record file.json  a record that scripts/export-run-records.mjs wrote: its `files` array
//                     declares each result file's name, bytes and sha256, and `server.id` is the
//                     record's id.
// --out dir           the directory of the files, as <dir>/<name>. Made when it does not exist.
// --only a,b,...      only the declared files whose name starts with one of these prefixes.
// --verify            download nothing: check the files already in --out and report each one.
// --timeout minutes   the longest that one download may take (default 10). A download that takes
//                     longer is stopped and reported as failed.
//
// A file that is already in --out with the declared size and SHA-256 is not downloaded again.
// A download goes to a temporary file in --out and gets its name only when its size and SHA-256
// match the record. A download that does not match is deleted and reported. Parts
// (`<name>.part-NNNN`) download as they are; scripts/reassemble-result-files.mjs joins them. A file
// that the page skipped (`skipped`) is listed and not downloaded.
//
// Each download runs `wrangler r2 object get <bucket>/<key> --file <path> --remote` with the
// repository's wrangler (node_modules/wrangler, the one that `npx wrangler` runs), without a shell
// and from the repository root, so that wrangler reads the account of wrangler.jsonc. The
// environment passes through to wrangler: CLOUDFLARE_API_TOKEN must be a token with R2 read access
// on that account, and behind an HTTP proxy NODE_USE_ENV_PROXY=1 makes wrangler's fetch use
// HTTPS_PROXY. The exit code is 1 when a selected file is missing, does not match or does not
// download, and when the selection holds no uploaded file (a typo in --only), so that a
// verification never passes without checking a file.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, rename, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { resultFileNameValid } from '../website/src/lib/run-record.ts';

/** The key prefix of the stored result files (`FILE_PREFIX` of website/worker.mjs). */
export const FILE_PREFIX = 'records/';
/** A record id: the receipt time and eight hex digits (`RECORD_ID` of website/worker.mjs). */
const RECORD_ID = /^[0-9]{17}-[0-9a-f]{8}$/;
const ROOT = fileURLToPath(new URL('..', import.meta.url));
const WRANGLER = path.join(ROOT, 'node_modules', 'wrangler', 'bin', 'wrangler.js');
/** The statuses of a selected file that count as a success. */
const GOOD = new Set(['ok', 'present', 'downloaded']);
/** The longest that one download may take, in minutes, when --timeout does not say. */
export const DOWNLOAD_TIMEOUT_MINUTES = 10;

/** The bucket of the RECORD_FILES binding in the text of wrangler.jsonc, or undefined. */
export function recordFilesBucket(config) {
  const line = config.split('\n').find((text) => /"binding":\s*"RECORD_FILES"/.test(text));
  return line ? /"bucket_name":\s*"([^"]+)"/.exec(line)?.[1] : undefined;
}

/**
 * The record's id and the declared files to fetch, in the record's order: each uploaded file
 * whose name starts with one of `only` (every file when `only` is empty), with its R2 key, and
 * the selected files that the page skipped. Throws on a manifest that the collector would not
 * have stored: an invalid or repeated name, or an uploaded file without a size or a SHA-256.
 */
export function plannedFiles(record, { only = [] } = {}) {
  const id = record?.server?.id;
  if (typeof id !== 'string' || !RECORD_ID.test(id))
    throw new Error('The record has no valid server.id: use a record of export-run-records.mjs');
  if (!Array.isArray(record.files)) throw new Error('The record declares no result files');
  const names = new Set(),
    files = [],
    skipped = [];
  for (const file of record.files) {
    // A valid name has no path separator, so <out>/<name> stays in --out.
    if (!resultFileNameValid(file?.name))
      throw new Error('Invalid result file name: ' + String(file?.name));
    if (names.has(file.name)) throw new Error('Repeated result file name: ' + file.name);
    names.add(file.name);
    if (only.length && !only.some((prefix) => file.name.startsWith(prefix))) continue;
    if (file.skipped !== undefined) {
      skipped.push({ name: file.name, reason: String(file.skipped) });
      continue;
    }
    if (
      !Number.isSafeInteger(file.bytes) ||
      file.bytes < 0 ||
      typeof file.sha256 !== 'string' ||
      !/^[0-9a-f]{64}$/.test(file.sha256)
    )
      throw new Error('Missing or invalid size or SHA-256: ' + file.name);
    files.push({
      name: file.name,
      bytes: file.bytes,
      sha256: file.sha256,
      key: `${FILE_PREFIX}${id}/${file.name}`,
    });
  }
  return { id, files, skipped };
}

/**
 * Whether the file at `filePath` has the declared size and SHA-256: status `ok`, `missing` or
 * `mismatch` (with the reason), and the size and the digest that were read.
 */
export async function verifyFile(filePath, declared) {
  let info;
  try {
    info = await stat(filePath);
  } catch (error) {
    if (error.code === 'ENOENT') return { status: 'missing' };
    throw error;
  }
  if (!info.isFile()) return { status: 'mismatch', reason: 'not a regular file' };
  if (info.size !== declared.bytes)
    return {
      status: 'mismatch',
      bytes: info.size,
      reason: `${info.size} bytes, the record declares ${declared.bytes}`,
    };
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filePath)) hash.update(chunk);
  const sha256 = hash.digest('hex');
  return sha256 === declared.sha256
    ? { status: 'ok', bytes: info.size, sha256 }
    : { status: 'mismatch', bytes: info.size, sha256, reason: 'the SHA-256 differs' };
}

/**
 * Why the plan selects nothing to fetch or verify, or undefined when it selects a file. A
 * selection of skipped files only, or of no declared file, would let a verification pass
 * without checking a file.
 */
export function emptySelection(plan, only = []) {
  if (plan.files.length) return undefined;
  const which = only.length
    ? `--only ${only.join(',')} selects no uploaded file of the record`
    : 'The record declares no uploaded file';
  const skipped = plan.skipped.map((file) => file.name).join(', ');
  return skipped ? `${which} (the page skipped ${skipped})` : which;
}

/** The wrangler arguments that download the object `key` of `bucket` to `file`. */
export function wranglerArgs(bucket, key, file) {
  return ['r2', 'object', 'get', `${bucket}/${key}`, '--file', file, '--remote'];
}

/**
 * Run Node with `args` from the repository root, without a shell. Resolves when the child exits
 * with 0. Rejects with the end of its output when it exits otherwise, and when it runs longer
 * than `timeoutMs` (default DOWNLOAD_TIMEOUT_MINUTES): the child is then stopped. `name` names
 * the child in the messages.
 */
export function runNode(
  args,
  { name = 'node', timeoutMs = DOWNLOAD_TIMEOUT_MINUTES * 60_000 } = {},
) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, {
      cwd: ROOT,
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let output = '',
      timedOut = false;
    const keep = (data) => (output = (output + data).slice(-4000));
    child.stdout.on('data', keep);
    child.stderr.on('data', keep);
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill();
    }, timeoutMs);
    child.on('error', (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      const tail = output.trim().split('\n').slice(-3).join(' | ');
      if (timedOut)
        reject(
          new Error(
            `${name} did not finish within ${timeoutMs / 1000} s and was stopped` +
              (tail ? `: ${tail}` : ''),
          ),
        );
      else if (code === 0) resolve();
      else reject(new Error(`${name} exited with ${code}: ${tail}`));
    });
  });
}

/** Download one object with the repository's wrangler, stopped after `timeoutMs` (`runNode`). */
function wranglerGet(bucket, key, file, timeoutMs) {
  return runNode([WRANGLER, ...wranglerArgs(bucket, key, path.resolve(file))], {
    name: 'wrangler',
    timeoutMs,
  });
}

/** The --timeout value in ms: a positive number of minutes, within the range of a timer. */
export function parseTimeout(value) {
  if (value === undefined) return DOWNLOAD_TIMEOUT_MINUTES * 60_000;
  const minutes = value.trim() === '' ? NaN : Number(value);
  // setTimeout fires at once for a delay above 2³¹ − 1 ms (about 24.8 days).
  if (!(minutes > 0) || minutes * 60_000 > 2 ** 31 - 1)
    throw new Error('--timeout must be a positive number of minutes, at most 35,791');
  return minutes * 60_000;
}

/**
 * Fetch or verify the files of `plan` in `out`, one at a time, and report each result to
 * `onResult` as it comes. With `verify`, check the files in `out` only (status `ok`, `missing`
 * or `mismatch`). Without it, keep a file that matches (`present`), download the others with
 * `download(bucket, key, file)` to a temporary file, and give a download the file's name only
 * when it matches (`downloaded`, `replaced` when it replaced a file that did not match). A
 * download that does not match is deleted (`mismatch`); a download that throws is `failed`.
 */
export async function fetchRecordFiles(
  plan,
  { out, bucket, verify = false, download = wranglerGet, onResult = () => {} },
) {
  if (!verify) await mkdir(out, { recursive: true });
  const results = [];
  const report = (result) => {
    results.push(result);
    onResult(result);
  };
  for (const file of plan.files) {
    const target = path.join(out, file.name);
    const existing = await verifyFile(target, file);
    if (verify) {
      report({ name: file.name, ...existing });
      continue;
    }
    if (existing.status === 'ok') {
      report({ name: file.name, status: 'present', bytes: file.bytes, sha256: file.sha256 });
      continue;
    }
    const temporary = path.join(out, `.${file.name}.${process.pid}.download`);
    await rm(temporary, { force: true });
    try {
      await download(bucket, file.key, temporary);
    } catch (error) {
      await rm(temporary, { force: true });
      report({ name: file.name, status: 'failed', reason: error.message });
      continue;
    }
    const fetched = await verifyFile(temporary, file);
    if (fetched.status !== 'ok') {
      await rm(temporary, { force: true });
      report({
        name: file.name,
        status: 'mismatch',
        reason: fetched.status === 'missing' ? 'the download wrote no file' : fetched.reason,
      });
      continue;
    }
    await rename(temporary, target);
    report({
      name: file.name,
      status: 'downloaded',
      bytes: file.bytes,
      sha256: file.sha256,
      ...(existing.status === 'mismatch' ? { replaced: true } : {}),
    });
  }
  return results;
}

/** The printed line of one result. */
export function resultLine(result) {
  const detail =
    result.reason ??
    (result.bytes !== undefined
      ? `${result.bytes} bytes${result.replaced ? ', replaced' : ''}`
      : '');
  return `${result.status.padEnd(10)} ${result.name}${detail ? `  ${detail}` : ''}`;
}

async function main(argv) {
  const { values } = parseArgs({
    args: argv,
    strict: true,
    options: {
      record: { type: 'string' },
      out: { type: 'string' },
      only: { type: 'string' },
      verify: { type: 'boolean', default: false },
      timeout: { type: 'string' },
    },
  });
  if (!values.record || !values.out) throw new Error('--record and --out are required');
  const limit = parseTimeout(values.timeout);
  const record = JSON.parse((await readFile(values.record, 'utf8')).replace(/^\uFEFF/, ''));
  const only = (values.only ?? '')
    .split(',')
    .map((prefix) => prefix.trim())
    .filter(Boolean);
  const plan = plannedFiles(record, { only });
  const empty = emptySelection(plan, only);
  if (empty) throw new Error(empty);
  let bucket;
  if (!values.verify) {
    bucket = recordFilesBucket(await readFile(path.join(ROOT, 'wrangler.jsonc'), 'utf8'));
    if (!bucket) throw new Error('wrangler.jsonc has no RECORD_FILES bucket');
    if (!process.env.CLOUDFLARE_API_TOKEN) throw new Error('CLOUDFLARE_API_TOKEN is not set');
  }
  console.log(
    `record ${plan.id}: ${record.files.length} files declared, ${plan.files.length} selected` +
      (bucket ? `, bucket ${bucket}` : ''),
  );
  for (const file of plan.skipped)
    console.log(`${'skipped'.padEnd(10)} ${file.name}  ${file.reason}`);
  const results = await fetchRecordFiles(plan, {
    out: values.out,
    bucket,
    verify: values.verify,
    download: (bucketName, key, file) => wranglerGet(bucketName, key, file, limit),
    onResult: (result) => console.log(resultLine(result)),
  });
  const counts = new Map();
  for (const { status } of results) counts.set(status, (counts.get(status) ?? 0) + 1);
  console.log([...counts].map(([status, n]) => `${n} ${status}`).join(', '));
  if (results.some((result) => !GOOD.has(result.status))) process.exitCode = 1;
}

if (import.meta.main)
  main(process.argv.slice(2)).catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
