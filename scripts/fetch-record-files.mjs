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
// download.
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

/** The wrangler arguments that download the object `key` of `bucket` to `file`. */
export function wranglerArgs(bucket, key, file) {
  return ['r2', 'object', 'get', `${bucket}/${key}`, '--file', file, '--remote'];
}

/** Download one object with the repository's wrangler. Rejects with the end of its output. */
function wranglerGet(bucket, key, file) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      [WRANGLER, ...wranglerArgs(bucket, key, path.resolve(file))],
      { cwd: ROOT, env: process.env, stdio: ['ignore', 'pipe', 'pipe'] },
    );
    let output = '';
    const keep = (data) => (output = (output + data).slice(-4000));
    child.stdout.on('data', keep);
    child.stderr.on('data', keep);
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else
        reject(
          new Error(
            `wrangler exited with ${code}: ` + output.trim().split('\n').slice(-3).join(' | '),
          ),
        );
    });
  });
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
    },
  });
  if (!values.record || !values.out) throw new Error('--record and --out are required');
  const record = JSON.parse((await readFile(values.record, 'utf8')).replace(/^﻿/, ''));
  const only = (values.only ?? '')
    .split(',')
    .map((prefix) => prefix.trim())
    .filter(Boolean);
  const plan = plannedFiles(record, { only });
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
    onResult: (result) => console.log(resultLine(result)),
  });
  const counts = new Map();
  for (const { status } of results) counts.set(status, (counts.get(status) ?? 0) + 1);
  console.log([...counts].map(([status, n]) => `${n} ${status}`).join(', ') || 'no file selected');
  if (results.some((result) => !GOOD.has(result.status))) process.exitCode = 1;
}

if (import.meta.main)
  main(process.argv.slice(2)).catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
