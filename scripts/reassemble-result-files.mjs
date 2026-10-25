#!/usr/bin/env node
/** Reassemble declared result parts without loading an original file into memory.
 * node scripts/reassemble-result-files.mjs --record run.json --parts downloaded --out recovered
 * Add --file typecv-dense-c000.ply to recover only one declared original.
 */
import { createHash } from 'node:crypto';
import { lstat, mkdir, mkdtemp, open, readFile, link, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  RESULT_FILE_MAX_COUNT,
  RESULT_FILE_MAX_BYTES,
  RESULT_FILE_PART_BYTES,
  RESULT_FILES_MAX_BYTES,
  RUN_RECORD_MAX_BYTES,
  resultFileBaseNameValid,
  resultFileNameValid,
} from '../website/src/lib/run-record.ts';

/** Validate the complete manifest before creating any output. Ordinary files need no joining. */
export function multipartPlans(record, original) {
  if (original !== undefined && !resultFileBaseNameValid(original))
    throw new Error('Invalid original result filename');
  const files = record?.files;
  if (!Array.isArray(files) || files.length > RESULT_FILE_MAX_COUNT)
    throw new Error('The record has no valid result-file manifest');
  const names = new Set();
  const ordinary = new Map();
  const groups = new Map();
  let uploadedBytes = 0;
  for (const file of files) {
    if (!file || !resultFileNameValid(file.name)) throw new Error('Invalid result filename');
    if (names.has(file.name)) throw new Error('Duplicate result filename: ' + file.name);
    names.add(file.name);
    if (file.bytes !== undefined && (!Number.isSafeInteger(file.bytes) || file.bytes < 0))
      throw new Error('Invalid declared byte count: ' + file.name);
    if (
      file.sha256 !== undefined &&
      (typeof file.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(file.sha256))
    )
      throw new Error('Missing or invalid SHA-256: ' + file.name);
    if (
      file.skipped !== undefined &&
      (typeof file.skipped !== 'string' || file.skipped.length === 0 || file.skipped.length > 200)
    )
      throw new Error('Invalid skipped reason: ' + file.name);
    if (file.skipped === undefined) {
      if (!Number.isSafeInteger(file.bytes) || file.bytes < 0 || file.bytes > RESULT_FILE_MAX_BYTES)
        throw new Error('Missing or invalid declared byte count: ' + file.name);
      if (file.sha256 === undefined) throw new Error('Missing or invalid SHA-256: ' + file.name);
      uploadedBytes += file.bytes;
      if (uploadedBytes > RESULT_FILES_MAX_BYTES)
        throw new Error('Uploaded result bytes exceed the total size limit');
    }
    const match = /^(.*)\.part-([0-9]{4})$/.exec(file.name);
    if (!match) {
      if (file.part !== undefined) throw new Error('Part metadata on an ordinary result file');
      ordinary.set(file.name, file);
      continue;
    }
    const part = file.part;
    if (
      !part ||
      !resultFileBaseNameValid(part.original) ||
      part.original !== match[1] ||
      !Number.isSafeInteger(part.totalBytes) ||
      part.totalBytes <= RESULT_FILE_MAX_BYTES ||
      part.totalBytes > RESULT_FILES_MAX_BYTES ||
      !Number.isSafeInteger(part.offset) ||
      part.offset < 0 ||
      !Number.isSafeInteger(file.bytes) ||
      file.bytes <= 0 ||
      file.bytes > RESULT_FILE_PART_BYTES
    )
      throw new Error('Invalid part metadata: ' + file.name);
    const group = groups.get(part.original) ?? {
      name: part.original,
      totalBytes: part.totalBytes,
      end: 0,
      parts: [],
    };
    if (
      group.totalBytes !== part.totalBytes ||
      group.end !== part.offset ||
      group.parts.length !== Number(match[2]) ||
      group.end + file.bytes > group.totalBytes
    )
      throw new Error('Part sequence has a gap, overlap or inconsistent total: ' + file.name);
    group.parts.push({
      name: file.name,
      bytes: file.bytes,
      sha256: file.sha256,
      offset: part.offset,
      ...(file.skipped !== undefined ? { skipped: file.skipped } : {}),
    });
    group.end += file.bytes;
    groups.set(part.original, group);
  }
  for (const group of groups.values()) {
    if (ordinary.has(group.name))
      throw new Error('Both original and parts declared: ' + group.name);
    if (group.end !== group.totalBytes) throw new Error('Missing final part: ' + group.name);
  }
  const plans = [...groups.values()].filter(
    (group) => original === undefined || group.name === original,
  );
  if (!plans.length) {
    if (original && ordinary.get(original)?.skipped !== undefined)
      throw new Error('The requested original was skipped: ' + original);
    throw new Error('No multipart result declared' + (original ? ': ' + original : ''));
  }
  for (const group of plans)
    for (const part of group.parts) {
      if (part.skipped !== undefined) throw new Error('A required part was skipped: ' + part.name);
      if (typeof part.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(part.sha256))
        throw new Error('Missing or invalid SHA-256: ' + part.name);
    }
  return plans.map(({ end: _end, ...group }) => group);
}

async function writeAll(file, bytes) {
  let offset = 0;
  while (offset < bytes.length) {
    const { bytesWritten } = await file.write(bytes, offset, bytes.length - offset, null);
    if (!bytesWritten) throw new Error('Could not write the reconstructed file');
    offset += bytesWritten;
  }
}

/** Each original is published only after every part has passed its own size and SHA check. */
export async function reassembleResultFiles(record, { partsDir, outDir, original } = {}) {
  if (typeof partsDir !== 'string' || typeof outDir !== 'string')
    throw new Error('Both partsDir and outDir are required');
  const plans = multipartPlans(record, original);
  if (!(await stat(partsDir)).isDirectory()) throw new Error('The parts path is not a directory');
  await mkdir(outDir, { recursive: true, mode: 0o700 });
  // Preflight all existing names; link() below also prevents overwrite if a name appears later.
  for (const plan of plans)
    try {
      await lstat(path.join(outDir, plan.name));
      throw new Error('Output already exists: ' + plan.name);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
  const results = [];
  for (const plan of plans) {
    const temporary = await mkdtemp(path.join(outDir, '.typecv-reassemble-'));
    const filename = path.join(temporary, 'result');
    let output;
    try {
      output = await open(filename, 'wx', 0o600);
      const wholeHash = createHash('sha256');
      let total = 0;
      for (const part of plan.parts) {
        const inputPath = path.join(partsDir, part.name);
        let input;
        try {
          const before = await lstat(inputPath);
          if (!before.isFile() || before.isSymbolicLink())
            throw new Error('Part is not a regular file: ' + part.name);
          input = await open(inputPath, 'r');
          const opened = await input.stat();
          if (before.dev !== opened.dev || before.ino !== opened.ino || opened.size !== part.bytes)
            throw new Error('Part changed or has the wrong byte count: ' + part.name);
          const hash = createHash('sha256');
          let read = 0;
          for await (const chunk of input.createReadStream({
            highWaterMark: 256 * 1024,
            autoClose: false,
          })) {
            read += chunk.length;
            if (read > part.bytes) throw new Error('Part grew while reading: ' + part.name);
            hash.update(chunk);
            wholeHash.update(chunk);
            await writeAll(output, chunk);
          }
          if (read !== part.bytes || hash.digest('hex') !== part.sha256)
            throw new Error('Part byte count or SHA-256 mismatch: ' + part.name);
          total += read;
        } catch (error) {
          if (error.code === 'ENOENT')
            throw new Error('Missing part: ' + part.name, { cause: error });
          throw error;
        } finally {
          await input?.close();
        }
      }
      if (total !== plan.totalBytes) throw new Error('Original byte count mismatch: ' + plan.name);
      await output.sync();
      await output.close();
      output = undefined;
      // Same-filesystem hard linking publishes verified bytes without a second full disk copy.
      // Unlike rename(), it never overwrites an existing output, including a concurrent creation.
      await link(filename, path.join(outDir, plan.name));
      results.push({ name: plan.name, bytes: total, sha256: wholeHash.digest('hex') });
    } finally {
      try {
        await output?.close();
      } finally {
        await rm(temporary, { recursive: true, force: true });
      }
    }
  }
  return results;
}

async function main(args) {
  if (args.includes('--help') || args.includes('-h')) {
    console.log(
      'Usage: node scripts/reassemble-result-files.mjs --record run.json --parts downloaded --out recovered [--file original]',
    );
    return;
  }
  const options = {};
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i];
    if (!['--record', '--parts', '--out', '--file'].includes(key) || !args[i + 1] || options[key])
      throw new Error('Invalid command-line options; use --help');
    options[key] = args[i + 1];
  }
  if (!options['--record'] || !options['--parts'] || !options['--out'])
    throw new Error('--record, --parts and --out are required');
  const info = await stat(options['--record']);
  if (!info.isFile() || info.size > RUN_RECORD_MAX_BYTES)
    throw new Error('Expected a small run-record JSON, not a reconstruction or result part');
  const record = JSON.parse((await readFile(options['--record'], 'utf8')).replace(/^\uFEFF/, ''));
  const results = await reassembleResultFiles(record, {
    partsDir: options['--parts'],
    outDir: options['--out'],
    original: options['--file'],
  });
  for (const result of results) console.log(JSON.stringify(result));
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url)
  main(process.argv.slice(2)).catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
