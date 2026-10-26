import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  DOWNLOAD_TIMEOUT_MINUTES,
  FILE_PREFIX,
  emptySelection,
  fetchRecordFiles,
  parseTimeout,
  plannedFiles,
  recordFilesBucket,
  resultLine,
  runNode,
  verifyFile,
  wranglerArgs,
  type FetchResult,
} from '../scripts/fetch-record-files.mjs';

const ID = '20261026043150261-58532c69';
const sha256 = (text: string) => createHash('sha256').update(text).digest('hex');
/** The bytes of each uploaded file of the record. */
const CONTENTS: Record<string, string> = {
  'typecv-reconstruction.json': '{"format":"typecv-reconstruction-v1"}',
  'typecv-reconstruction.ply': 'ply\nformat ascii 1.0\n',
  'typecv-dense.ply': 'ply\ndense\n',
  'typecv-depths-c000.json': '{"views":[]}',
  'typecv-depths-c000-0000.tar.gz': 'depth archive 0',
};
const record = {
  format: 'typecv-run-record-v6',
  server: { id: ID, receivedAt: '2026-10-26T04:31:50.261Z' },
  files: [
    ...Object.entries(CONTENTS).map(([name, text]) => ({
      name,
      bytes: Buffer.byteLength(text),
      sha256: sha256(text),
    })),
    { name: 'typecv-mesh.ply', skipped: 'The mesh is larger than 50 MiB' },
  ],
};

let directory: string;
before(async () => {
  directory = await mkdtemp(path.join(os.tmpdir(), 'typecv-fetch-test-'));
});
after(async () => {
  await rm(directory, { recursive: true, force: true });
});

test('the bucket and the key of a stored file are those of wrangler.jsonc and the worker', () => {
  assert.equal(
    recordFilesBucket(readFileSync(new URL('../wrangler.jsonc', import.meta.url), 'utf8')),
    'typecv-records',
  );
  assert.equal(
    recordFilesBucket('"r2_buckets": [\n  { "binding": "SAMPLES", "bucket_name": "x" },\n]'),
    undefined,
  );
  assert.equal(FILE_PREFIX, 'records/');
  assert.deepEqual(wranglerArgs('typecv-records', 'records/a/b.json', '/tmp/b.json'), [
    'r2',
    'object',
    'get',
    'typecv-records/records/a/b.json',
    '--file',
    '/tmp/b.json',
    '--remote',
  ]);
});

test('the plan selects the declared files by prefix and lists the skipped ones', () => {
  const plan = plannedFiles(record);
  assert.equal(plan.id, ID);
  assert.deepEqual(
    plan.files.map((file) => [file.name, file.key]),
    Object.keys(CONTENTS).map((name) => [name, `records/${ID}/${name}`]),
  );
  assert.equal(plan.files[2]!.bytes, CONTENTS['typecv-dense.ply']!.length);
  assert.equal(plan.files[2]!.sha256, sha256(CONTENTS['typecv-dense.ply']!));
  assert.deepEqual(plan.skipped, [
    { name: 'typecv-mesh.ply', reason: 'The mesh is larger than 50 MiB' },
  ]);
  const names = (only: string[]) => plannedFiles(record, { only }).files.map((file) => file.name);
  assert.deepEqual(names(['typecv-depths-c000.json']), ['typecv-depths-c000.json']);
  assert.deepEqual(names(['typecv-depths-c000-']), ['typecv-depths-c000-0000.tar.gz']);
  assert.deepEqual(names(['typecv-reconstruction', 'typecv-dense']), [
    'typecv-reconstruction.json',
    'typecv-reconstruction.ply',
    'typecv-dense.ply',
  ]);
  const mesh = plannedFiles(record, { only: ['typecv-mesh'] });
  assert.deepEqual([mesh.files.length, mesh.skipped.length], [0, 1]);
  // Manifests that the collector would not have stored.
  const entry = record.files[0]!;
  assert.throws(() => plannedFiles({ files: record.files }), /server\.id/);
  assert.throws(() => plannedFiles({ ...record, server: { id: '../x' } }), /server\.id/);
  assert.throws(
    () => plannedFiles({ ...record, files: [{ ...entry, name: '../typecv-dense.ply' }] }),
    /Invalid result file name/,
  );
  assert.throws(
    () => plannedFiles({ ...record, files: [entry, entry] }),
    /Repeated result file name/,
  );
  assert.throws(
    () => plannedFiles({ ...record, files: [{ name: entry.name, bytes: 1 }] }),
    /SHA-256/,
  );
});

test('a selection without an uploaded file is refused, so that a verification checks a file', () => {
  assert.equal(emptySelection(plannedFiles(record)), undefined);
  const select = (only: string[]) => emptySelection(plannedFiles(record, { only }), only);
  assert.equal(
    select(['typecv-dense-c']),
    '--only typecv-dense-c selects no uploaded file of the record',
  );
  assert.equal(
    select(['typecv-mesh']),
    '--only typecv-mesh selects no uploaded file of the record (the page skipped typecv-mesh.ply)',
  );
  assert.equal(
    emptySelection(plannedFiles({ ...record, files: [] })),
    'The record declares no uploaded file',
  );
});

test('the command exits 1 when --only selects no file, and 0 when the selected files verify', async () => {
  const dir = path.join(directory, 'command');
  await mkdir(dir, { recursive: true });
  const recordPath = path.join(dir, 'record.json');
  // With a byte order mark, as Windows PowerShell 5.1 writes a file.
  await writeFile(recordPath, '\uFEFF' + JSON.stringify(record));
  await writeFile(path.join(dir, 'typecv-dense.ply'), CONTENTS['typecv-dense.ply']!);
  const script = fileURLToPath(new URL('../scripts/fetch-record-files.mjs', import.meta.url));
  const run = (...args: string[]) =>
    spawnSync(process.execPath, [script, '--record', recordPath, '--out', dir, ...args], {
      encoding: 'utf8',
    });
  const typo = run('--verify', '--only', 'typecv-dense-c');
  assert.equal(typo.status, 1);
  assert.equal(typo.stderr.trim(), '--only typecv-dense-c selects no uploaded file of the record');
  const good = run('--verify', '--only', 'typecv-dense');
  assert.equal(good.status, 0, good.stderr);
  assert.match(good.stdout, /^ok {9}typecv-dense\.ply {2}10 bytes$/m);
  assert.equal(run('--verify', '--timeout', '0').status, 1);
});

test('a download that runs too long is stopped and reported', async () => {
  assert.equal(DOWNLOAD_TIMEOUT_MINUTES, 10);
  assert.equal(parseTimeout(undefined), 600_000);
  assert.equal(parseTimeout('1.5'), 90_000);
  for (const value of ['', ' ', '0', '-1', 'x', '40000'])
    assert.throws(() => parseTimeout(value), /--timeout must be a positive number/, value);
  const started = performance.now();
  await assert.rejects(runNode(['-e', 'setTimeout(() => {}, 60000)'], { timeoutMs: 300 }), {
    message: 'node did not finish within 0.3 s and was stopped',
  });
  assert.ok(performance.now() - started < 30_000);
  await assert.rejects(
    runNode(['-e', 'console.error("The specified key does not exist."); process.exit(3)'], {
      name: 'wrangler',
    }),
    { message: 'wrangler exited with 3: The specified key does not exist.' },
  );
  await runNode(['-e', 'process.exit(0)']);
});

test('a file verifies by its size and its SHA-256', async () => {
  const dir = path.join(directory, 'verify');
  await mkdir(path.join(dir, 'folder'), { recursive: true });
  const declared = { bytes: 5, sha256: sha256('abcde') };
  await writeFile(path.join(dir, 'good'), 'abcde');
  await writeFile(path.join(dir, 'short'), 'abcd');
  await writeFile(path.join(dir, 'other'), 'abcdf');
  assert.deepEqual(await verifyFile(path.join(dir, 'good'), declared), {
    status: 'ok',
    bytes: 5,
    sha256: declared.sha256,
  });
  assert.deepEqual(await verifyFile(path.join(dir, 'short'), declared), {
    status: 'mismatch',
    bytes: 4,
    reason: '4 bytes, the record declares 5',
  });
  const other = await verifyFile(path.join(dir, 'other'), declared);
  assert.deepEqual([other.status, other.reason], ['mismatch', 'the SHA-256 differs']);
  assert.deepEqual(await verifyFile(path.join(dir, 'none'), declared), { status: 'missing' });
  assert.equal((await verifyFile(path.join(dir, 'folder'), declared)).status, 'mismatch');
});

test('fetch: keep a matching file, rename a matching download, delete a mismatch', async () => {
  const out = path.join(directory, 'fetch');
  await mkdir(out);
  await writeFile(
    path.join(out, 'typecv-reconstruction.json'),
    CONTENTS['typecv-reconstruction.json']!,
  );
  await writeFile(path.join(out, 'typecv-dense.ply'), 'ply\nstale\n');
  const plan = plannedFiles(record);
  const calls: [string | undefined, string, string][] = [];
  const download = async (bucket: string | undefined, key: string, file: string) => {
    calls.push([bucket, key, file]);
    const name = key.slice(`records/${ID}/`.length);
    if (name === 'typecv-reconstruction.ply') throw new Error('wrangler exited with 1: not found');
    await writeFile(file, name.endsWith('.tar.gz') ? 'a corrupted archive' : CONTENTS[name]!);
  };
  const reported: FetchResult[] = [];
  const results = await fetchRecordFiles(plan, {
    out,
    bucket: 'typecv-records',
    download,
    onResult: (result) => reported.push(result),
  });
  assert.deepEqual(reported, results);
  assert.deepEqual(
    results.map((r) => [r.name, r.status, r.replaced ?? false]),
    [
      ['typecv-reconstruction.json', 'present', false],
      ['typecv-reconstruction.ply', 'failed', false],
      ['typecv-dense.ply', 'downloaded', true],
      ['typecv-depths-c000.json', 'downloaded', false],
      ['typecv-depths-c000-0000.tar.gz', 'mismatch', false],
    ],
  );
  assert.equal(results[1]!.reason, 'wrangler exited with 1: not found');
  assert.equal(results[4]!.reason, '19 bytes, the record declares 15');
  // No download for the matching file; each download goes to a temporary file in --out.
  assert.deepEqual(
    calls.map(([bucket, key]) => [bucket, key]),
    plan.files.slice(1).map((file) => ['typecv-records', file.key]),
  );
  for (const [, key, file] of calls) {
    assert.equal(path.dirname(file), out);
    assert.notEqual(path.basename(file), path.basename(key));
  }
  // Only the matching files are left: no temporary file and no mismatch.
  assert.deepEqual((await readdir(out)).sort(), [
    'typecv-dense.ply',
    'typecv-depths-c000.json',
    'typecv-reconstruction.json',
  ]);
  assert.equal(
    await readFile(path.join(out, 'typecv-dense.ply'), 'utf8'),
    CONTENTS['typecv-dense.ply'],
  );
  // --verify reads the files in --out and downloads nothing.
  const verified = await fetchRecordFiles(plan, {
    out,
    verify: true,
    download: async () => assert.fail('--verify must not download'),
  });
  assert.deepEqual(
    verified.map((r) => r.status),
    ['ok', 'missing', 'ok', 'ok', 'missing'],
  );
});

test('the printed line of a result', () => {
  assert.equal(
    resultLine({ name: 'typecv-dense.ply', status: 'downloaded', bytes: 12, replaced: true }),
    'downloaded typecv-dense.ply  12 bytes, replaced',
  );
  assert.equal(
    resultLine({ name: 'typecv-depths-c000.json', status: 'missing' }),
    'missing    typecv-depths-c000.json',
  );
  assert.equal(
    resultLine({ name: 'a.ply', status: 'mismatch', bytes: 3, reason: 'the SHA-256 differs' }),
    'mismatch   a.ply  the SHA-256 differs',
  );
});
