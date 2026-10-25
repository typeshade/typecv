import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdtemp, mkdir, open, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { multipartPlans, reassembleResultFiles } from '../scripts/reassemble-result-files.mjs';
import {
  RESULT_FILE_MAX_BYTES,
  RESULT_FILE_PART_BYTES,
  RESULT_FILES_MAX_BYTES,
  type RunRecordFile,
} from '../website/src/lib/run-record.ts';

let directory: string, partsDir: string, files: RunRecordFile[], originalHash: string;
const name = 'typecv-dense-c000.ply';
const totalBytes = RESULT_FILE_MAX_BYTES + 123;
before(async () => {
  directory = await mkdtemp(path.join(os.tmpdir(), 'typecv-reassemble-test-'));
  partsDir = path.join(directory, 'parts');
  await mkdir(partsDir);
  const hash = createHash('sha256');
  files = [];
  for (let offset = 0, index = 0; offset < totalBytes; index++) {
    const bytes = Math.min(RESULT_FILE_PART_BYTES, totalBytes - offset);
    const buffer = Buffer.alloc(bytes, index + 1);
    const partName = `${name}.part-${String(index).padStart(4, '0')}` as const;
    await writeFile(path.join(partsDir, partName), buffer);
    hash.update(buffer);
    files.push({
      name: partName,
      bytes,
      sha256: createHash('sha256').update(buffer).digest('hex'),
      part: { original: name, totalBytes, offset },
    });
    offset += bytes;
  }
  originalHash = hash.digest('hex');
});
after(async () => {
  await rm(directory, { recursive: true, force: true });
});
const manifest = () => ({ files: structuredClone(files) });
function declaredParts(
  original: string,
  total: number,
  firstSizes: number[] = [],
): RunRecordFile[] {
  const result: RunRecordFile[] = [];
  for (let offset = 0, index = 0; offset < total; index++) {
    const size = Math.min(firstSizes[index] ?? RESULT_FILE_PART_BYTES, total - offset);
    result.push({
      name: `${original}.part-${String(index).padStart(4, '0')}` as RunRecordFile['name'],
      bytes: size,
      sha256: 'a'.repeat(64),
      part: { original: original as typeof name, totalBytes: total, offset },
    });
    offset += size;
  }
  return result;
}

test('multipart recovery streams a complete original and verifies exact coverage and digests', async () => {
  const outDir = path.join(directory, 'complete');
  const results = await reassembleResultFiles(manifest(), { partsDir, outDir });
  assert.deepEqual(results, [{ name, bytes: totalBytes, sha256: originalHash }]);
  assert.deepEqual(await readdir(outDir), [name]);
  const hash = createHash('sha256');
  let read = 0;
  for await (const chunk of createReadStream(path.join(outDir, name), { highWaterMark: 65536 })) {
    hash.update(chunk);
    read += chunk.length;
  }
  assert.equal(read, totalBytes);
  assert.equal(hash.digest('hex'), originalHash);
  const output = await open(path.join(outDir, name), 'r');
  try {
    for (const [index, part] of files.entries()) {
      const byte = Buffer.alloc(1);
      await output.read(byte, 0, 1, part.part!.offset);
      assert.equal(byte[0], index + 1);
    }
  } finally {
    await output.close();
  }
  await assert.rejects(reassembleResultFiles(manifest(), { partsDir, outDir }), /already exists/);
  assert.deepEqual(await readdir(outDir), [name]);
});

test('multipart manifest rejects unsafe names, duplicates, gaps, overlaps and missing coverage', () => {
  const bad = (change: (r: { files: RunRecordFile[] }) => void, error: RegExp) => {
    const record = manifest();
    change(record);
    assert.throws(() => multipartPlans(record), error);
  };
  for (const unsafe of [
    '../typecv-dense.ply.part-0000',
    name + '.part-00000',
    name + '.part-0000.part-0000',
  ])
    bad((r) => {
      r.files[0] = { ...r.files[0]!, name: unsafe as RunRecordFile['name'] };
    }, /filename/);
  bad((r) => {
    r.files.push(r.files[0]!);
  }, /Duplicate/);
  bad((r) => {
    r.files[0] = { ...r.files[0]!, part: { ...r.files[0]!.part!, offset: 1 } };
  }, /gap|overlap/);
  bad((r) => {
    r.files[1] = {
      ...r.files[1]!,
      part: { ...r.files[1]!.part!, offset: RESULT_FILE_PART_BYTES - 1 },
    };
  }, /gap|overlap/);
  bad((r) => {
    r.files.reverse();
  }, /sequence/);
  bad((r) => {
    r.files.pop();
  }, /Missing final/);
  bad((r) => {
    r.files[1] = { ...r.files[1]!, part: { ...r.files[1]!.part!, totalBytes: totalBytes + 1 } };
  }, /inconsistent total/);
  bad((r) => {
    r.files.push({ name, bytes: 1, sha256: 'a'.repeat(64) });
  }, /Both original/);
  bad((r) => {
    r.files[0] = {
      ...r.files[0]!,
      part: { ...r.files[0]!.part!, original: '../outside' as typeof name },
    };
  }, /metadata/);
});

test('skipped parts and missing digests fail explicitly before creating outputs', async () => {
  const skipped = manifest();
  skipped.files[1] = {
    ...skipped.files[1]!,
    skipped: 'The original bytes are unreadable',
    sha256: undefined,
  };
  const outDir = path.join(directory, 'skipped');
  await assert.rejects(
    reassembleResultFiles(skipped, { partsDir, outDir }),
    /required part was skipped/,
  );
  await assert.rejects(readdir(outDir), /ENOENT/);
  const invalid = manifest();
  invalid.files[0] = { ...invalid.files[0]!, sha256: '0'.repeat(63) };
  assert.throws(() => multipartPlans(invalid), /SHA-256/);
  assert.throws(
    () => multipartPlans({ files: [{ name, skipped: 'unreadable' }] }, name),
    /original was skipped/,
  );
});

test('per-original part order can be interleaved, and an explicitly selected healthy group can recover', () => {
  const other = 'typecv-reconstruction-c001.json';
  const second = files.map((part, index) => ({
    ...part,
    name: `${other}.part-${String(index).padStart(4, '0')}` as RunRecordFile['name'],
    part: { ...part.part!, original: other },
  }));
  const mixed = { files: files.flatMap((part, i) => [part, second[i]!]) };
  assert.deepEqual(
    multipartPlans(mixed).map((plan) => plan.name),
    [name, other],
  );
  mixed.files[1] = { ...mixed.files[1]!, skipped: 'unavailable', sha256: undefined };
  assert.equal(multipartPlans(mixed, name).length, 1);
  assert.throws(() => multipartPlans(mixed), /required part was skipped/);
});

test('any positive contiguous part sizes up to the maximum are accepted', () => {
  const parts = declaredParts(name, totalBytes, [1, 2, 3]);
  const plans = multipartPlans({ files: parts });
  assert.equal(plans.length, 1);
  assert.deepEqual(
    plans[0]!.parts.slice(0, 3).map((part) => [part.bytes, part.offset]),
    [
      [1, 0],
      [2, 1],
      [3, 3],
    ],
  );
  assert.equal(
    plans[0]!.parts.reduce((sum, part) => sum + part.bytes, 0),
    totalBytes,
  );
});

test('the shared upload budget counts healthy ordinary files and excludes skipped groups', () => {
  const other = 'typecv-mesh-c001.ply';
  const skipped = declaredParts(other, RESULT_FILES_MAX_BYTES).map((part) => ({
    ...part,
    skipped: 'Not readable',
    sha256: undefined,
  }));
  // Declared original totals exceed 4 GiB, but only the healthy first group was uploaded.
  assert.equal(multipartPlans({ files: [...files, ...skipped] }, name).length, 1);
  assert.throws(
    () => multipartPlans({ files: [...files, ...skipped] }, other),
    /required part was skipped/,
  );
  const ordinary: RunRecordFile[] = Array.from({ length: 82 }, (_, index) => ({
    name: `typecv-mesh-c${String(index + 100).padStart(3, '0')}.ply` as RunRecordFile['name'],
    bytes: RESULT_FILE_MAX_BYTES,
    sha256: 'b'.repeat(64),
  }));
  assert.throws(() => multipartPlans({ files: [...files, ...ordinary] }, name), /total size limit/);
  assert.equal(
    multipartPlans(
      {
        files: [...files, ...ordinary.map((file) => ({ name: file.name, skipped: 'unreadable' }))],
      },
      name,
    ).length,
    1,
  );
});

test('ordinary metadata requires healthy bytes and digest or a valid nonempty skipped reason', () => {
  const ordinary = { name: 'typecv-mesh.ply' };
  for (const invalid of [
    ordinary,
    { ...ordinary, bytes: 1 },
    { ...ordinary, bytes: RESULT_FILE_MAX_BYTES + 1, sha256: 'b'.repeat(64) },
    { ...ordinary, skipped: '' },
    { ...ordinary, skipped: 'x'.repeat(201) },
    { ...ordinary, skipped: 'unreadable', bytes: -1 },
    { ...ordinary, skipped: 'unreadable', sha256: 'broken' },
  ])
    assert.throws(
      () => multipartPlans({ files: [...files, invalid] }, name),
      /byte count|SHA-256|skipped reason/,
    );
  assert.equal(
    multipartPlans({ files: [...files, { ...ordinary, skipped: 'unreadable' }] }, name).length,
    1,
  );
  assert.equal(
    multipartPlans({ files: [...files, { ...ordinary, bytes: 0, sha256: 'b'.repeat(64) }] }, name)
      .length,
    1,
  );
});

test('corrupt and missing local parts remove their partial temporary output', async () => {
  const corrupted = manifest();
  corrupted.files[corrupted.files.length - 1] = {
    ...corrupted.files.at(-1)!,
    sha256: '0'.repeat(64),
  };
  const corruptOut = path.join(directory, 'corrupted');
  await assert.rejects(
    reassembleResultFiles(corrupted, { partsDir, outDir: corruptOut }),
    /SHA-256 mismatch/,
  );
  assert.deepEqual(await readdir(corruptOut), []);
  const missingDir = path.join(directory, 'missing-parts');
  await mkdir(missingDir);
  const missingOut = path.join(directory, 'missing');
  await assert.rejects(
    reassembleResultFiles(manifest(), { partsDir: missingDir, outDir: missingOut }),
    /Missing part/,
  );
  assert.deepEqual(await readdir(missingOut), []);
});

test('existing output bytes survive and selected recovery does not silently accept undeclared files', async () => {
  const outDir = path.join(directory, 'existing');
  await mkdir(outDir);
  await writeFile(path.join(outDir, name), 'keep my original');
  await assert.rejects(reassembleResultFiles(manifest(), { partsDir, outDir }), /already exists/);
  assert.equal(await readFile(path.join(outDir, name), 'utf8'), 'keep my original');
  assert.deepEqual(await readdir(outDir), [name]);
  assert.equal(multipartPlans(manifest(), name).length, 1);
  assert.throws(() => multipartPlans(manifest(), 'typecv-mesh.ply'), /No multipart/);
  assert.throws(() => multipartPlans(manifest(), '../outside'), /Invalid original/);
});
