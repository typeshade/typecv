import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { camerasAt, LAB, loadSample, SAMPLES } from '../digest-samples.ts';
import * as script from '../../scripts/measure-focal.mjs';

// The baseline of the output of scripts/measure-focal.mjs (docs/library-structure.md, section 5,
// phase 0; phase 2's gate). The fixture holds the rows of the default command. This test runs one
// start of each sample by default, the four samples in parallel, and compares the rows; each row
// is a run of its own, so it equals its row of the default command. TYPECV_MEASURE_FOCAL=full
// runs every start; TYPECV_DIGEST_UPDATE=1 runs every start and records the rows.

const root = join(import.meta.dirname, '../..');
const file = join(import.meta.dirname, 'measure-focal.json');
const fixture = JSON.parse(readFileSync(file, 'utf8'));
const update = process.env.TYPECV_DIGEST_UPDATE === '1';
const starts: number[] =
  update || process.env.TYPECV_MEASURE_FOCAL === 'full' ? fixture.starts : [0.8];

async function rowsOf(sample: string): Promise<Record<string, unknown>[]> {
  const { stdout } = await promisify(execFile)(
    process.execPath,
    ['scripts/measure-focal.mjs', `--samples=${sample}`, `--starts=${starts.join(',')}`],
    { cwd: root, maxBuffer: 1 << 24 },
  );
  return stdout
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line) => {
      const { ms: _ms, ...row } = JSON.parse(line);
      return row;
    });
}

test(`scripts/measure-focal.mjs prints its recorded rows (starts ${starts.join(', ')})`, async () => {
  const rows = (await Promise.all(SAMPLES.map(rowsOf))).flat();
  if (update) {
    writeFileSync(file, `${JSON.stringify({ ...fixture, rows }, null, 2)}\n`);
    return;
  }
  assert.deepEqual(
    rows,
    fixture.rows.filter((row: { start: number }) => starts.includes(row.start)),
    'The output of scripts/measure-focal.mjs changed; phase 2 of docs/library-structure.md keeps it.',
  );
});

test('the digests read the samples and the lab controls as scripts/measure-focal.mjs does', () => {
  assert.deepEqual(LAB, script.LAB);
  for (const sample of SAMPLES) {
    const ours = loadSample(sample),
      theirs = script.loadSample(sample);
    assert.deepEqual(ours, theirs, sample);
    assert.deepEqual(camerasAt(ours.frames, 1.2, true), script.camerasAt(theirs.frames, 1.2, true));
  }
});
