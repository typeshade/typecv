// The fixtures of the characterization digests (docs/library-structure.md, section 5, phase 0):
// a JSON file of `{ cases: { <name>: <record> } }` in tests/digests/. A test computes the record of
// each case and hands it to `check`; at the end `write` writes what the environment asks for:
//
// - `TYPECV_DIGEST_UPDATE=1`: the fixture itself, with the records of this run;
// - `TYPECV_DIGEST_OUT=<directory>`: `<directory>/<fixture name>` with the records of this run, and
//   no comparison (the compare job of CI, which compares a base and a head on one runner).
//
// Otherwise `check` compares the fields `compared` of each record with the fixture's.
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';

const byName = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
const sorted = (cases) =>
  Object.fromEntries(
    Object.keys(cases)
      .sort(byName)
      .map((k) => [k, cases[k]]),
  );

/** The recorded cases of a fixture file; none when it does not exist. */
export function readFixture(file) {
  try {
    return JSON.parse(readFileSync(file, 'utf8')).cases;
  } catch {
    return {};
  }
}

/** Compare the fields `compared` of a case's record with the fixture's, naming what changed. */
export function compareRecord(name, recorded, current, compared) {
  assert.ok(recorded, `${name}: no recorded digest; run with TYPECV_DIGEST_UPDATE=1 to record it`);
  const parts = Object.keys({ ...recorded.parts, ...current.parts }).filter(
    (key) => recorded.parts?.[key] !== current.parts?.[key],
  );
  const pick = (record) => Object.fromEntries(compared.map((key) => [key, record[key]]));
  assert.deepEqual(
    pick(current),
    pick(recorded),
    `${name}: the digests changed (changed parts of the output: ${parts.join(', ') || 'none'}). ` +
      'A phase that renews a digest names it and records it with TYPECV_DIGEST_UPDATE=1.',
  );
}

/**
 * Collect the records of one fixture file; see the head of this file. `update` is the command
 * that records the fixture; the fixture names it.
 */
export function digestRecorder(
  fixture,
  { compared = ['output', 'trace', 'requests'], update: command } = {},
) {
  const recorded = readFixture(fixture);
  const current = {};
  const out = process.env.TYPECV_DIGEST_OUT;
  const update = process.env.TYPECV_DIGEST_UPDATE === '1';
  return {
    check(name, record) {
      current[name] = record;
      if (!out && !update) compareRecord(name, recorded[name], record, compared);
    },
    write() {
      if (update) {
        const cases = sorted({ ...recorded, ...current });
        writeFileSync(
          fixture,
          `${JSON.stringify({ ...(command ? { update: command } : {}), cases }, null, 2)}\n`,
        );
      }
      if (out) {
        mkdirSync(out, { recursive: true });
        writeFileSync(
          join(out, basename(fixture)),
          `${JSON.stringify({ cases: sorted(current) }, null, 2)}\n`,
        );
      }
    },
  };
}
