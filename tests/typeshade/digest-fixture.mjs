// The fixtures of the characterization digests (docs/library-structure.md, section 5, phase 0):
// a JSON file of `{ update, configuration, cases: { <name>: <record> } }` in tests/digests/. A
// test computes the record of each case and hands it to `check`; at the end `write` writes what
// the environment asks for:
//
// - `TYPECV_DIGEST_UPDATE=1`: the fixture itself, with the records and the configuration of this
//   run;
// - `TYPECV_DIGEST_OUT=<directory>`: `<directory>/<fixture name>` with the records and the
//   configuration of this run, and no comparison (the compare job of CI, which compares a base
//   and a head on one runner).
//
// Otherwise `check` compares the fields `compared` of each record with the fixture's.
//
// The configuration of a fixture is the one that recorded it: the Node and V8 versions, the
// platform, the commit and the git tree of src/. Fact: identity of the CPU-tier digests across
// Node majors is not established. On 09894a7 the CI job check (24) found the two focal-search
// cases of the scene fixture different from their Node 22 records, with equal traces and request
// counts. Inference: an intrinsic such as `**` or `Math.pow`, which the focal grid computes
// (`focalGrid` of src/reconstruction/focal-search.ts), differs in its last bits between the V8 of
// Node 22 and that of Node 24. So the check job of .github/workflows/ci.yml compares the fixtures
// on the Node major that recorded them only, and the compare job compares a base and a head on
// every Node major of the matrix, on one runner. A comparison on another Node major still runs,
// and its message names both configurations.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';

const ROOT = join(import.meta.dirname, '../..');
const byName = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
const sorted = (cases) =>
  Object.fromEntries(
    Object.keys(cases)
      .sort(byName)
      .map((k) => [k, cases[k]]),
  );

function git(args) {
  try {
    return execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', stdio: 'pipe' }).trim();
  } catch {
    return undefined;
  }
}

/**
 * The configuration of this run: the Node and V8 versions, the platform, the commit and the git
 * tree of src/ (`modified` when src/ differs from the commit).
 */
export function runConfiguration() {
  const modified = (git(['status', '--porcelain', '--', 'src']) ?? '') !== '';
  return {
    node: process.version,
    v8: process.versions.v8,
    platform: `${process.platform} ${process.arch}`,
    commit: git(['rev-parse', 'HEAD']) ?? 'unknown',
    src: modified ? 'modified' : (git(['rev-parse', 'HEAD:src']) ?? 'unknown'),
  };
}

const major = (version) => /^v?(\d+)/.exec(version ?? '')?.[1];

/** A fixture file: its configuration and its cases; none when it does not exist. */
export function readFixtureFile(file) {
  try {
    const data = JSON.parse(readFileSync(file, 'utf8'));
    return { configuration: data.configuration, cases: data.cases ?? {} };
  } catch {
    return { configuration: undefined, cases: {} };
  }
}

/** The recorded cases of a fixture file; none when it does not exist. */
export function readFixture(file) {
  return readFixtureFile(file).cases;
}

/** What a mismatch says of the configuration that recorded the fixture and of this run's. */
export function configurationNote(recorded, current) {
  if (!recorded) return '';
  const describe = (c) => `Node ${c.node} (V8 ${c.v8}, ${c.platform})`;
  const note =
    ` The fixture was recorded on ${describe(recorded)} at ${recorded.commit}; ` +
    `this run is on ${describe(current)}.`;
  if (major(recorded.node) === major(current.node)) return note;
  return (
    `${note} Identity across Node majors is not established for the CPU tier (CI check (24) ` +
    `on 09894a7), so CI compares this fixture on Node ${major(recorded.node)} only.`
  );
}

/** Compare the fields `compared` of a case's record with the fixture's, naming what changed. */
export function compareRecord(name, recorded, current, compared, note = '') {
  assert.ok(recorded, `${name}: no recorded digest; run with TYPECV_DIGEST_UPDATE=1 to record it`);
  const parts = Object.keys({ ...recorded.parts, ...current.parts }).filter(
    (key) => recorded.parts?.[key] !== current.parts?.[key],
  );
  const pick = (record) => Object.fromEntries(compared.map((key) => [key, record[key]]));
  assert.deepEqual(
    pick(current),
    pick(recorded),
    `${name}: the digests changed (changed parts of the output: ${parts.join(', ') || 'none'}). ` +
      `A phase that renews a digest names it and records it with TYPECV_DIGEST_UPDATE=1.${note}`,
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
  const { configuration: recordedConfiguration, cases: recorded } = readFixtureFile(fixture);
  const configuration = runConfiguration();
  const note = configurationNote(recordedConfiguration, configuration);
  const current = {};
  const out = process.env.TYPECV_DIGEST_OUT;
  const update = process.env.TYPECV_DIGEST_UPDATE === '1';
  return {
    check(name, record) {
      current[name] = record;
      if (!out && !update) compareRecord(name, recorded[name], record, compared, note);
    },
    write() {
      if (update) {
        const cases = sorted({ ...recorded, ...current });
        const text = { ...(command ? { update: command } : {}), configuration, cases };
        writeFileSync(fixture, `${JSON.stringify(text, null, 2)}\n`);
      }
      if (out) {
        mkdirSync(out, { recursive: true });
        writeFileSync(
          join(out, basename(fixture)),
          `${JSON.stringify({ configuration, cases: sorted(current) }, null, 2)}\n`,
        );
      }
    },
  };
}
