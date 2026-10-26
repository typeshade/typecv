import test, { after } from 'node:test';
import { join } from 'node:path';
import { characterize, sceneCases } from '../characterization.ts';
import { digestRecorder } from '../typeshade/digest-fixture.mjs';

// The characterization digests of docs/library-structure.md, section 5, phase 0, on the scenes of
// tests/scene.ts: `reconstructIncremental`, `reconstructIncrementalScene`, `refineIncremental`,
// `reconstructComponent` and the focal search on the CPU path, each with its per-request trace
// (tests/characterization.ts).
//
// Fact: the fixture holds the digests of Node 22 (its `configuration`), and identity across Node
// majors is not established: on 09894a7 the CI job check (24) found the two focal-search cases of
// this file different, with equal traces and request counts. So this file is not in `npm test`,
// which runs on every Node major. `npm run test:digests` runs it, and the check job of CI runs that
// script on Node 22 only; the compare job compares a base and a head on every Node major
// (tests/typeshade/digest-fixture.mjs).

const recorder = digestRecorder(join(import.meta.dirname, 'cpu-scenes.json'), {
  update: 'TYPECV_DIGEST_UPDATE=1 node --test tests/digests/cpu-scenes.test.ts',
});
after(() => recorder.write());

for (const entry of sceneCases())
  test(`digest of ${entry.name}`, async () => {
    recorder.check(entry.name, await characterize(entry));
  });
