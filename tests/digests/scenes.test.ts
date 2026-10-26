import test, { after } from 'node:test';
import { join } from 'node:path';
import { characterize, largerSceneCases } from '../characterization.ts';
import { digestRecorder } from '../typeshade/digest-fixture.mjs';

// The characterization digests of docs/library-structure.md, section 5, phase 0, on the larger
// synthetic scenes: twelve views of tests/scene.ts and the view-graph scene of
// tests/view-graph-scene.ts (`npm run test:digests`; tests/characterization.ts says how).
//
// The fixture holds the digests of Node 22 (its `configuration`). Identity across Node majors
// is not established (CI check (24) on 09894a7), so the check job of CI runs `npm run
// test:digests` on Node 22 only, and the compare job compares a base and a head on every Node
// major (tests/typeshade/digest-fixture.mjs).

const recorder = digestRecorder(join(import.meta.dirname, 'cpu-larger-scenes.json'), {
  update: 'TYPECV_DIGEST_UPDATE=1 node --test tests/digests/scenes.test.ts',
});
after(() => recorder.write());

for (const entry of largerSceneCases())
  test(`digest of ${entry.name}`, async () => {
    recorder.check(entry.name, await characterize(entry));
  });
