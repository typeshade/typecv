import test, { after } from 'node:test';
import { join } from 'node:path';
import { characterize, sampleCases } from '../characterization.ts';
import { digestRecorder } from '../typeshade/digest-fixture.mjs';

// The characterization digests of docs/library-structure.md, section 5, phase 0, on the lab
// sample south-building that scripts/measure-focal.mjs reads (`npm run test:digests`;
// tests/characterization.ts says how). One file for each sample, so that the samples run in
// parallel.

const recorder = digestRecorder(join(import.meta.dirname, 'cpu-south-building.json'), {
  update: 'TYPECV_DIGEST_UPDATE=1 node --test tests/digests/south-building.test.ts',
});
after(() => recorder.write());

for (const entry of sampleCases('south-building'))
  test(`digest of ${entry.name}`, async () => {
    recorder.check(entry.name, await characterize(entry));
  });
