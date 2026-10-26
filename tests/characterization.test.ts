import test, { after } from 'node:test';
import { join } from 'node:path';
import { characterize, sceneCases } from './characterization.ts';
import { digestRecorder } from './typeshade/digest-fixture.mjs';

// The characterization digests of docs/library-structure.md, section 5, phase 0, on the scenes of
// tests/scene.ts: `reconstructIncremental`, `reconstructIncrementalScene`, `refineIncremental`,
// `reconstructComponent` and the focal search on the CPU path, each with its per-request trace
// (tests/characterization.ts). The larger scenes and the four samples are in tests/digests/
// (`npm run test:digests`). A structure phase keeps every digest; a phase that renews one names
// it and records it with TYPECV_DIGEST_UPDATE=1 npm test.

const recorder = digestRecorder(join(import.meta.dirname, 'digests/cpu-scenes.json'), {
  update: 'TYPECV_DIGEST_UPDATE=1 node --test tests/characterization.test.ts',
});
after(() => recorder.write());

for (const entry of sceneCases())
  test(`digest of ${entry.name}`, async () => {
    recorder.check(entry.name, await characterize(entry));
  });
