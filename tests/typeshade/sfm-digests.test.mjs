import test, { after } from 'node:test';
import { join } from 'node:path';
import * as api from '../../dist/gpu.js';
import { digestRecorder } from './digest-fixture.mjs';
import { runDeviceCase } from './sfm-digest-cases.mjs';
import { sceneDeviceCases } from './sfm-digest-inputs.mjs';

// The characterization digests of docs/library-structure.md, section 5, phase 0, on the CPU device
// (TypeShade's CPU tier): `IncrementalSfmSession`, `reconstructIncrementalGpu` and
// `refineIncrementalGpu` on the scenes of tests/scene.ts, with the plan reports and the kernel
// step names of each case as the baselines of the later gates (sfm-digest-cases.mjs says what each
// holds). The CPU device runs the whole work of a chunk key as one chunk, so these do not depend
// on time. sfm-digests-<sample>.test.mjs run the lab samples. `npm run test:digests` runs them
// after `npm run build`; TYPECV_DIGEST_UPDATE=1 records the fixture.
//
// The fixture holds the digests of Node 22 (its `configuration`). Identity across Node majors
// is not established (CI check (24) on 09894a7), so the check job of CI runs `npm run
// test:digests` on Node 22 only, and the compare job compares a base and a head on every Node
// major (tests/typeshade/digest-fixture.mjs).

const recorder = digestRecorder(join(import.meta.dirname, '../digests/cpu-device.json'), {
  compared: ['output', 'trace', 'requests', 'plan', 'steps'],
  update:
    'npm run build && TYPECV_DIGEST_UPDATE=1 node --test tests/typeshade/sfm-digests.test.mjs',
});
after(() => recorder.write());

const open = (onProgress) => api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0, onProgress });

for (const data of sceneDeviceCases())
  test(`digest of ${data.name} on the CPU device`, async (t) => {
    const record = await runDeviceCase(api, open, data, { plans: true });
    if (record === undefined) t.skip('the library lacks an entry point of the case');
    else recorder.check(data.name, record);
  });
