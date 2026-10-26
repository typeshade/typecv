import test, { after } from 'node:test';
import { join } from 'node:path';
import * as api from '../../dist/gpu.js';
import { digestRecorder } from './digest-fixture.mjs';
import { runDeviceCase } from './sfm-digest-cases.mjs';
import { sampleDeviceCase } from './sfm-digest-inputs.mjs';

// The characterization digests of docs/library-structure.md, section 5, phase 0, on the CPU device
// (TypeShade's CPU tier): `IncrementalSfmSession` on the first six frames of the lab sample
// graham-hall, with the lab's controls, and the plan reports and kernel step names of the case
// (sfm-digest-cases.mjs). One file for each sample, so that the samples run in parallel.
// `npm run test:digests` runs it after `npm run build`; TYPECV_DIGEST_UPDATE=1 records the fixture.

const recorder = digestRecorder(
  join(import.meta.dirname, '../digests/cpu-device-graham-hall.json'),
  {
    compared: ['output', 'trace', 'requests', 'plan', 'steps'],
    update:
      'npm run build && TYPECV_DIGEST_UPDATE=1 node --test tests/typeshade/sfm-digests-graham-hall.test.mjs',
  },
);
after(() => recorder.write());

const open = (onProgress) => api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0, onProgress });
const data = sampleDeviceCase('graham-hall');

test(`digest of ${data.name} on the CPU device`, async (t) => {
  const record = await runDeviceCase(api, open, data, { plans: true });
  if (record === undefined) t.skip('the library lacks an entry point of the case');
  else recorder.check(data.name, record);
});
