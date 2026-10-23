// Standalone BA assembly probe; reports CPU-emulation times unless hardware is required.
import { writeFile } from 'node:fs/promises';
import { buildComputeProbe } from '../tests/typeshade/compute-probe-build.mjs';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const file = await buildComputeProbe('bundle-camera-probe');
const { result, errors } = await inBrowser(
  async (hardware) => {
    const { measureCameraWorkgroups } = await import('/probe/camera.js');
    return measureCameraWorkgroups({ hardware });
  },
  process.env.TYPECV_HARDWARE_GPU === '1',
  { timeoutMs: 180000, serve: { '/probe/camera.js': file } },
);
if (errors.length || result.validation.length)
  throw new Error(JSON.stringify({ errors, validation: result.validation }));
console.log(JSON.stringify(result, null, 2));
const out = process.argv.indexOf('--out');
if (out >= 0) await writeFile(process.argv[out + 1], JSON.stringify(result, null, 2));
