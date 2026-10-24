import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { buildComputeProbe } from './compute-probe-build.mjs';
import { assertContinuedObservations } from './continued-observations-assertions.mjs';

test('continuation projects only appended measurements while preserving complete guarded depth and validation on CPU', async () => {
  const file = await buildComputeProbe('continued-observations-probe');
  const { checkContinuedObservations } = await import(pathToFileURL(file).href);
  assertContinuedObservations(await checkContinuedObservations());
});
