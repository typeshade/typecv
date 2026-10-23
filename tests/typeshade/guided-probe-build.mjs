import { resolve } from 'node:path';
import { buildComputeProbe } from './compute-probe-build.mjs';

export const guidedProbeFile = resolve(
  import.meta.dirname,
  '../../.typecv-build/guided-probe/guided-probe.js',
);
export const buildGuidedProbe = () => buildComputeProbe('guided-probe');
