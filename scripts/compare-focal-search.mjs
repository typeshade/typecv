// Compare the automatic schedule with the previous exhaustive grid on identical inputs.
// node scripts/compare-focal-search.mjs --samples=pyramid,south-building --distortion=0
// Numerical controls are identical; only the number/order of evaluated grid points changes.
import { reconstructIncremental } from '../src/index.ts';
import assert from 'node:assert/strict';
import {
  applyFocal,
  focalSearchControls,
  focalSearchSteps,
} from '../src/reconstruction/focal-search.ts';
import { executeReconstructionJob } from '../src/reconstruction/mapper.ts';
import { LAB, camerasAt, loadPyramid, loadSample, pyramidRelief } from './measure-focal.mjs';

const args = Object.fromEntries(process.argv.slice(2).map((a) => a.replace(/^--/, '').split('=')));
const names = (args.samples ?? 'pyramid,south-building,gerrard-hall,person-hall,graham-hall').split(
  ',',
);
for (const name of names) {
  const { frames, pairs } = name === 'pyramid' ? loadPyramid() : loadSample(name);
  const images = camerasAt(frames, 0.8, args.distortion === '1');
  const options = { ...LAB, refineSharedFocal: true };
  let reference;
  for (const exhaustive of [true, false]) {
    const start = performance.now(),
      jobs = {},
      machine = focalSearchSteps(images, pairs, options, focalSearchControls(true), exhaustive);
    let step = machine.next();
    while (!step.done) {
      jobs[step.value.kind] = (jobs[step.value.kind] ?? 0) + 1;
      let value;
      try {
        value = executeReconstructionJob(step.value);
      } catch (error) {
        step = machine.throw(error);
        continue;
      }
      step = machine.next(value);
    }
    const found = step.value;
    const result = reconstructIncremental(applyFocal(images, found.ratio), pairs, options);
    const row = {
      sample: name,
      mode: exhaustive ? 'exhaustive' : 'adaptive',
      elapsedMs: Math.round(performance.now() - start),
      jobs,
      gridRatio: found.gridRatio,
      ratio: found.ratio,
      finalRatio:
        result.reconstruction.views[0].camera.fx /
        (2 * result.reconstruction.views[0].camera.cx + 1),
      views: result.reconstruction.views.length,
      points: result.reconstruction.landmarks.length,
      observations: result.reconstruction.landmarks.reduce((n, p) => n + p.observations.length, 0),
      rmsePx: result.diagnostics.bundleAdjustments.at(-1).finalRmsePx,
      ...(name === 'pyramid' ? { relief: pyramidRelief(result.reconstruction) } : {}),
      candidates: found.candidates,
    };
    console.log(JSON.stringify(row));
    if (exhaustive) reference = row;
    else {
      for (const key of ['views', 'points', 'observations'])
        assert.equal(row[key], reference[key], `${name}: ${key}`);
      for (const key of ['ratio', 'finalRatio', 'rmsePx'])
        assert.ok(Math.abs(row[key] - reference[key]) < 1e-6, `${name}: ${key}`);
    }
  }
}
