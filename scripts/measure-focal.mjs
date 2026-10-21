// Measure the shared-focal estimate of the incremental mapper on the lab's calibrated samples,
// treated as uncalibrated: every view starts at fx = fy = ratio × width. See docs/auto-focal.md.
import { readFileSync } from 'node:fs';
import { reconstructIncremental } from '../src/index.ts';

const SAMPLES = ['south-building', 'gerrard-hall', 'person-hall', 'graham-hall'];

/** The sample's frames as mapper input: COLMAP's SIFT keypoints and raw matches. */
export function loadSample(name) {
  const dir = new URL(`../website/public/sample/${name}/`, import.meta.url);
  const sample = JSON.parse(readFileSync(new URL('sample.json', dir), 'utf8'));
  const bytes = readFileSync(new URL('colmap-sift.bin', dir));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const keypoints = new Float32Array(buffer, 0, sample.features.keypointBytes / 4);
  const matches = new Uint16Array(buffer, sample.features.keypointBytes);
  const blocks = new Map(sample.features.keypoints.map((b) => [b.id, b]));
  const frames = sample.frames.map((frame) => {
    const block = blocks.get(frame.id);
    return {
      id: frame.id,
      width: frame.width,
      height: frame.height,
      truth: frame.camera,
      keypoints: Float64Array.from(
        keypoints.subarray(block.offset, block.offset + 2 * block.count),
      ),
    };
  });
  const pairs = sample.features.pairs.map((p) => ({
    first: p.first,
    second: p.second,
    matches: Uint32Array.from(matches.subarray(p.offset, p.offset + 2 * p.count)),
  }));
  return { frames, pairs };
}

/** The lab's numerical controls (`LAB_OPTIONS` in `website/src/workers/sfm.worker.ts`). */
export const LAB = {
  ransacIterations: 500,
  maxErrorPx: 3,
  minInitialAngleDeg: 1,
  minTriangulationAngleDeg: 0.5,
  bundleIterations: 4,
  finalBundleIterations: 8,
};

export function camerasAt(frames, ratio, distortion) {
  return frames.map((f) => ({
    id: f.id,
    keypoints: f.keypoints,
    camera: {
      fx: ratio * f.width,
      fy: ratio * f.width,
      cx: f.truth.cx,
      cy: f.truth.cy,
      k1: distortion ? f.truth.k1 : 0,
      k2: distortion ? f.truth.k2 : 0,
    },
  }));
}

const median = (v) => {
  const s = [...v].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

if (import.meta.main) {
  const args = Object.fromEntries(
    process.argv.slice(2).map((a) => a.replace(/^--/, '').split('=')),
  );
  const starts = (args.starts ?? '0.5,0.8,1.2').split(',').map(Number);
  const distortion = args.distortion !== '0';
  const finalIterations = Number(args.final ?? LAB.finalBundleIterations);
  const rows = [];
  for (const name of (args.samples ?? SAMPLES.join(',')).split(',')) {
    const { frames, pairs } = loadSample(name);
    const truth = median(frames.map((f) => f.truth.fx / f.width));
    for (const start of starts) {
      const t0 = performance.now();
      let row;
      try {
        const result = reconstructIncremental(camerasAt(frames, start, distortion), pairs, {
          ...LAB,
          finalBundleIterations: finalIterations,
          refineSharedFocal: true,
        });
        const width = new Map(frames.map((f) => [f.id, f.width]));
        const ratio = median(result.reconstruction.views.map((v) => v.camera.fx / width.get(v.id)));
        const last = result.diagnostics.bundleAdjustments.at(-1);
        row = {
          sample: name,
          truth: +truth.toFixed(3),
          start,
          final: +ratio.toFixed(3),
          errorPct: +((100 * (ratio - truth)) / truth).toFixed(1),
          views: result.reconstruction.views.length,
          points: result.reconstruction.landmarks.length,
          rmsePx: +last.finalRmsePx.toFixed(3),
          accepted: last.acceptedSteps,
          rejected: last.rejectedSteps,
          ms: Math.round(performance.now() - t0),
        };
      } catch (error) {
        row = { sample: name, truth, start, error: String(error.message ?? error) };
      }
      rows.push(row);
      console.log(JSON.stringify(row));
    }
  }
}
