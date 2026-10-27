// Measure the shared-focal estimate of the incremental mapper on the lab's calibrated samples,
// treated as uncalibrated: every view starts at fx = fy = ratio × width. See docs/auto-focal.md.
import { readFileSync } from 'node:fs';
import { reconstructIncremental } from '../src/index.ts';
import { LAB_MAPPER_NUMBERS } from '../src/sfm/config/lab-preset.ts';

const SAMPLES = ['south-building', 'gerrard-hall', 'person-hall', 'graham-hall'];

/** The sample's frames as mapper input: COLMAP's SIFT keypoints and raw matches. */
export function loadSample(name) {
  const dir = new URL(`../tests/fixtures/lab-images/${name}/`, import.meta.url);
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

/**
 * The pyramid sample's frames as mapper input: SIFT keypoints and mutual ratio-0.8 matches of
 * every pair, from the CPU tier (`tests/fixtures/pyramid-sift-cpu.json`). The focal is unknown.
 */
export function loadPyramid() {
  const d = JSON.parse(
    readFileSync(new URL('../tests/fixtures/pyramid-sift-cpu.json', import.meta.url), 'utf8'),
  );
  const truth = { fx: NaN, cx: (d.width - 1) / 2, cy: (d.height - 1) / 2, k1: 0, k2: 0 };
  return {
    frames: d.frames.map((f) => ({
      id: f.id,
      width: d.width,
      height: d.height,
      truth,
      keypoints: Float64Array.from(f.keypoints),
    })),
    pairs: d.pairs.map((p) => ({
      first: p.first,
      second: p.second,
      matches: Uint32Array.from(p.matches),
    })),
  };
}

/**
 * The pyramid's relief in the last frame: the median depth of the points seen more than 220 px
 * from the apex (320, 170) over the points seen within 90 px of it, minus 1. Above 0 when the apex is
 * nearer the camera than the ground, as it is.
 */
export function pyramidRelief(reconstruction, viewId = 7) {
  const view = reconstruction.views.find((v) => v.id === viewId);
  if (!view) return NaN;
  const { R, t } = view.pose;
  const apex = [],
    ground = [];
  for (const point of reconstruction.landmarks) {
    const o = point.observations.find((x) => x.viewId === viewId);
    if (!o) continue;
    const [X, Y, Z] = point.xyz;
    const depth = R[6] * X + R[7] * Y + R[8] * Z + t[2];
    const d = Math.hypot(o.pixel[0] - 320, o.pixel[1] - 170);
    if (d < 90) apex.push(depth);
    else if (d > 220) ground.push(depth);
  }
  if (!apex.length || !ground.length) return NaN;
  return median(ground) / median(apex) - 1;
}

/** The lab's numerical controls: the six mapper numbers of `src/sfm/config/lab-preset.ts`. */
export const LAB = { ...LAB_MAPPER_NUMBERS };

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

function median(v) {
  const s = [...v].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

if (import.meta.main) {
  const args = Object.fromEntries(
    process.argv.slice(2).map((a) => a.replace(/^--/, '').split('=')),
  );
  const starts = (args.starts ?? '0.5,0.8,1.2').split(',').map(Number);
  const distortion = args.distortion !== '0';
  const finalIterations = Number(args.final ?? LAB.finalBundleIterations);
  const search =
    args.search === '1' ? (args.views ? { views: Number(args.views) } : true) : undefined;
  const rows = [];
  for (const name of (args.samples ?? SAMPLES.join(',')).split(',')) {
    const { frames, pairs } = name === 'pyramid' ? loadPyramid() : loadSample(name);
    const truth = median(frames.map((f) => f.truth.fx / f.width));
    for (const start of starts) {
      const t0 = performance.now();
      let row;
      try {
        const result = reconstructIncremental(camerasAt(frames, start, distortion), pairs, {
          ...LAB,
          finalBundleIterations: finalIterations,
          refineSharedFocal: true,
          focalSearch: search,
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
          ...(result.diagnostics.focalSearch
            ? {
                grid: +result.diagnostics.focalSearch.gridRatio.toFixed(3),
                subset: +result.diagnostics.focalSearch.ratio.toFixed(3),
                subsetViews: result.diagnostics.focalSearch.views.length,
              }
            : {}),
          ...(name === 'pyramid'
            ? { relief: +pyramidRelief(result.reconstruction).toFixed(3) }
            : {}),
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
