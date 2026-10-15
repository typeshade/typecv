/** Run TypeCV (not OpenCV) on prepared video correspondences. */
import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { reconstructTwoView } from 'typecv';

const [input, output] = process.argv.slice(2);
if (!input || !output)
  throw new Error('Usage: node scripts/evaluate-video.mjs matches.json output-directory');
const prepared = JSON.parse(fs.readFileSync(input, 'utf8'));
fs.mkdirSync(output, { recursive: true });
const assumptions = {
  focalLengthWidthRatios: [0.8, 1.2, 1.6],
  principalPoint: 'image center',
  distortion: 'assumed zero; not calibrated',
  thresholdPxApprox: 1.5,
  iterations: 1000,
  refinementIterations: 10,
  seed: 42,
};
const results = [];
for (const pair of prepared.pairs) {
  for (const ratio of assumptions.focalLengthWidthRatios) {
    const focal = prepared.width * ratio;
    const K = { fx: focal, fy: focal, cx: prepared.width / 2, cy: prepared.height / 2 };
    const entry = {
      pair: pair.id,
      seconds: pair.seconds,
      focalWidthRatio: ratio,
      inputMatches: pair.matches.length,
      homographyInlierFraction: pair.homographyInlierFraction,
    };
    const started = performance.now();
    try {
      const result = reconstructTwoView(Float64Array.from(pair.matches.flat()), K, K, {
        iterations: assumptions.iterations,
        seed: assumptions.seed,
        threshold: 1.5 / focal,
        refinementIterations: assumptions.refinementIterations,
      });
      Object.assign(entry, {
        status: 'reconstructed',
        inlierCount: result.inlierCount,
        acceptedPoints: result.positiveDepthCount,
        meanReprojectionErrorPx: result.meanReprojectionErrorPx,
        R: Array.from(result.R),
        t: Array.from(result.t),
        diagnostics: result.diagnostics,
      });
      const xyz = [];
      const acceptedMatches = [];
      for (let i = 0; i < result.validPoints.length; i++)
        if (result.validPoints[i]) {
          xyz.push(Array.from(result.points.subarray(i * 3, i * 3 + 3)).join(' '));
          acceptedMatches.push(pair.matches[i]);
        }
      const stem = `${pair.id}-f${ratio}`;
      fs.writeFileSync(
        path.join(output, `${stem}.ply`),
        `ply\nformat ascii 1.0\nelement vertex ${xyz.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n${xyz.join('\n')}\n`,
      );
      fs.writeFileSync(path.join(output, `${stem}-accepted.json`), JSON.stringify(acceptedMatches));
    } catch (error) {
      Object.assign(entry, { status: 'rejected', reason: error.message });
    }
    entry.elapsedMs = performance.now() - started;
    results.push(entry);
    console.log(JSON.stringify(entry));
    // Persist partial progress so long runs remain inspectable.
    fs.writeFileSync(
      path.join(output, 'results.json'),
      JSON.stringify(
        {
          videoSha256: prepared.videoSha256,
          width: prepared.width,
          height: prepared.height,
          assumptions,
          results,
        },
        null,
        2,
      ) + '\n',
    );
  }
}
