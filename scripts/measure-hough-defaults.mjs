// Measures Canny and Hough settings on the first frame of the sample photographs, on the CPU
// reference. Usage:
//   node scripts/measure-hough-defaults.mjs [--edge=640] [--samples=a,b] [--options='[{"label":"x","options":{}}]']
// Per sample and option set it prints the edge pixels, the lines, the segments, the share of
// segments longer than 60 px, the lines with another line within 2 degrees ("parallel") and the
// lines with another line within 2 degrees and 18 px of rho ("duplicate").
import sharp from 'sharp';
import { houghLines } from '../dist/gpu.js';

const arg = (name, fallback) => {
  const prefix = `--${name}=`;
  const value = process.argv.slice(2).find((item) => item.startsWith(prefix));
  return value ? value.slice(prefix.length) : fallback;
};
const edge = Number(arg('edge', '640')),
  samples = arg('samples', 'south-building,gerrard-hall,person-hall,graham-hall,pyramid').split(
    ',',
  ),
  sets = JSON.parse(arg('options', '[{"label":"default","options":{}}]'));

const load = async (sample) => {
  const url = new URL(`../website/public/sample/${sample}/frame-00.jpg`, import.meta.url);
  const { data, info } = await sharp(url.pathname)
    .resize({ width: edge, height: edge, fit: 'inside', withoutEnlargement: true })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { rgba: new Uint8Array(data), width: info.width, height: info.height };
};

const angleGap = (a, b) => {
  const d = Math.abs(a - b) % Math.PI;
  return Math.min(d, Math.PI - d) * (180 / Math.PI);
};

export function summarize(result) {
  const { lines, segments } = result,
    long = segments.filter((s) => Math.hypot(s.x2 - s.x1, s.y2 - s.y1) > 60).length;
  let parallel = 0,
    duplicate = 0;
  lines.forEach((a, i) => {
    let p = false,
      d = false;
    lines.forEach((b, j) => {
      if (i === j || angleGap(a.theta, b.theta) >= 2) return;
      p = true;
      // The same rho means the same line only when theta is near, or near pi with rho negated.
      const rho = Math.abs(a.theta - b.theta) < Math.PI / 2 ? b.rho : -b.rho;
      if (Math.abs(a.rho - rho) <= 18) d = true;
    });
    if (p) parallel++;
    if (d) duplicate++;
  });
  return {
    edges: result.count,
    lines: lines.length,
    segments: segments.length,
    longShare: segments.length ? long / segments.length : 0,
    parallel,
    duplicate,
  };
}

const rows = [];
for (const sample of samples) {
  const image = await load(sample);
  for (const { label, options } of sets) {
    // `divisor` sets the vote threshold to ceil(min(width, height) / divisor).
    const { divisor, ...rest } = options,
      run = divisor
        ? { ...rest, threshold: Math.ceil(Math.min(image.width, image.height) / divisor) }
        : rest,
      result = houghLines(image.rgba, image.width, image.height, run);
    rows.push({
      sample,
      label,
      size: `${image.width}x${image.height}`,
      low: +result.low.toFixed(4),
      high: +result.high.toFixed(4),
      votes: run.threshold ?? Math.ceil(Math.min(image.width, image.height) / 6),
      ...summarize(result),
    });
  }
}
console.log('| sample | settings | edges | lines | segments | >60 px | parallel | duplicate |');
console.log('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |');
for (const r of rows)
  console.log(
    `| ${r.sample} | ${r.label} (votes ${r.votes}) | ${r.edges} | ${r.lines} | ${r.segments} | ${(100 * r.longShare).toFixed(0)} % | ${r.parallel} | ${r.duplicate} |`,
  );
