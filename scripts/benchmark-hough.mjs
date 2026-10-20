// Canny edges and Hough lines on one sample frame: wall-clock per stage for the CPU reference, the
// CPU tier of the device and the WebGPU tier (SwiftShader unless a hardware GPU is present). Usage:
//   node scripts/benchmark-hough.mjs --sample=south-building --frame=0 [--edge=640] [--repeats=7] [--options='{"lowThreshold":0.04}']
// Each time is the best of `repeats` runs after one warm run (the container is shared). The stages are differences of whole calls: Canny alone (with each hysteresis mode on the device),
// Canny plus vote and peaks (`segments: false`), and the full call with segments.
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const arg = (name, fallback) => {
  const prefix = `--${name}=`;
  const value = process.argv.slice(2).find((item) => item.startsWith(prefix));
  return value ? value.slice(prefix.length) : fallback;
};
const sample = arg('sample', 'south-building'),
  frame = Number(arg('frame', '0')),
  edge = Number(arg('edge', '640')),
  repeats = Number(arg('repeats', '7')),
  options = JSON.parse(arg('options', '{}')),
  output = resolve(arg('output', `work/benchmark-results/hough-${sample}.json`));
if (!/^[a-z0-9-]+$/.test(sample)) throw new RangeError('--sample is a directory name');
const encoded = (
  await readFile(
    new URL(
      `../website/public/sample/${sample}/frame-${String(frame).padStart(2, '0')}.jpg`,
      import.meta.url,
    ),
  )
).toString('base64');

const { result, errors } = await inBrowser(
  async ({ encoded, edge, repeats, options }) => {
    const api = await import('/dist/gpu.js');
    const bytes = Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
    const scale = Math.min(1, edge / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const { width, height } = canvas,
      pixels = context.getImageData(0, 0, width, height).data.slice();
    const adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice(),
      devices = {
        cpuTier: await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 }),
        webgpu: await api.ShadeDevice.open({ gpuDevice: gpu, backend: 'webgpu', poolBytes: 0 }),
      };
    const median = (values) => values.slice().sort((a, b) => a - b)[Math.floor(values.length / 2)];
    const time = async (call) => {
      await call(); // warm: pipelines, pools
      const samples = [];
      let last;
      for (let i = 0; i < repeats; i++) {
        const t0 = performance.now();
        last = await call();
        samples.push(performance.now() - t0);
      }
      return { ms: Math.min(...samples), median: median(samples), last };
    };
    const runners = {
      cpuReference: {
        canny: (o) => () => api.canny(pixels, width, height, o),
        hough: (o) => () => api.houghLines(pixels, width, height, o),
      },
      cpuTier: {
        canny: (o) => () => api.cannyGpu(devices.cpuTier, pixels, width, height, o),
        hough: (o) => () => api.houghLinesGpu(devices.cpuTier, pixels, width, height, o),
      },
      webgpu: {
        canny: (o) => () => api.cannyGpu(devices.webgpu, pixels, width, height, o),
        hough: (o) => () => api.houghLinesGpu(devices.webgpu, pixels, width, height, o),
      },
    };
    const rows = {};
    try {
      for (const [name, make] of Object.entries(runners)) {
        for (const hysteresis of name === 'cpuReference' ? ['reference'] : ['gpu', 'cpu']) {
          const o = hysteresis === 'reference' ? options : { ...options, hysteresis },
            edges = await time(make.canny(o)),
            votes = await time(make.hough({ ...o, segments: false })),
            full = await time(make.hough(o)),
            result = await full.last;
          rows[`${name}${hysteresis === 'reference' ? '' : `:${hysteresis}`}`] = {
            cannyMs: edges.ms,
            voteAndPeaksMs: votes.ms - edges.ms,
            segmentsMs: full.ms - votes.ms,
            totalMs: full.ms,
            edgePixels: result.count,
            lines: result.lines.length,
            segments: result.segments.length,
            hysteresisPasses: result.hysteresisPasses,
          };
        }
      }
      const a = await runners.cpuReference.hough(options)(),
        b = await runners.webgpu.hough(options)();
      let same = 0;
      for (let i = 0; i < a.edges.length; i++) if (a.edges[i] === b.edges[i]) same++;
      return {
        width,
        height,
        rows,
        edgeAgreementCpuReferenceVsWebgpu: same / a.edges.length,
        sameLines: JSON.stringify(a.lines) === JSON.stringify(b.lines),
        accumulatorEqual: a.accumulator.every((v, i) => v === b.accumulator[i]),
        adapter: { ...adapter.info },
      };
    } finally {
      await devices.cpuTier.dispose();
      await devices.webgpu.dispose();
    }
  },
  { encoded, edge, repeats, options },
  { timeoutMs: 900000 },
);
if (errors.length) console.error(errors);
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
