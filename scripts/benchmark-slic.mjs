// SLIC superpixels on one sample frame: wall-clock per stage for the CPU reference, the CPU tier
// of the device and the WebGPU tier (SwiftShader unless a hardware GPU is present). Usage:
//   node scripts/benchmark-slic.mjs --sample=south-building --frame=0 [--edge=640] [--repeats=3]
// The stages are differences of whole calls: the call with no iteration (upload, Lab, seeds, one
// assignment, readback), the slope of the iterations, and the connectivity pass.
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
  repeats = Number(arg('repeats', '3')),
  output = resolve(arg('output', `work/benchmark-results/slic-${sample}.json`));
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
  async ({ encoded, edge, repeats }) => {
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
      return { ms: median(samples), last };
    };
    const runners = {
      cpuReference: (o) => () => api.slic(pixels, width, height, o),
      cpuTier: (o) => () => api.slicGpu(devices.cpuTier, pixels, width, height, o),
      webgpu: (o) => () => api.slicGpu(devices.webgpu, pixels, width, height, o),
    };
    const rows = {};
    try {
      for (const [name, make] of Object.entries(runners)) {
        const noConn = { enforceConnectivity: false },
          zero = await time(make({ ...noConn, maxIterations: 0 })),
          ten = await time(make({ ...noConn, maxIterations: 10 })),
          full = await time(make({})),
          fullLabels = full.last instanceof Promise ? await full.last : full.last;
        rows[name] = {
          setupMs: zero.ms,
          perIterationMs: (ten.ms - zero.ms) / 10,
          tenIterationsMs: ten.ms,
          connectivityMs: full.ms - ten.ms,
          totalMs: full.ms,
          segments: fullLabels.count,
        };
      }
      const a = await runners.cpuReference({})(),
        b = await runners.webgpu({})();
      let same = 0;
      for (let i = 0; i < a.labels.length; i++) if (a.labels[i] === b.labels[i]) same++;
      return {
        width,
        height,
        rows,
        agreementCpuReferenceVsWebgpu: same / a.labels.length,
        adapter: { ...adapter.info },
      };
    } finally {
      await devices.cpuTier.dispose();
      await devices.webgpu.dispose();
    }
  },
  { encoded, edge, repeats },
  { timeoutMs: 900000 },
);
if (errors.length) console.error(errors);
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
