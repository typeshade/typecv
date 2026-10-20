// Measures the dense cloud cleanup (docs/dense-mvs.md, 3.12) on a dense PLY that the lab exported
// ("Export dense PLY", binary, x y z [nx ny nz] r g b). Runs the CPU reference in Node and the
// GPU kernels on headless WebGPU (SwiftShader without a physical GPU) and prints one JSON record
// with the point count after each stage and the time of each stage.
// Usage: node scripts/measure-cloud-clean.mjs path/to/dense.ply
import { readFileSync } from 'node:fs';
import { cleanDenseCloud } from '../src/reconstruction/cloud-clean.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const file = process.argv[2];
// An optional point limit takes the first N points, to time how the GPU path scales.
const limit = Number(process.argv[3] ?? Infinity);
if (!file) throw new Error('Usage: node scripts/measure-cloud-clean.mjs dense.ply');

/** The cloud of a binary little-endian PLY of the lab (27 B or 15 B per point). */
function readPly(bytes) {
  const text = new TextDecoder().decode(bytes.subarray(0, 1024));
  const end = text.indexOf('end_header\n') + 'end_header\n'.length;
  const count = Number(/element vertex (\d+)/.exec(text)[1]);
  const normals = text.includes('property float nx');
  const stride = normals ? 27 : 15;
  const view = new DataView(bytes.buffer, bytes.byteOffset + end, count * stride);
  const positions = new Float32Array(3 * count),
    colors = new Uint8Array(3 * count),
    normalsOut = new Int16Array(2 * count);
  for (let i = 0; i < count; i++) {
    const at = i * stride;
    for (let a = 0; a < 3; a++) positions[3 * i + a] = view.getFloat32(at + 4 * a, true);
    const c = at + (normals ? 24 : 12);
    for (let a = 0; a < 3; a++) colors[3 * i + a] = view.getUint8(c + a);
  }
  return { count, positions, colors, normals: normalsOut };
}

const bytes = readFileSync(file);
const full = readPly(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.length));
const n = Math.min(full.count, limit);
const cloud = {
  count: n,
  positions: full.positions.slice(0, 3 * n),
  colors: full.colors.slice(0, 3 * n),
  normals: full.normals.slice(0, 2 * n),
};
const cpuStarted = performance.now();
const cpu = await cleanDenseCloud(cloud);
const cpuMs = performance.now() - cpuStarted;

const { result, errors } = await inBrowser(
  async (limit) => {
    const { ShadeDevice, gpuCleanPrimitives, cleanCloudWith } =
      await import('/dist/internal/gpu.js');
    const response = await fetch('/dense.ply');
    const bytes = new Uint8Array(await response.arrayBuffer());
    const text = new TextDecoder().decode(bytes.subarray(0, 1024));
    const end = text.indexOf('end_header\n') + 'end_header\n'.length;
    const count = Math.min(limit, Number(/element vertex (\d+)/.exec(text)[1]));
    const total = Number(/element vertex (\d+)/.exec(text)[1]);
    const stride = text.includes('property float nx') ? 27 : 15;
    const view = new DataView(bytes.buffer, end, total * stride);
    const positions = new Float32Array(3 * count),
      colors = new Uint8Array(3 * count);
    for (let i = 0; i < count; i++) {
      for (let a = 0; a < 3; a++) positions[3 * i + a] = view.getFloat32(i * stride + 4 * a, true);
      const c = i * stride + (stride === 27 ? 24 : 12);
      for (let a = 0; a < 3; a++) colors[3 * i + a] = view.getUint8(c + a);
    }
    const input = { count, positions, colors, normals: new Int16Array(2 * count) };
    const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
    try {
      const primitives = gpuCleanPrimitives(device);
      // A first run compiles the shaders (the cold run); the second is warm.
      const cold = await cleanCloudWith(primitives, input);
      const started = performance.now();
      const warm = await cleanCloudWith(primitives, input);
      return {
        coldReport: cold.report,
        warmReport: warm.report,
        warmMs: performance.now() - started,
        liveBytes: device.statistics.liveBytes,
      };
    } finally {
      await device.dispose();
    }
  },
  Number.isFinite(limit) ? limit : 1e12,
  { timeoutMs: 1800000, serve: { '/dense.ply': file } },
);
if (errors.length) throw new Error(errors.join('\n'));
console.log(
  JSON.stringify(
    {
      date: new Date().toISOString().slice(0, 10),
      points: cloud.count,
      cpu: { totalMs: Math.round(cpuMs), report: cpu.report },
      webgpu: result,
    },
    null,
    1,
  ),
);
