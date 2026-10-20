// Measures the dense mesh (docs/dense-mvs.md, 3.13) on a dense PLY that the lab exported (binary,
// x y z nx ny nz r g b). Cleans the cloud on the CPU tier with the lab's options (minFraction 0.1),
// meshes it with the CPU reference in Node and with the kernels on headless WebGPU (SwiftShader
// without a physical GPU), and prints one JSON record with the counts and the time of each stage.
// Usage: node scripts/measure-dense-mesh.mjs path/to/dense.ply
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { cleanDenseCloud, encodeOctahedral } from '../src/reconstruction/cloud-clean.ts';
import { meshDenseCloud } from '../src/reconstruction/dense-mesh.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const file = process.argv[2];
if (!file) throw new Error('Usage: node scripts/measure-dense-mesh.mjs dense.ply');

function readPly(bytes) {
  const text = new TextDecoder().decode(bytes.subarray(0, 1024));
  const end = text.indexOf('end_header\n') + 'end_header\n'.length;
  const count = Number(/element vertex (\d+)/.exec(text)[1]);
  if (!text.includes('property float nx')) throw new Error('The PLY has no normals');
  const view = new DataView(bytes.buffer, bytes.byteOffset + end, count * 27);
  const positions = new Float32Array(3 * count),
    colors = new Uint8Array(3 * count),
    normals = new Int16Array(2 * count);
  for (let i = 0; i < count; i++) {
    const at = i * 27;
    for (let a = 0; a < 3; a++) {
      positions[3 * i + a] = view.getFloat32(at + 4 * a, true);
      colors[3 * i + a] = view.getUint8(at + 24 + a);
    }
    const [x, y] = encodeOctahedral(
      view.getFloat32(at + 12, true),
      view.getFloat32(at + 16, true),
      view.getFloat32(at + 20, true),
    );
    normals[2 * i] = x;
    normals[2 * i + 1] = y;
  }
  return { count, positions, colors, normals };
}

const bytes = readFileSync(file);
const raw = readPly(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.length));
const clean = await cleanDenseCloud(raw, { cluster: { minFraction: 0.1 } });
const cloud = clean.cloud;
// The cleaned cloud for the browser: count, then positions, colors (padded to 4 B) and normals.
const pad = (3 * cloud.count + 3) & ~3;
const packed = new Uint8Array(4 + 12 * cloud.count + pad + 4 * cloud.count);
new DataView(packed.buffer).setUint32(0, cloud.count, true);
packed.set(new Uint8Array(cloud.positions.buffer), 4);
packed.set(cloud.colors, 4 + 12 * cloud.count);
packed.set(new Uint8Array(cloud.normals.buffer), 4 + 12 * cloud.count + pad);
const directory = mkdtempSync(join(tmpdir(), 'dense-mesh-'));
const cloudFile = join(directory, 'cloud.bin');
writeFileSync(cloudFile, packed);

const cpuRuns = [];
let cpu;
for (let run = 0; run < 3; run++) {
  const started = performance.now();
  cpu = meshDenseCloud(cloud);
  cpuRuns.push(Math.round(performance.now() - started));
}

const { result, errors } = await inBrowser(
  async () => {
    const { ShadeDevice, meshDenseCloudGpu } = await import('/dist/internal/gpu.js');
    const bytes = new Uint8Array(await (await fetch('/cloud.bin')).arrayBuffer());
    const count = new DataView(bytes.buffer).getUint32(0, true);
    const pad = (3 * count + 3) & ~3;
    const cloud = {
      count,
      positions: new Float32Array(bytes.slice(4, 4 + 12 * count).buffer),
      colors: bytes.slice(4 + 12 * count, 4 + 12 * count + 3 * count),
      normals: new Int16Array(
        bytes.slice(4 + 12 * count + pad, 4 + 12 * count + pad + 4 * count).buffer,
      ),
    };
    const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
    try {
      const cold = await meshDenseCloudGpu(device, cloud);
      const started = performance.now();
      const warm = await meshDenseCloudGpu(device, cloud);
      return {
        cold: cold.report,
        warm: warm.report,
        warmMs: performance.now() - started,
        indices: Array.from(warm.mesh.indices.subarray(0, 3000)),
        faces: warm.mesh.faceCount,
        liveBytes: device.statistics.liveBytes,
      };
    } finally {
      await device.dispose();
    }
  },
  undefined,
  { timeoutMs: 1800000, serve: { '/cloud.bin': cloudFile } },
);
if (errors.length) throw new Error(errors.join('\n'));
const sameStart = result.indices.every((v, i) => v === cpu.mesh.indices[i]);
delete result.indices;
console.log(
  JSON.stringify(
    {
      date: new Date().toISOString().slice(0, 10),
      points: cloud.count,
      cleanedFrom: raw.count,
      cpu: { runsMs: cpuRuns, report: cpu.report },
      webgpu: result,
      webgpuFacesEqualCpu: result.faces === cpu.mesh.faceCount,
      webgpuFirstFacesEqualCpu: sameStart,
    },
    null,
    1,
  ),
);
