// Fixed-input numerical diagnosis. This compares arithmetic, not inferred depth against truth.
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { planMvsViews } from '../dist/internal/gpu.js';
const args = process.argv.slice(2),
  flag = (n) => (args.includes(n) ? args[args.indexOf(n) + 1] : undefined);
if (!flag('--case') || !flag('--out'))
  throw new Error('Usage: --case <prepared-native-case> --out <report.json> [--source shader.ts]');
const dir = resolve(flag('--case')),
  info = JSON.parse(await readFile(dir + '/case.json', 'utf8')),
  scene = JSON.parse(await readFile(dir + '/scene.json', 'utf8')),
  plan = planMvsViews(scene),
  ref = plan.views.find((v) => v.id === info.reference).index;
const source = await readFile(
  flag('--source') ?? new URL('../src/gpu/mvs.shade.ts', import.meta.url),
  'utf8',
);
const b = await build({
  stdin: {
    contents: "export {compile,compileModuleJs} from 'typeshade';",
    resolveDir: resolve(import.meta.dirname, '..'),
  },
  bundle: true,
  platform: 'node',
  format: 'cjs',
  write: false,
  external: ['typescript'],
});
const c = { exports: {} };
new Function('require', 'module', 'exports', b.outputFiles[0].text)(
  createRequire(import.meta.url),
  c,
  c.exports,
);
const parsed = c.exports.compile(source);
if (parsed.diagnostics.some((d) => d.category === 'error'))
  throw new Error(JSON.stringify(parsed.diagnostics));
const W = info.width,
  H = info.height,
  P = W * H,
  gray = new Float32Array(plan.packed.length * P);
for (let v = 0; v < plan.packed.length; v++) {
  const id = scene.views[plan.packed[v]].id,
    a = new Uint8Array(await readFile(dir + `/${id}.rgba`));
  for (let i = 0; i < P; i++)
    gray[v * P + i] = (0.299 * a[4 * i] + 0.587 * a[4 * i + 1] + 0.114 * a[4 * i + 2]) / 255;
}
const control = {
  width: W,
  height: H,
  reference: ref,
  relativePoseCache: 0,
  windowRadius: 5,
  windowStep: 2,
  axisSamples: 5,
  minSamples: 7,
  bilateral: 1,
  boundaryAware: 1,
  spatialFalloff: 1 / 50,
  colorFalloff: 50,
  minAngleCos: Math.cos(Math.PI / 180),
};
const modules = ['f64', 'f32'].map((precision) => {
  const m = c.exports.compileModuleJs(parsed.module, { precision });
  m.setBinding('mvsControl', control);
  m.setBinding('mvsCameras', plan.cameras);
  m.setBinding('mvsGray', gray);
  return m;
});
const out = {
  input: info,
  sourceSha256: createHash('sha256').update(source).digest('hex'),
  protocol:
    'Identical f32 gray/cameras in f64/f32 CPU generators. Fronto-parallel hypothetical model-Z=3; not ground-truth depth or full GPU reproduction.',
  regions: {},
};
for (const name of ['metal', 'skyNearPole']) {
  const mask = new Uint8Array(await readFile(dir + `/mask-${name}.u8`)),
    rows = [],
    masses = [];
  for (let i = 0; i < P; i++) {
    if (!mask[i] || i % 4 !== 0) continue;
    const x = i % W,
      y = Math.floor(i / W),
      stats = modules.map((m) => m.fns.referenceWindow(x, y));
    if (stats.some((s) => !s.ok)) continue;
    let spatial = 0;
    for (const dx of [-4, -2, 0, 2, 4])
      for (const dy of [-4, -2, 0, 2, 4])
        if (x + dx >= 0 && x + dx < W && y + dy >= 0 && y + dy < H)
          spatial += Math.exp(-(dx * dx + dy * dy) / 50);
    masses.push(stats[0].weight / spatial);
    for (const s of plan.views.find((v) => v.id === info.reference).sources) {
      const costs = modules.map((m, k) =>
        m.fns.sourceCost(
          s,
          x,
          y,
          m.fns.referenceRay(x, y).map((v) => (v * 3) / plan.normalization.scale),
          [0, 0, -1],
          stats[k],
        ),
      );
      if (costs.every(Number.isFinite))
        rows.push({
          x,
          y,
          source: scene.views[plan.packed[s]].id,
          f64: costs[0],
          f32: costs[1],
          error: Math.abs(costs[0] - costs[1]),
        });
    }
  }
  const q = (a, p) => {
    a.sort((x, y) => x - y);
    return a[Math.floor(p * (a.length - 1))] ?? null;
  };
  out.regions[name] = {
    probes: rows.length,
    referenceSupportMassMedian: q(masses, 0.5),
    medianCostError: q(
      rows.map((r) => r.error),
      0.5,
    ),
    p95CostError: q(
      rows.map((r) => r.error),
      0.95,
    ),
    costErrorsOver001: rows.filter((r) => r.error > 0.01).length,
    costErrorsOver01: rows.filter((r) => r.error > 0.1).length,
    validityDisagreements: rows.filter((r) => r.f64 < 2 !== r.f32 < 2).length,
    worst: rows.sort((a, b) => b.error - a.error).slice(0, 6),
  };
}
await writeFile(resolve(flag('--out')), JSON.stringify(out, null, 2) + '\n');
console.log(JSON.stringify(out.regions));
