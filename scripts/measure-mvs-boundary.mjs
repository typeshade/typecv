// Fixed-pose replay of previously exported analysis RGBA. Dataset paths are external; no user
// pixels or scene geometry are committed. Compare each option independently before lab adoption.
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { inBrowser } from '../tests/typeshade/browser.mjs';
const args = process.argv.slice(2),
  flag = (key, other) => (args.includes(key) ? args[args.indexOf(key) + 1] : other);
if (!flag('--cases') || !flag('--baseline') || !flag('--out'))
  throw new Error(
    'Usage: --cases <prepared-directory> --baseline <baseline-bundle.mjs> --out <directory>',
  );
const root = resolve(flag('--cases'));
const baseline = resolve(flag('--baseline'));
const output = resolve(flag('--out'));
const modes = flag('--modes', 'legacy,boundary,observable,combined').split(',');
const configurations = {
  legacy: {},
  boundary: { boundaryAware: true },
  observable: { minFiniteGain: 0.02 },
  combined: { boundaryAware: true, minFiniteGain: 0.02 },
};
await mkdir(output, { recursive: true });
const cases = JSON.parse(await readFile(join(root, 'cases.json'), 'utf8'));
const results = [];
for (const item of cases) {
  if (flag('--case') && item.name !== flag('--case')) continue;
  const dir = join(root, item.name),
    info = JSON.parse(await readFile(join(dir, 'case.json'), 'utf8'));
  const scene = JSON.parse(await readFile(join(dir, 'scene.json'), 'utf8'));
  const serve = Object.fromEntries(
    scene.views.map((v) => [`/input-${v.id}.rgba`, join(dir, `${v.id}.rgba`)]),
  );
  serve['/baseline.mjs'] = baseline;
  for (const name of info.masks) serve[`/mask-${name}.u8`] = join(dir, `mask-${name}.u8`);
  for (const mode of modes) {
    if (!configurations[mode]) throw new Error('Unknown mode ' + mode);
    const { result, errors } = await inBrowser(
      async ({ info, scene, mode, options, iterations }) => {
        const api = await import(mode === 'legacy' ? '/baseline.mjs' : '/dist/internal/gpu.js');
        const images = new Map(
          await Promise.all(
            scene.views.map(async (v) => [
              v.id,
              {
                width: info.width,
                height: info.height,
                rgba: new Uint8Array(await (await fetch(`/input-${v.id}.rgba`)).arrayBuffer()),
              },
            ]),
          ),
        );
        const masks = Object.fromEntries(
          await Promise.all(
            info.masks.map(async (name) => [
              name,
              new Uint8Array(await (await fetch(`/mask-${name}.u8`)).arrayBuffer()),
            ]),
          ),
        );
        const plan = api.planMvsViews(scene),
          entry = plan.views.find((v) => v.id === info.reference),
          reference = entry.index;
        const device = await api.ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
        try {
          const started = performance.now();
          const result = await api.mvsDepthMapsGpu(device, scene, plan, images, {
            photometricIterations: iterations,
            geometricIterations: 4,
            readStages: true,
            geometricOrder: [reference],
            ...options,
          });
          const ms = performance.now() - started,
            map = result.views[reference];
          const filename = `${info.name}-${mode}`;
          await fetch('/upload/' + filename + '-depth.f32le', { method: 'PUT', body: map.depth });
          await fetch('/upload/' + filename + '-normal.u32le', { method: 'PUT', body: map.normal });
          const regions = {};
          for (const [name, mask] of Object.entries(masks)) {
            let pixels = 0,
              kept = 0;
            for (let i = 0; i < mask.length; i++)
              if (mask[i]) {
                pixels++;
                if (map.depth[i] > 0) kept++;
              }
            regions[name] = { pixels, kept, share: kept / pixels };
          }
          return {
            case: info.name,
            mode,
            options,
            ms,
            regions,
            depthScaleToModel: plan.normalization.scale,
            referenceSources: entry.sources.map((i) => scene.views[plan.packed[i]].id),
            valid: map.depth.filter((d) => d > 0).length,
            device: device.info,
            liveBytes: device.statistics.liveBytes,
            photoIterations: iterations,
            geometricIterations: 4,
          };
        } finally {
          await device.dispose();
        }
      },
      {
        info,
        scene,
        mode,
        options: configurations[mode],
        iterations: Number(flag('--iterations', '4')),
      },
      { serve, uploads: output, timeoutMs: 900000 },
    );
    if (errors.length) throw new Error(errors.join('\n'));
    results.push(result);
    console.log(JSON.stringify(result));
    await writeFile(
      join(output, `${item.name}-${mode}.json`),
      JSON.stringify({ input: info, result }, null, 2),
    );
    await writeFile(join(output, 'results.json'), JSON.stringify(results, null, 2));
  }
}
