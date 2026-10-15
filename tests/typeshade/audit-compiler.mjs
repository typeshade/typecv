// Build-time only: parse actual TypeShade source, derive an operator, and pack WGSL.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { compile, compileModule, grad, gradCheck, packModule } from 'typeshade';
import { fn, module, externFn, f32T, builtin, storageBuffer, vec4fT } from 'typeshade/core/ir';

async function readModule(name) {
  const source = await readFile(`experimental/typeshade/audit/${name}.shade.ts`, 'utf8');
  const result = compile(source);
  assert.deepEqual(
    result.diagnostics.filter((d) => d.category === 'error'),
    [],
  );
  assert.ok(result.module);
  return result.module;
}

const precision = await readModule('precision'),
  quartic = await readModule('quartic');
const workgroup = await readModule('workgroup'),
  projection = await readModule('projection');
const derivative = grad(projection, 'projection', 'z');
const gradient = externFn(derivative.name, { x: f32T, z: f32T, focal: f32T, center: f32T }, f32T);
const points = storageBuffer('points', vec4fT, { group: 0, binding: 0, access: 'read' });
const outputs = storageBuffer('outputs', f32T, { group: 0, binding: 1, access: 'read_write' });
const entry = fn(
  'derivative',
  { gid: builtin('global_invocation_id') },
  ({ gid }) => {
    const p = points.at(gid.x);
    outputs.at(gid.x).assign(gradient({ x: p.x, z: p.y, focal: p.z, center: p.w }));
  },
  { stage: 'compute', workgroupSize: 4 },
);
const extra = module({ uses: [points, outputs], funcs: [entry] });
const differentiated = {
  ...derivative.module,
  bindings: [...derivative.module.bindings, ...extra.bindings],
  funcs: [...derivative.module.funcs, ...extra.funcs],
};
const at = [
  [0.5, 2, 900, 320],
  [-1, 4, 700, 480],
  [1.5, 8, 1200, 640],
  [0.25, 0.5, 400, 0],
];
const forward = gradCheck(projection, 'projection', { wrt: 'z', at });
const reverse = gradCheck(projection, 'projection', {
  wrt: ['x', 'z', 'focal', 'center'],
  at,
  mode: 'reverse',
});
assert.equal(forward.ok, true);
assert.equal(reverse.ok, true);
const oracle = compileModule(derivative.module);
const cpuDerivative = at.map((args) => oracle.fns[derivative.name](...args));

// A broken input must be visible to the same compiler used for all passing probes.
const broken = compile(
  '"use typeshade"; export function wrong(x: f32): f32 { return missing(x); }',
);
assert.ok(broken.diagnostics.some((d) => d.category === 'error'));
const flavors = ['float', 'integer'];
const packs = {};
for (const flavor of flavors) {
  packs[`precision_${flavor}`] = packModule(precision, { emit: { fp64Flavor: flavor } });
  packs[`quartic_${flavor}`] = packModule(quartic, { emit: { fp64Flavor: flavor } });
}
packs.workgroup = packModule(workgroup);
packs.derivative = packModule(differentiated);
await writeFile(
  process.argv[2],
  JSON.stringify({
    packs,
    at,
    forward,
    reverse,
    cpuDerivative,
    guards: Object.fromEntries(
      Object.entries(packs).map(([name, p]) => [name, p.bindings.some((b) => b.name === '_fp64')]),
    ),
  }),
);
