import { ShadeDevice } from '../../src/gpu/device.ts';
export { pairRefitRowChunk } from '../../src/gpu/pair-estimation.ts';
import {
  appendPairModelRows,
  appendPairModelRowsInPlace,
} from '../../src/gpu/pair-models.shade.ts';

const camera = { fx: 1, fy: 1, cx: 0, cy: 0, k1: 0, k2: 0 };

function fixture({
  kind = 0,
  rows = 129,
  models = 1,
  refit = true,
  zero = false,
  nonfinite = false,
  rowChunk = 128,
}) {
  const coords = Float64Array.from({ length: rows * 4 }, (_, i) =>
    zero ? 0 : Math.sin(i * 0.17) * 0.7 + Math.cos(i * 0.03) * 0.2,
  );
  if (nonfinite) {
    coords[8] = NaN;
    coords[21] = Infinity;
  }
  const modelWork = new Float64Array(models * 98);
  const modelState = new Uint32Array(models * 12);
  for (let h = 0; h < models; h++) {
    modelState[h * 12] = h === 3 ? 0 : 1;
    modelState[h * 12 + 1] = refit ? rows : kind === 0 ? 8 : 4;
    for (let j = 0; j < 8; j++) modelState[h * 12 + 2 + j] = (h * 7 + j * 3) % rows;
    modelWork[h * 98 + 94] = zero ? 1 : 0.9;
    modelWork[h * 98 + 95] = zero ? 1 : 1.1;
    if (!zero) for (let j = 0; j < 4; j++) modelWork[h * 98 + 90 + j] = j * 0.03;
  }
  return {
    control: {
      first: camera,
      second: camera,
      kind,
      threshold: 0.001,
      seed: 71,
      start: 0,
      trials: models,
      maximum: models,
      minTrials: 1,
      blocks: Math.ceil(rows / 64),
      refit: Number(refit),
      logFailure: Math.log(0.01),
      verification: 0,
      rowStart: 0,
      rowCount: rowChunk,
    },
    coords,
    invertible: Uint32Array.from({ length: rows }, (_, i) => Number(i % 11 !== 0)),
    mask: Uint32Array.from({ length: rows }, (_, i) => Number(i % 7 !== 0)),
    valid: new Uint32Array(models),
    modelWork,
    modelState,
    rows,
    models,
    refit,
  };
}

function bindings(device, f, owned) {
  const own = (value) => (owned.push(value), value);
  return Object.fromEntries(
    ['control', 'coords', 'invertible', 'mask', 'valid', 'modelWork', 'modelState'].map((key) => [
      key,
      key === 'control'
        ? f[key]
        : own(
            f[key] instanceof Float64Array
              ? device.allocateF64(f[key].length, f[key])
              : device.allocateU32(f[key].length, f[key]),
          ),
    ]),
  );
}

async function runRows(device, f, kernel, { samples = 0 } = {}) {
  return device.enqueue(async () => {
    const owned = [];
    try {
      const b = bindings(device, f, owned);
      const steps = () => {
        const all = [];
        const limit = f.refit ? f.rows : f.control.kind === 0 ? 8 : 4;
        for (let rowStart = 0; rowStart < limit; rowStart += f.control.rowCount) {
          const chunk = { ...b, control: { ...b.control, rowStart } };
          all.push([kernel.name, () => kernel(chunk, Math.ceil(f.models / 8))]);
        }
        return all;
      };
      // Warm compilation and upload the resident bindings outside every measured sample.
      await device.plan(() => kernel(b, 0));
      const report = await device.planSteps(steps());
      const work = await device.read(b.modelWork, f.modelWork.length);
      const wallMs = [],
        gpuNs = [],
        longestDispatchNs = [];
      for (let sample = 0; sample < samples; sample++) {
        b.modelWork.write(f.modelWork);
        await device.plan(() => kernel(b, 0));
        const start = performance.now();
        const measured = await device.planSteps(steps());
        wallMs.push(performance.now() - start);
        gpuNs.push(measured.groups.reduce((sum, g) => sum + (g.gpuTime ?? 0), 0));
        longestDispatchNs.push(Math.max(0, ...measured.groups.map((g) => g.gpuTime ?? 0)));
      }
      return {
        bits: Array.from(new Uint32Array(work.buffer, work.byteOffset, work.byteLength / 4)),
        dispatches: report.dispatches,
        wallMs,
        gpuNs,
        longestDispatchNs,
      };
    } finally {
      for (const value of owned) device.drop(value);
    }
  });
}

export async function checkPairRowWindows({
  backend = 'cpu',
  fp64Flavor = 'auto',
  samples = 0,
  counts = [512, 513, 2049],
} = {}) {
  const device = await ShadeDevice.open({ backend, poolBytes: 0, fp64Flavor });
  const rows = [];
  try {
    for (const kind of [0, 1]) {
      for (const count of counts) {
        const input = { kind, rows: count };
        const first = await runRows(device, fixture(input), appendPairModelRows, { samples });
        const second = await runRows(
          device,
          fixture({ ...input, rowChunk: 512 }),
          appendPairModelRowsInPlace,
          { samples },
        );
        rows.push({
          input,
          exact: first.bits.every((v, i) => v === second.bits[i]),
          first,
          second,
        });
      }
    }
    return { backend, fp64Flavor, rows, statistics: device.statistics };
  } finally {
    await device.dispose();
  }
}

export async function measurePairRows({ samples = 5 } = {}) {
  const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
  const owned = [];
  const rows = [];
  try {
    for (const input of [
      { kind: 0, rows: 257, models: 256, refit: false },
      { kind: 1, rows: 257, models: 256, refit: false },
      { kind: 0, rows: 513 },
      { kind: 1, rows: 513 },
    ]) {
      const f = fixture(input);
      const variants = [appendPairModelRows, appendPairModelRowsInPlace].map((kernel) => ({
        kernel,
        b: bindings(device, f, owned),
        wallMs: [],
        gpuNs: [],
      }));
      for (const v of variants) await device.plan(() => v.kernel(v.b, 0));
      const execute = async (v, timed) => {
        v.b.modelWork.write(f.modelWork);
        await device.plan(() => v.kernel(v.b, 0));
        const steps = [];
        const limit = f.refit ? f.rows : f.control.kind === 0 ? 8 : 4;
        for (let rowStart = 0; rowStart < limit; rowStart += 128) {
          const chunk = { ...v.b, control: { ...v.b.control, rowStart } };
          steps.push([v.kernel.name, () => v.kernel(chunk, Math.ceil(f.models / 8))]);
        }
        const start = performance.now();
        const report = await device.planSteps(steps);
        if (timed) {
          v.wallMs.push(performance.now() - start);
          v.gpuNs.push(report.groups.reduce((sum, g) => sum + (g.gpuTime ?? 0), 0));
        }
      };
      for (const v of variants) await execute(v, false);
      for (let sample = 0; sample < samples; sample++)
        for (const v of sample % 2 ? [...variants].reverse() : variants) await execute(v, true);
      rows.push({
        input,
        variants: variants.map(({ kernel, wallMs, gpuNs }) => ({
          name: kernel.name,
          wallMs,
          gpuNs,
        })),
      });
      for (const value of owned.splice(0)) device.drop(value);
    }
    return { rows, statistics: device.statistics };
  } finally {
    for (const value of owned) device.drop(value);
    await device.dispose();
  }
}

export async function checkPairRows({ backend = 'cpu', samples = 0, fp64Flavor = 'auto' } = {}) {
  const device = await ShadeDevice.open({ backend, poolBytes: 0, fp64Flavor });
  const rows = [];
  try {
    const cases = [];
    for (const kind of [0, 1]) {
      cases.push({ kind, rows: 33, models: 13, refit: false });
      for (const count of [127, 128, 129, 513]) cases.push({ kind, rows: count });
      cases.push({ kind, rows: 129, zero: true });
      cases.push({ kind, rows: 129, nonfinite: true });
    }
    for (const input of cases) {
      const f = fixture(input);
      const first = await runRows(device, f, appendPairModelRows, { samples });
      const second = await runRows(device, f, appendPairModelRowsInPlace, { samples });
      rows.push({ input, exact: first.bits.every((v, i) => v === second.bits[i]), first, second });
    }
    return { backend, rows, statistics: device.statistics };
  } finally {
    await device.dispose();
  }
}
