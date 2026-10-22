// The real BA row kernels, specialized through TypeShade's public program runtime. This
// probe measures overrides independently of the call/fuse path used by the production solver.
import { createRuntime } from 'typeshade/runtime';
import { ShadeDevice } from '../../src/gpu/device.ts';
import programManifest, {
  bundleCameraRows,
  bundleCameraRowsCooperative,
} from '../../src/gpu/bundle.shade.ts';

function fixture(count, { first = 1, last = 1, active = 1 } = {}) {
  const C = 4,
    P = count * C - 3,
    M = P,
    blocks = new Float64Array(42 * M + 13 * P + 78 * C),
    index = new Uint32Array(3 * M + P + 2 * C + 2),
    cameraScratch = Float64Array.from({ length: C * 36 }, (_, i) => (i % 7 === 0 ? 2 : 0.01)),
    state = new Float64Array(26),
    lists = Array.from({ length: C }, () => []);
  state[0] = 0.001;
  state[1] = active;
  for (let i = 0; i < M; i++) {
    const c = i % C;
    lists[c].push(i);
    index[M + i] = i;
    for (let j = 0; j < 24; j++) blocks[i * 24 + j] = Math.sin(i * 0.7 + j) * 3;
    blocks[i * 24 + 20] = 0.5 + (i % 7) / 7;
    for (let j = 0; j < 18; j++) blocks[M * 24 + i * 18 + j] = Math.cos(i + j) / 3;
    for (let j = 0; j < 12; j++)
      blocks[M * 42 + i * 13 + j] = (j % 4 === 0 ? 0.1 : 0.001) + j / 1000;
  }
  let at = 0;
  for (let c = 0; c < C; c++) {
    index[2 * M + P + 1 + c] = at;
    index.set(lists[c], 2 * M + P + C + 2 + at);
    index[3 * M + P + C + 2 + c] = Number(c === 1 || c === 2);
    at += lists[c].length;
  }
  index[2 * M + P + C + 1] = at;
  for (let i = 42 * M + 13 * P; i < blocks.length; i++) blocks[i] = i % 11 === 0 ? 1 : 0.01;
  return {
    control: {
      cameras: C,
      points: P,
      observations: M,
      anchor: 0,
      scaleView: 1,
      normalizeScale: 0,
      huber: 1,
      relativeTolerance: 1e-6,
      obsBase: 0,
      obsEnd: M,
      pointBase: 0,
      pointEnd: P,
      obsCapacity: M,
      pointCapacity: P,
      first,
      last,
      sweep: 0,
      streamed: 0,
      base: 0,
    },
    blocks,
    index,
    cameraScratch,
    state,
  };
}

// Exercise the Apple arithmetic flavor on the changed kernels alone. Compiling the entire
// integer BA solver (including its matrix inverse) exceeds a software adapter's resources.
export async function checkIntegerCameraAssembly(hardware = false) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const description = [
    adapter.info?.vendor,
    adapter.info?.architecture,
    adapter.info?.description,
  ].join(' ');
  if (hardware && (/swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter))
    throw new Error(`Hardware requested, software adapter selected: ${description}`);
  const gpu = await adapter.requestDevice(),
    validation = [],
    rows = [];
  gpu.addEventListener('uncapturederror', (event) => validation.push(event.error.message));
  try {
    for (const mode of [{}, { first: 0, last: 0 }, { first: 0, last: 1 }, { active: 0 }]) {
      const input = fixture(131, mode),
        values = [];
      for (const backend of ['cpu', 'serial', 'cooperative']) {
        const device = await ShadeDevice.open(
          backend === 'cpu'
            ? { backend: 'cpu' }
            : { gpuDevice: gpu, fp64Flavor: 'integer', poolBytes: 0 },
        );
        const bindings = { control: input.control },
          owned = [];
        try {
          for (const [name, value] of Object.entries(input)) {
            if (name === 'control') continue;
            const buffer =
              name === 'index'
                ? device.allocateU32(value.length, value.slice())
                : device.allocateF64(value.length, value.slice());
            bindings[name] = buffer;
            owned.push(buffer);
          }
          await device.plan(() =>
            backend === 'cooperative'
              ? bundleCameraRowsCooperative(bindings, input.control.cameras * 6)
              : bundleCameraRows(bindings, Math.ceil((input.control.cameras * 6) / 64)),
          );
          values.push([
            ...(await device.read(bindings.blocks)).slice(-78 * input.control.cameras),
            ...(await device.read(bindings.cameraScratch)),
          ]);
        } finally {
          for (const buffer of owned) device.drop(buffer);
          await device.dispose();
        }
      }
      const [reference, ...runs] = values;
      rows.push({
        mode,
        errors: runs.map((run) =>
          Math.max(
            ...run.map((v, i) => Math.abs(v - reference[i]) / Math.max(1, Math.abs(reference[i]))),
          ),
        ),
      });
    }
    await gpu.queue.onSubmittedWorkDone();
    return { rows, validation };
  } finally {
    gpu.destroy();
  }
}

export async function measureCameraWorkgroups({
  counts = [40, 131, 1025],
  samples = 5,
  modes = [{}],
  hardware = false,
} = {}) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const description = [
    adapter.info?.vendor,
    adapter.info?.architecture,
    adapter.info?.description,
  ].join(' ');
  if (hardware && (/swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter))
    throw new Error(`Hardware requested, software adapter selected: ${description}`);
  const timed = adapter.features.has('timestamp-query'),
    gpu = await adapter.requestDevice({ requiredFeatures: timed ? ['timestamp-query'] : [] }),
    rt = await createRuntime({ device: gpu }),
    program = rt.load(programManifest),
    device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0 }),
    cpu = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 }),
    validation = [],
    rows = [];
  gpu.addEventListener('uncapturederror', (event) => validation.push(event.error.message));
  const query = timed ? gpu.createQuerySet({ type: 'timestamp', count: 2 }) : undefined,
    resolved = timed
      ? gpu.createBuffer({
          size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        })
      : undefined,
    staging = timed
      ? gpu.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
      : undefined;
  try {
    for (const count of counts)
      for (const mode of modes) {
        const input = fixture(count, mode),
          owned = [];
        const bindings = { control: input.control };
        for (const [name, value] of Object.entries(input)) {
          if (name === 'control') continue;
          const buffer =
            name === 'index'
              ? device.allocateU32(value.length, value)
              : device.allocateF64(value.length, value);
          bindings[name] = buffer;
          owned.push(buffer);
        }
        const oracle = { control: input.control },
          cpuOwned = [];
        for (const [name, value] of Object.entries(input)) {
          if (name === 'control') continue;
          const buffer =
            name === 'index'
              ? cpu.allocateU32(value.length, value)
              : cpu.allocateF64(value.length, value);
          oracle[name] = buffer;
          cpuOwned.push(buffer);
        }
        const suffix = 78 * input.control.cameras,
          variants = [];
        let expected;
        try {
          await cpu.plan(() =>
            bundleCameraRows(oracle, Math.ceil((input.control.cameras * 6) / 64)),
          );
          expected = [
            ...(await cpu.read(oracle.blocks)).slice(-suffix),
            ...(await cpu.read(oracle.cameraScratch)),
          ];
        } finally {
          for (const buffer of cpuOwned) cpu.drop(buffer);
        }
        try {
          for (const cooperative of [false, true])
            for (const size of cooperative ? [32, 64, 128] : [8, 16, 32, 64, 128]) {
              const entry = cooperative ? 'bundleCameraRowsCooperative' : 'bundleCameraRows',
                constants = cooperative
                  ? { BA_CAMERA_REDUCE_WG: size }
                  : { BA_CAMERA_ROWS_WG: size },
                pipeline = await program.compute(entry, { constants }),
                groups = cooperative
                  ? input.control.cameras * 6
                  : Math.ceil((input.control.cameras * 6) / pipeline.workgroupShape[0]),
                row = {
                  cooperative,
                  size,
                  shape: [...pipeline.workgroupShape],
                  groups,
                  cached: pipeline === (await program.compute(entry, { constants })),
                  wallMs: [],
                  gpuMs: [],
                };
              const run = async (measure) => {
                // Restore continuation inputs outside the measured pass. Fresh assembly overwrites
                // its outputs, so the normal benchmark never uploads between samples.
                if (!input.control.first) {
                  await bindings.blocks.write(input.blocks);
                  await bindings.cameraScratch.write(input.cameraScratch);
                }
                await gpu.queue.onSubmittedWorkDone();
                const start = performance.now(),
                  encoder = gpu.createCommandEncoder(),
                  pass = encoder.beginComputePass(
                    measure && timed
                      ? {
                          timestampWrites: {
                            querySet: query,
                            beginningOfPassWriteIndex: 0,
                            endOfPassWriteIndex: 1,
                          },
                        }
                      : {},
                  );
                pipeline.dispatch(pass, bindings, groups);
                pass.end();
                await rt.submit(encoder);
                if (measure) {
                  row.wallMs.push(performance.now() - start);
                  if (timed) {
                    const copy = gpu.createCommandEncoder();
                    copy.resolveQuerySet(query, 0, 2, resolved, 0);
                    copy.copyBufferToBuffer(resolved, 0, staging, 0, 16);
                    gpu.queue.submit([copy.finish()]);
                    await staging.mapAsync(GPUMapMode.READ);
                    const ticks = new BigUint64Array(staging.getMappedRange());
                    row.gpuMs.push(Number(ticks[1] - ticks[0]) / 1e6);
                    staging.unmap();
                  }
                }
              };
              await run(false);
              variants.push({ ...row, run });
            }
          for (let round = 0; round < samples; round++) {
            const order = variants.map((_, i) => variants[(i + round) % variants.length]);
            if (round % 2) order.reverse();
            for (const row of order) await row.run(true);
          }
          for (const row of variants) {
            await row.run(false);
            const actual = [
              ...(await device.read(bindings.blocks)).slice(-suffix),
              ...(await device.read(bindings.cameraScratch)),
            ];
            row.maxRelativeError = Math.max(
              ...actual.map(
                (v, i) => Math.abs(v - expected[i]) / Math.max(1, Math.abs(expected[i])),
              ),
            );
            row.bitExact = actual.every((v, i) => Object.is(v, expected[i]));
            if (!Number.isFinite(row.maxRelativeError) || row.maxRelativeError > 2e-10)
              throw new Error(
                `Camera assembly differs from CPU: ${JSON.stringify({
                  count,
                  mode,
                  size: row.size,
                  cooperative: row.cooperative,
                  error: row.maxRelativeError,
                })}`,
              );
          }
          rows.push({ count, mode, variants: variants.map(({ run, ...row }) => row) });
        } finally {
          for (const buffer of owned) device.drop(buffer);
        }
      }
    await gpu.queue.onSubmittedWorkDone();
    return { description, timestampQuery: timed, validation, rows };
  } finally {
    query?.destroy();
    resolved?.destroy();
    staging?.destroy();
    await cpu.dispose();
    await device.dispose();
    rt.destroy();
    gpu.destroy();
  }
}
