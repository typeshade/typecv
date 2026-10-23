// Compile only the changed product kernels: a complete integer-fp64 BA program exceeds
// SwiftShader's resources (the existing camera-assembly probe has the same restriction).
import { ShadeDevice } from '../../src/gpu/device.ts';
import {
  bundleSchurCamera as fixedRows,
  bundleSchurCameraWorkgroup as fixedCamera,
} from '../../src/gpu/bundle.shade.ts';
import {
  bundleSchurCamera as focalRows,
  bundleSchurCameraWorkgroup as focalCamera,
  finishBundleSchurFocal,
} from '../../src/gpu/joint-focal.shade.ts';

function fixture(shared, active) {
  const C = 3,
    P = 17,
    M = 131,
    m = 6 * C + Number(shared),
    stride = shared ? 21 : 18;
  const index = new Uint32Array(3 * M + P + 2 * C + 2),
    lists = Array.from({ length: C }, () => []);
  const blocks = new Float64Array((shared ? 45 : 42) * M + 13 * P + 78 * C);
  const vectors = Float64Array.from(
    { length: 5 * m + 3 * P + (shared ? C : 0) },
    (_, i) => Math.sin(i * 0.7) * 0.03,
  );
  for (let i = 0; i < M; i++) {
    const c = i % C;
    index[i] = c;
    index[M + i] = i % P;
    lists[c].push(i);
    for (let j = 0; j < stride; j++) blocks[24 * M + i * stride + j] = Math.cos(i * 0.3 + j) * 0.1;
  }
  let cursor = 0;
  for (let c = 0; c < C; c++) {
    index[2 * M + P + 1 + c] = cursor;
    index.set(lists[c], 2 * M + P + C + 2 + cursor);
    cursor += lists[c].length;
    for (let j = 0; j < 36; j++)
      blocks[(shared ? 45 : 42) * M + 13 * P + 36 * c + j] = j % 7 === 0 ? 2 : 0.001;
  }
  index[2 * M + P + 1 + C] = cursor;
  const state = new Float64Array(26);
  state[19] = Number(active);
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
      ...(shared ? { cgIterations: 32 } : {}),
      obsBase: 0,
      obsEnd: M,
      pointBase: 0,
      pointEnd: P,
      obsCapacity: M,
      pointCapacity: P,
      first: 1,
      last: 1,
      sweep: 1,
      streamed: 0,
      base: 0,
    },
    index,
    blocks,
    vectors,
    state,
    cameraLanes: new Float64Array((shared ? 7 : 6) * 64 * C),
    ...(shared
      ? { focal: Float64Array.from({ length: 6 * C + 6 + 2 * M }, (_, i) => Math.sin(i) * 0.1) }
      : {}),
  };
}

export async function checkCameraProductFlavors({ integer = false, hardware = false } = {}) {
  const adapter = await navigator.gpu.requestAdapter(),
    gpu = await adapter.requestDevice(),
    validation = [],
    rows = [];
  if (
    hardware &&
    (/swiftshader|software|llvmpipe/i.test(
      [adapter.info?.vendor, adapter.info?.description].join(' '),
    ) ||
      adapter.isFallbackAdapter)
  ) {
    gpu.destroy();
    throw new Error('Native camera-product check selected a software adapter');
  }
  gpu.addEventListener('uncapturederror', (e) => validation.push(e.error.message));
  try {
    for (const shared of [false, true])
      for (const active of [true, false])
        for (const carry of [false, true]) {
          const input = fixture(shared, active),
            values = [];
          for (const flavor of ['cpu', integer ? 'integer' : 'float']) {
            const device = await ShadeDevice.open(
              flavor === 'cpu'
                ? { backend: 'cpu', poolBytes: 0 }
                : { gpuDevice: gpu, fp64Flavor: flavor, poolBytes: 0 },
            );
            const bindings = { control: { ...input.control } },
              owned = [];
            try {
              for (const [name, value] of Object.entries(input)) {
                if (name === 'control') continue;
                const buffer =
                  name === 'index'
                    ? device.allocateU32(value.length, value)
                    : device.allocateF64(value.length, value);
                bindings[name] = buffer;
                owned.push(buffer);
              }
              const run = async (first, last, empty = false) => {
                bindings.control = {
                  ...input.control,
                  first,
                  last,
                  ...(empty ? { obsBase: input.control.observations } : {}),
                };
                await device.plan(() => {
                  if (flavor === 'cpu') {
                    const { cameraLanes, ...rows } = bindings;
                    (shared ? focalRows : fixedRows)(rows, 1);
                  } else {
                    (shared ? focalCamera : fixedCamera)(bindings, input.control.cameras);
                    if (shared && last) {
                      const { control, vectors, state, focal } = bindings;
                      finishBundleSchurFocal({ control, vectors, state, focal }, 1);
                    }
                  }
                });
              };
              if (carry) {
                await run(1, 0);
                await run(0, 1, true);
              } else await run(1, 1);
              const m = input.control.cameras * 6 + Number(shared);
              values.push({
                flavor,
                owner: device.fp64Flavor,
                vector: [...(await device.read(bindings.vectors)).slice(3 * m, 4 * m)],
              });
            } finally {
              for (const b of owned) device.drop(b);
              if (device.statistics.liveBuffers) throw new Error('Product probe leaked buffers');
              await device.dispose();
            }
          }
          const reference = values[0].vector,
            scale = Math.max(1, ...reference.map(Math.abs));
          rows.push({
            shared,
            active,
            carry,
            errors: values.slice(1).map((v) => ({
              flavor: v.flavor,
              owner: v.owner,
              error: Math.max(...v.vector.map((x, i) => Math.abs(x - reference[i]))) / scale,
            })),
          });
        }
    return { validation, rows };
  } finally {
    gpu.destroy();
  }
}
