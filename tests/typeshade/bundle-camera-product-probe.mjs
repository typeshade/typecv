// Compile only the changed product kernels: a complete integer-fp64 BA program exceeds
// SwiftShader's resources (the existing camera-assembly probe has the same restriction).
import { planCameraProduct } from '../../src/gpu/bundle-camera-product.ts';
import { ShadeDevice } from '../../src/gpu/device.ts';
import {
  bundleSchurCamera as fixedRows,
  bundleSchurCameraWorkgroup as fixedCamera,
  bundleSchurCameraSegments as fixedSegments,
  finishBundleSchurCameraSegments as fixedFinish,
} from '../../src/gpu/bundle.shade.ts';
import {
  bundleSchurCamera as focalRows,
  bundleSchurCameraWorkgroup as focalCamera,
  bundleSchurCameraSegments as focalSegments,
  finishBundleSchurCameraSegments as focalFinish,
  finishBundleSchurFocal,
} from '../../src/gpu/joint-focal.shade.ts';

function fixture(shared, active) {
  const C = 3,
    P = 17,
    M = 1031,
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
    const c = i < 773 ? 0 : i % C;
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
      directionalPoints: 0,
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

export async function checkCameraProductFlavors({
  integer = false,
  hardware = false,
  sizes = [64],
  tiles = [undefined],
} = {}) {
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
          const variants = [
            { flavor: 'cpu', size: 64, tile: undefined },
            ...sizes.flatMap((size) =>
              tiles.map((tile) => ({ flavor: integer ? 'integer' : 'float', size, tile })),
            ),
          ];
          for (const { flavor, size, tile } of variants) {
            const device = await ShadeDevice.open(
              flavor === 'cpu'
                ? { backend: 'cpu', poolBytes: 0 }
                : { gpuDevice: gpu, fp64Flavor: flavor, poolBytes: 0 },
            );
            const counts = Array.from(
              { length: input.control.cameras },
              (_, c) =>
                input.index[2 * input.control.observations + input.control.points + 2 + c] -
                input.index[2 * input.control.observations + input.control.points + 1 + c],
            );
            const plan = planCameraProduct(device, counts, shared, size, tile);
            const data = {
              ...input,
              cameraLanes: new Float64Array(plan.laneElements),
              ...(tile === undefined
                ? {}
                : { cameraTasks: plan.tasks, cameraParts: new Float64Array(plan.partialElements) }),
            };
            const bindings = { control: { ...input.control } },
              owned = [];
            try {
              for (const [name, value] of Object.entries(data)) {
                if (name === 'control') continue;
                const buffer =
                  name === 'index' || name === 'cameraTasks'
                    ? device.allocateU32(value.length, value)
                    : device.allocateF64(value.length, value);
                bindings[name] = buffer;
                owned.push(buffer);
              }
              const run = async (first, last, obsBase = 0, obsEnd = input.control.observations) => {
                if (obsBase) {
                  const packed = input.blocks.slice();
                  const stride = shared ? 21 : 18,
                    M = input.control.observations;
                  packed.set(
                    input.blocks.subarray(24 * M + obsBase * stride, 24 * M + obsEnd * stride),
                    24 * M,
                  );
                  await bindings.blocks.write(packed);
                }
                bindings.control = {
                  ...input.control,
                  first,
                  last,
                  obsBase,
                  obsEnd,
                };
                await device.plan(() => {
                  if (flavor === 'cpu') {
                    const { cameraLanes, cameraTasks, cameraParts, ...rows } = bindings;
                    (shared ? focalRows : fixedRows)(rows, 1);
                  } else {
                    const constants = { constants: { BA_CAMERA_PRODUCT_WG: size } };
                    if (tile === undefined)
                      (shared ? focalCamera : fixedCamera)(
                        bindings,
                        input.control.cameras,
                        constants,
                      );
                    else {
                      const { focal, ...segments } = bindings;
                      (shared ? focalSegments : fixedSegments)(segments, plan.taskCount, constants);
                      if (last) {
                        const { control, blocks, vectors, state, cameraTasks, cameraParts } =
                          bindings;
                        (shared ? focalFinish : fixedFinish)(
                          {
                            control,
                            blocks,
                            vectors,
                            state,
                            cameraTasks,
                            cameraParts,
                            ...(shared ? { focal } : {}),
                          },
                          1,
                        );
                      }
                    }
                    if (shared && last) {
                      const { control, vectors, state, focal } = bindings;
                      finishBundleSchurFocal({ control, vectors, state, focal }, 1);
                    }
                  }
                });
              };
              if (carry) {
                await run(1, 0, 0, 517);
                await run(0, 1, 517);
              } else await run(1, 1);
              const m = input.control.cameras * 6 + Number(shared);
              values.push({
                flavor,
                size,
                tile,
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
              size: v.size,
              tile: v.tile,
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
