// The browser bundle of the guided-matching probe: it runs `nearestFloatGuided` (the scalar
// oracle) or `nearestFloatGuidedTiles` on raw inputs and returns their raw outputs, so a test can
// hold the two kernels to each other bit for bit. `guided-probe-build.mjs` bundles it; the
// package does not export these kernels.
import { ShadeDevice } from '../../src/gpu/device.ts';
import matchingProgram, {
  nearestFloatGuided,
  nearestFloatGuidedTiles,
} from '../../src/gpu/kernels/float-matching.shade.ts';
import { createRuntime } from 'typeshade/runtime';

export { ShadeDevice };

/**
 * Nearest and second-nearest neighbours of each query under the epipolar gate. `matrix` is the
 * row-major 3 × 3 matrix that maps a query point to its line in the candidate image. The result
 * holds `indices` (best, second, invalid per query) and the f32 bits of `distances` (best,
 * second per query).
 */
export async function nearestGuided(device, kernel, input) {
  const n = input.queries.length / input.dimensions;
  return device.enqueue(async () => {
    const owned = [];
    try {
      const bindings = guidedBindings(device, input, owned);
      const report = await device.planSteps([
        [
          kernel,
          () =>
            kernel === 'tiles'
              ? nearestFloatGuidedTiles(bindings, Math.ceil(n / 8))
              : nearestFloatGuided(bindings, Math.ceil(n / 64)),
        ],
      ]);
      const indices = await device.read(bindings.nearestIndices, n * 3),
        distances = await device.read(bindings.nearestDistances, n * 2);
      return {
        tier: report.tier,
        indices: Array.from(indices.subarray(0, n * 3)),
        distanceBits: Array.from(
          new Uint32Array(Float32Array.from(distances.subarray(0, n * 2)).buffer),
        ),
      };
    } finally {
      for (const buffer of owned) device.drop(buffer);
    }
  });
}

function guidedBindings(device, input, owned) {
  const { queries, candidates, dimensions, queryPoints, candidatePoints, matrix, maxLinePx } =
    input;
  const n = queries.length / dimensions,
    m = candidates.length / dimensions;
  const own = (buffer) => (owned.push(buffer), buffer);
  const floatQueries = own(device.allocate(queries.length, Float32Array.from(queries))),
    floatCandidates = own(device.allocate(candidates.length, Float32Array.from(candidates))),
    queryBuffer = own(device.allocate(queryPoints.length, Float32Array.from(queryPoints))),
    candidateBuffer = own(
      device.allocate(candidatePoints.length, Float32Array.from(candidatePoints)),
    ),
    nearestIndices = own(device.allocateU32(n * 3)),
    nearestDistances = own(device.allocate(n * 2));
  const m32 = Array.from(matrix, Math.fround);
  const bindings = {
    floatQueries,
    floatCandidates,
    queryPoints: queryBuffer,
    candidatePoints: candidateBuffer,
    nearestIndices,
    nearestDistances,
    floatMatchControl: {
      queries: n,
      candidates: m,
      dimensions,
      ratioSquared: Math.fround(0.64),
      maxDistanceSquared: Math.fround(1e38),
      mutual: 0,
    },
    epipolarGuide: {
      m0: m32[0],
      m1: m32[1],
      m2: m32[2],
      m3: m32[3],
      m4: m32[4],
      m5: m32[5],
      m6: m32[6],
      m7: m32[7],
      m8: m32[8],
      maxLinePx: Math.fround(maxLinePx),
    },
  };
  return bindings;
}

/**
 * Experiment with the actual SfM kernel through TypeShade's program runtime. Compile once per
 * size, keep descriptor buffers resident, alternate variant order, and fence every sample.
 * GPU timestamps cover the compute pass; wall time covers encoding, submission and completion.
 * Compilation, descriptor uploads, output readback and timestamp mapping are outside samples.
 */
export async function measureGuidedWorkgroups(gpu, input, options = {}) {
  const sizes = options.sizes ?? [64, 128, 256];
  const samples = options.samples ?? 7;
  const warmup = options.warmup ?? 2;
  if (
    !Number.isSafeInteger(samples) ||
    samples < 1 ||
    !Number.isSafeInteger(warmup) ||
    warmup < 1 ||
    !sizes.length ||
    new Set(sizes).size !== sizes.length ||
    sizes.some((n) => ![64, 128, 256].includes(n))
  )
    throw new RangeError('Expected distinct workgroup sizes 64/128/256 and positive sample counts');
  const n = input.queries.length / input.dimensions;
  const m = input.candidates.length / input.dimensions;
  if (
    !Number.isSafeInteger(input.dimensions) ||
    input.dimensions < 1 ||
    !Number.isSafeInteger(n) ||
    n < 1 ||
    !Number.isSafeInteger(m) ||
    m < 1 ||
    input.queryPoints.length !== 2 * n ||
    input.candidatePoints.length !== 2 * m ||
    input.matrix.length !== 9 ||
    !Array.from(input.matrix).every(Number.isFinite) ||
    !Number.isFinite(input.maxLinePx) ||
    input.maxLinePx <= 0
  )
    throw new RangeError('Invalid guided-matching probe shape or epipolar guide');
  const device = await ShadeDevice.open({ gpuDevice: gpu });
  let rt;
  const owned = [];
  let querySet, resolved, staging;
  try {
    rt = await createRuntime({ device: gpu });
    const program = rt.load(matchingProgram);
    const reference = await nearestGuided(device, 'scalar', input);
    const bindings = guidedBindings(device, input, owned);
    const timed = gpu.features.has('timestamp-query');
    if (timed) {
      querySet = gpu.createQuerySet({ type: 'timestamp', count: 2 });
      resolved = gpu.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      staging = gpu.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }
    const variants = [],
      skipped = [];
    for (const size of sizes) {
      if (
        size >
        Math.min(gpu.limits.maxComputeInvocationsPerWorkgroup, gpu.limits.maxComputeWorkgroupSizeX)
      ) {
        skipped.push({ size, reason: 'device workgroup limit' });
        continue;
      }
      const start = performance.now();
      const pipeline = await program.compute('nearestFloatGuided', {
        constants: { GUIDED_WG: size },
      });
      const compileMs = performance.now() - start;
      const shape = [...pipeline.workgroupShape];
      if (shape[0] !== size || shape[1] !== 1 || shape[2] !== 1)
        throw new Error('The specialized pipeline reported the wrong workgroup shape');
      const groups = Math.ceil(n / shape[0]);
      if (groups > gpu.limits.maxComputeWorkgroupsPerDimension) {
        skipped.push({ size, reason: 'device dispatch limit' });
        continue;
      }
      variants.push({
        size,
        shape,
        groups,
        compileMs,
        pipeline,
        wallMs: [],
        gpuMs: [],
        cached:
          pipeline ===
          (await program.compute('nearestFloatGuided', { constants: { GUIDED_WG: size } })),
      });
    }
    if (!variants.length) throw new RangeError('No workgroup variant fits the device limits');
    const run = async (row, measure) => {
      await gpu.queue.onSubmittedWorkDone();
      const start = performance.now();
      const encoder = gpu.createCommandEncoder();
      const pass = encoder.beginComputePass(
        measure && timed
          ? {
              timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
            }
          : {},
      );
      row.pipeline.dispatch(pass, bindings, row.groups);
      pass.end();
      await rt.submit(encoder);
      await gpu.queue.onSubmittedWorkDone();
      const wallMs = performance.now() - start;
      if (!measure) return;
      row.wallMs.push(wallMs);
      if (timed) {
        const copy = gpu.createCommandEncoder();
        copy.resolveQuerySet(querySet, 0, 2, resolved, 0);
        copy.copyBufferToBuffer(resolved, 0, staging, 0, 16);
        gpu.queue.submit([copy.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        try {
          const ticks = new BigUint64Array(staging.getMappedRange());
          if (ticks[1] < ticks[0]) throw new Error('GPU timestamps are not monotonic');
          row.gpuMs.push(Number(ticks[1] - ticks[0]) / 1e6);
        } finally {
          staging.unmap();
        }
      }
    };
    const check = async () => {
      const indices = Array.from(await device.read(bindings.nearestIndices, 3 * n));
      const distances = await device.read(bindings.nearestDistances, 2 * n);
      const bits = Array.from(new Uint32Array(Float32Array.from(distances).buffer));
      if (
        indices.some((v, i) => v !== reference.indices[i]) ||
        bits.some((v, i) => v !== reference.distanceBits[i])
      )
        throw new Error('Specialized guided matching differs from the direct 64-lane kernel');
    };
    for (const row of variants) {
      for (let k = 0; k < warmup; k++) await run(row, false);
      await check();
    }
    // Rotate and reverse the order to distribute warm-device and system-load effects.
    for (let round = 0; round < samples; round++) {
      const order = variants.map((_, i) => variants[(i + round) % variants.length]);
      if (round % 2) order.reverse();
      for (const row of order) await run(row, true);
    }
    for (const row of variants) {
      await run(row, false);
      await check();
    }
    const median = (values) => {
      if (!values.length) return null;
      const a = [...values].sort((x, y) => x - y),
        m = a.length >> 1;
      return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
    };
    return {
      queries: n,
      candidates: input.candidates.length / input.dimensions,
      dimensions: input.dimensions,
      maxLinePx: input.maxLinePx,
      bitExact: true,
      timestampQuery: timed,
      skipped,
      variants: variants.map(({ pipeline, ...row }) => ({
        ...row,
        medianWallMs: median(row.wallMs),
        medianGpuMs: median(row.gpuMs),
      })),
    };
  } finally {
    for (const buffer of owned) device.drop(buffer);
    querySet?.destroy();
    resolved?.destroy();
    staging?.destroy();
    rt?.destroy();
    await device.dispose();
  }
}

/** Shared browser entry for the regression and the native-GPU benchmark. */
export async function guidedWorkgroupBenchmark({ fixtures, samples, warmup, hardware }) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const info = adapter.info;
  const identity = [info?.vendor, info?.architecture, info?.device, info?.description]
    .filter(Boolean)
    .join(' ');
  const software =
    /swiftshader|software|llvmpipe|softpipe/i.test(identity) || adapter.isFallbackAdapter === true;
  if (hardware && (software || !identity.trim()))
    throw new Error('Hardware requested without a verified native GPU: ' + identity);
  const gpu = await adapter.requestDevice({
    requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [],
  });
  const errors = [];
  gpu.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
  try {
    const results = [];
    for (const fixture of fixtures)
      results.push({
        name: fixture.name,
        ...(await measureGuidedWorkgroups(gpu, fixture, { samples, warmup })),
      });
    if (errors.length) throw new Error('WebGPU validation: ' + errors.join('; '));
    return { adapter: identity, software, results };
  } finally {
    gpu.destroy();
  }
}
