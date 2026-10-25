// Browser-side runtime only. Fixtures and the f64 oracle are built independently in Node.
import { createRuntime, configure, resident } from 'typeshade/runtime';

export async function runNccGpu({ packsUrl, fixtures, hardware }) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const description = [adapter.info?.vendor, adapter.info?.architecture, adapter.info?.description]
    .filter(Boolean)
    .join(' ');
  if (hardware && (/swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter))
    throw new Error(`Hardware requested, software adapter selected: ${description}`);
  const gpu = await adapter.requestDevice(),
    validation = [],
    uncaptured = [],
    lost = [],
    native = {
      pipelines: 0,
      dispatches: 0,
      readbacks: 0,
      createdBuffers: 0,
      destroyedBuffers: 0,
      liveBuffers: null,
    },
    buffers = new Set(),
    restore = [];
  const wrap = (object, name, callback) => {
    const own = Object.getOwnPropertyDescriptor(object, name),
      original = object[name];
    Object.defineProperty(object, name, {
      configurable: true,
      value(...args) {
        return callback(original, this, args);
      },
    });
    restore.push(() => {
      if (own) Object.defineProperty(object, name, own);
      else delete object[name];
    });
  };
  for (const method of ['createComputePipeline', 'createComputePipelineAsync'])
    wrap(gpu, method, (fn, self, args) => {
      native.pipelines++;
      return Reflect.apply(fn, self, args);
    });
  wrap(gpu, 'createBuffer', (fn, self, args) => {
    const buffer = Reflect.apply(fn, self, args),
      destroy = buffer.destroy.bind(buffer);
    native.createdBuffers++;
    buffers.add(buffer);
    buffer.destroy = () => {
      if (buffers.delete(buffer)) native.destroyedBuffers++;
      destroy();
    };
    return buffer;
  });
  wrap(GPUComputePassEncoder.prototype, 'dispatchWorkgroups', (fn, self, args) => {
    native.dispatches++;
    return Reflect.apply(fn, self, args);
  });
  wrap(GPUBuffer.prototype, 'mapAsync', (fn, self, args) => {
    if (args[0] & GPUMapMode.READ) native.readbacks++;
    return Reflect.apply(fn, self, args);
  });
  gpu.addEventListener('uncapturederror', (event) => uncaptured.push(event.error.message));
  gpu.lost.then((info) => {
    if (info.reason !== 'destroyed') lost.push(info.message);
  });
  const rt = await createRuntime({ device: gpu });
  configure({ runtime: rt, prefer: ['webgpu'] });
  const packs = await fetch(packsUrl).then((response) => response.json()),
    rows = [];
  try {
    for (const fixture of fixtures) {
      const variants = {};
      for (const [label, pack] of Object.entries(packs)) {
        const owned = [],
          keep = (array) => {
            const value = resident(array);
            owned.push(value);
            return value;
          };
        gpu.pushErrorScope('out-of-memory');
        gpu.pushErrorScope('internal');
        gpu.pushErrorScope('validation');
        try {
          const output = keep(new Float32Array(20)),
            pipeline = await rt.load(pack).compute('nccGpuProbe'),
            frame = rt.frame();
          // Separate output, uploaded inputs and invocation-local stats/patch for each variant.
          frame.dispatch(
            pipeline,
            {
              mvsControl: { ...fixture.mvsControl },
              mvsCameras: keep(Float32Array.from(fixture.mvsCameras)),
              mvsGray: keep(Float32Array.from(fixture.mvsGray)),
              mvsSources: keep(Uint32Array.from(fixture.mvsSources)),
              mvsDepth: keep(Float32Array.from(fixture.mvsDepth)),
              nccProbeHypothesis: keep(Float32Array.from(fixture.hypothesis)),
              nccProbeOutput: output,
            },
            1,
          );
          await frame.submit();
          variants[label] = { values: Array.from(await output.read()) };
        } catch (error) {
          variants[label] = { error: String(error) };
        } finally {
          for (const value of owned) value.destroy();
          for (const scope of ['validation', 'internal', 'out-of-memory']) {
            const error = await gpu.popErrorScope();
            if (error)
              validation.push({ fixture: fixture.id, label, scope, message: error.message });
          }
        }
      }
      rows.push({ id: fixture.id, variants });
    }
    await gpu.queue.onSubmittedWorkDone();
    await new Promise((resolve) => setTimeout(resolve, 0));
    return { backend: rt.tier, description, validation, uncaptured, lost, native, rows };
  } finally {
    rt.destroy();
    configure({ runtime: null });
    configure({});
    native.liveBuffers = buffers.size;
    gpu.destroy();
    for (const reset of restore.reverse()) reset();
  }
}
