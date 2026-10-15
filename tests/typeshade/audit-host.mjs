// Browser-side consumer of precompiled manifests. No parser or compiler is bundled here.
import { createRuntime, configure, resident } from 'typeshade/runtime';

export async function runAudit({ packs, at, problems, flavors }) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const gpu = await adapter.requestDevice(),
    errors = [];
  gpu.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
  const rt = await createRuntime({ device: gpu });
  configure({ runtime: rt, prefer: ['webgpu'] });
  const owned = [];
  const keep = (values) => {
    const r = resident(values);
    owned.push(r);
    return r;
  };
  const results = {};
  let dispatches = 0,
    submissions = 0;
  const originalDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups;
  const originalSubmit = GPUQueue.prototype.submit;
  GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
    dispatches++;
    return originalDispatch.apply(this, args);
  };
  GPUQueue.prototype.submit = function (...args) {
    submissions++;
    return originalSubmit.apply(this, args);
  };
  try {
    const sourceCoordinates = Float64Array.of(1e8 + 0.0625, -1e8 + 0.0625, 2 ** 24 + 0.0625);
    const input = keep(sourceCoordinates);
    for (const flavor of flavors) {
      console.debug(`audit: precision ${flavor}`);
      const precise = keep(new Float64Array(3)),
        ordinary = keep(new Float32Array(3));
      const pipeline = await rt.load(packs[`precision_${flavor}`]).compute('compareWords');
      const frame = rt.frame();
      frame.dispatch(pipeline, { values: input, wide: precise, ordinary }, 1);
      await frame.submit();
      const wideValues = await precise.read(),
        stored = await ordinary.read();
      results[`precision_${flavor}`] = {
        precise: [...wideValues],
        ordinaryStored: [...stored],
        ordinary: Array.from(stored, (value, i) => value - Math.fround(sourceCoordinates[i])),
      };
      const roots = keep(new Float64Array(problems.length)),
        valid = keep(new Uint32Array(problems.length));
      const data = keep(problems);
      const solve = await rt.load(packs[`quartic_${flavor}`]).compute('bisect');
      console.debug(`audit: quartic ${flavor}`);
      const solving = rt.frame();
      solving.dispatch(solve, { problems: data, roots, valid }, 1);
      await solving.submit();
      results[`quartic_${flavor}`] = {
        roots: [...(await roots.read())],
        valid: [...(await valid.read())],
      };
    }
    const source = Float32Array.from({ length: 513 }, (_, i) => (i % 7) - 3),
      values = keep(source);
    results.workgroups = [];
    for (const width of [32, 64]) {
      console.debug(`audit: workgroup ${width}`);
      const groups = Math.ceil(source.length / width),
        sums = keep(new Float32Array(groups)),
        accepted = keep(new Uint32Array(1));
      const pipeline = await rt
        .load(packs.workgroup)
        .compute('reduce', { constants: { WG: width } });
      const frame = rt.frame();
      frame.dispatch(pipeline, { values, sums, accepted }, groups);
      await frame.submit();
      results.workgroups.push({
        width,
        sums: [...(await sums.read())],
        accepted: [...(await accepted.read())],
      });
    }
    const points = keep(Float32Array.from(at.flat())),
      outputs = keep(new Float32Array(at.length));
    const derivative = await rt.load(packs.derivative).compute('derivative');
    console.debug('audit: derivative');
    const frame = rt.frame();
    frame.dispatch(derivative, { points, outputs }, 1);
    await frame.submit();
    results.derivative = [...(await outputs.read())];
    const info = adapter.info;
    return {
      backend: rt.tier,
      adapter: [info?.vendor, info?.architecture, info?.description].join(' ').trim(),
      results,
      errors,
      dispatches,
      submissions,
    };
  } finally {
    GPUComputePassEncoder.prototype.dispatchWorkgroups = originalDispatch;
    GPUQueue.prototype.submit = originalSubmit;
    for (const r of owned) r.destroy();
    rt.destroy();
    configure({ runtime: null });
    configure({});
    gpu.destroy();
  }
}
