/** Actual borrowed-device instrumentation; device invalidation is not buffer.destroy(). */
function instrument(gpu) {
  const records = [],
    live = new Map(),
    snapshots = [],
    restore = [];
  const native = { pipelines: 0, dispatches: 0, readbacks: 0, deviceDestroyCalls: 0 };
  let phase = 'open',
    sequence = 0;
  const wrap = (object, name, callback) => {
    const own = Object.getOwnPropertyDescriptor(object, name),
      original = object[name];
    Object.defineProperty(object, name, {
      configurable: true,
      value(...args) {
        return callback(original, this, args);
      },
    });
    restore.push(() => (own ? Object.defineProperty(object, name, own) : delete object[name]));
  };
  wrap(gpu, 'createBuffer', (fn, self, args) => {
    const descriptor = args[0],
      stack = new Error('GPUBuffer allocation').stack;
    const buffer = Reflect.apply(fn, self, args);
    const row = {
      id: ++sequence,
      size: Number(descriptor.size),
      usage: Number(descriptor.usage),
      label: descriptor.label ?? '',
      mappedAtCreation: descriptor.mappedAtCreation ?? false,
      phase,
      stack,
      destroyed: false,
      destroyCalls: 0,
      invalidatedByDevice: false,
    };
    records.push(row);
    live.set(buffer, row);
    wrap(buffer, 'destroy', (destroy, target, values) => {
      const result = Reflect.apply(destroy, target, values);
      row.destroyCalls++;
      row.destroyed = true;
      live.delete(buffer);
      return result;
    });
    return buffer;
  });
  for (const name of ['createComputePipeline', 'createComputePipelineAsync'])
    wrap(gpu, name, (fn, self, args) => {
      native.pipelines++;
      return Reflect.apply(fn, self, args);
    });
  wrap(GPUComputePassEncoder.prototype, 'dispatchWorkgroups', (fn, self, args) => {
    native.dispatches++;
    return Reflect.apply(fn, self, args);
  });
  wrap(GPUBuffer.prototype, 'mapAsync', (fn, self, args) => {
    if (args[0] & GPUMapMode.READ) native.readbacks++;
    return Reflect.apply(fn, self, args);
  });
  wrap(gpu, 'destroy', (fn, self, args) => {
    const result = Reflect.apply(fn, self, args);
    native.deviceDestroyCalls++;
    for (const row of live.values()) row.invalidatedByDevice = true;
    return result;
  });
  return {
    phase(value) {
      phase = value;
    },
    snapshot(name, statistics) {
      snapshots.push({
        name,
        retained: [...live.values()].map((row) => row.id),
        liveBuffers: statistics?.liveBuffers ?? 0,
        liveBytes: statistics?.liveBytes ?? 0,
      });
    },
    result: { native, records, snapshots },
    restore() {
      for (const reset of restore.reverse()) reset();
    },
  };
}

export async function runIndependentPhotometricGpu({ hardware, legacyUrl, input, inputSHA }) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const description = [adapter.info?.vendor, adapter.info?.architecture, adapter.info?.description]
    .filter(Boolean)
    .join(' ');
  if (hardware && (/swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter))
    throw new Error(`Hardware requested, selected ${description}`);
  const inputHash = [
    ...new Uint8Array(
      await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(input))),
    ),
  ]
    .map((v) => v.toString(16).padStart(2, '0'))
    .join('');
  if (inputHash !== inputSHA) throw new Error('Synthetic scene/RGBA input hash mismatch');
  const scene = {
      ...input,
      images: new Map(
        input.images.map(([id, image]) => [id, { ...image, rgba: Uint8Array.from(image.rgba) }]),
      ),
    },
    rows = [],
    lifetimes = [];
  const groups = [
    {
      url: '/dist/internal/gpu.js',
      cases: [
        ['default', {}],
        ['off', { independentRefinement: false }],
        ['off-stream', { independentRefinement: false, streamGroup: 1 }],
        ['on', { independentRefinement: true }],
        ['on-scalar-pose', { independentRefinement: true, relativePoseCache: false }],
        ['on-stream', { independentRefinement: true, streamGroup: 1 }],
      ],
    },
    ...(legacyUrl
      ? [
          {
            url: legacyUrl,
            cases: [
              ['frozen-legacy', {}],
              ['frozen-legacy-stream', { streamGroup: 1 }],
            ],
          },
        ]
      : []),
  ];
  for (const group of groups) {
    // Each module creates its own ShadeDevice and borrows its own native GPUDevice.
    const api = await import(group.url),
      gpu = await adapter.requestDevice();
    const spy = instrument(gpu),
      validation = [],
      uncaptured = [],
      lost = [];
    gpu.addEventListener('uncapturederror', (event) => uncaptured.push(event.error.message));
    const settledLoss = gpu.lost.then((info) => {
      if (info.reason !== 'destroyed') lost.push(info.message);
    });
    let device;
    try {
      device = await api.ShadeDevice.open({ backend: 'webgpu', gpuDevice: gpu, poolBytes: 0 });
      const plan = api.planMvsViews(scene),
        reference = plan.views.find((v) => v.id === 2).index;
      const plain = ({ depth, normal, cost }) => ({
        depth: Array.from(depth),
        normal: Array.from(normal),
        cost: Array.from(cost),
      });
      const stageHash = async ({ depth, normal, cost }) => {
        const bytes = new Uint8Array(depth.byteLength + normal.byteLength + cost.byteLength);
        let at = 0;
        for (const array of [depth, normal, cost]) {
          bytes.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength), at);
          at += array.byteLength;
        }
        return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
          .map((v) => v.toString(16).padStart(2, '0'))
          .join('');
      };
      for (const [name, options] of group.cases) {
        spy.phase(name);
        for (const scope of ['out-of-memory', 'internal', 'validation']) gpu.pushErrorScope(scope);
        try {
          const result = await api.mvsPhotometricGpu(device, scene, plan, scene.images, reference, {
            snapshots: true,
            ...options,
          });
          await gpu.queue.onSubmittedWorkDone();
          rows.push({
            name,
            final: plain(result),
            ...(name === 'on' ? { early: result.snapshots.slice(0, 2).map(plain) } : {}),
            snapshots: await Promise.all(result.snapshots.map(stageHash)),
            residency: result.residency,
          });
          spy.snapshot(name, device.statistics);
        } finally {
          for (const scope of ['validation', 'internal', 'out-of-memory']) {
            const error = await gpu.popErrorScope();
            if (error) validation.push({ name, scope, message: error.message });
          }
        }
      }
      await gpu.queue.onSubmittedWorkDone();
    } finally {
      try {
        await device?.dispose();
      } finally {
        spy.snapshot('after-shade-dispose', device?.statistics);
        gpu.destroy();
        await settledLoss;
        spy.snapshot('after-native-device-destroy', device?.statistics);
        lifetimes.push({ module: group.url, ...spy.result, validation, uncaptured, lost });
        spy.restore();
      }
    }
  }
  return { description, fallback: adapter.isFallbackAdapter, inputSHA: inputHash, rows, lifetimes };
}
