import test from 'node:test';
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { inBrowser } from './browser.mjs';
import {
  strictWitnessFixture,
  assertStrictWitnessCases,
} from './observation-strict-witness-fixture.mjs';

function assertStrictWitnessNativeLifetime(native, caseNames) {
  assert.equal(native.createdBuffers, native.buffers.length);
  assert.equal(new Set(native.buffers.map((buffer) => buffer.id)).size, native.createdBuffers);
  const cases = native.snapshots.filter((snapshot) => snapshot.name.startsWith('after-case:'));
  assert.deepEqual(
    cases.map((snapshot) => snapshot.name),
    caseNames.map((name) => `after-case:${name}`),
  );
  const plain = cases[0];
  assert.equal(plain.name, 'after-case:plain');
  assert.equal(plain.retainedWithoutExplicitBufferDestroyIds.length, 1);
  const ringId = plain.retainedWithoutExplicitBufferDestroyIds[0],
    ring = native.buffers.find((buffer) => buffer.id === ringId);
  assert.ok(ring, 'the baseline retained identity has a captured creation descriptor');
  assert.equal(ring.createdInPhase, 'case:plain');
  assert.equal(ring.size, 4096);
  assert.equal(ring.usage, 72, 'GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST');
  assert.equal(ring.label, '');
  assert.equal(ring.mappedAtCreation, false);
  assert.match(ring.creationStack, /\bat ringFor\b/);
  assert.match(ring.creationStack, /\bat uniformsOf\b/);
  assert.equal(ring.explicitlyDestroyed, false);
  assert.equal(ring.explicitDestroyCalls, 0);
  for (const snapshot of cases) {
    assert.deepEqual(snapshot.retainedWithoutExplicitBufferDestroyIds, [ringId], snapshot.name);
    assert.equal(snapshot.retainedWithoutExplicitBufferDestroyBytes, 4096, snapshot.name);
    assert.equal(snapshot.publicDeviceStatistics.liveBuffers, 0, snapshot.name);
    assert.equal(snapshot.publicDeviceStatistics.liveBytes, 0, snapshot.name);
    assert.equal(snapshot.gpuDeviceDestroyCalled, false, snapshot.name);
  }
  for (const name of ['before-shade-dispose', 'after-shade-dispose']) {
    const snapshot = native.snapshots.find((row) => row.name === name);
    assert.ok(snapshot, name);
    assert.deepEqual(snapshot.retainedWithoutExplicitBufferDestroyIds, [ringId], name);
    assert.equal(snapshot.retainedWithoutExplicitBufferDestroyBytes, 4096, name);
    assert.equal(snapshot.publicDeviceStatistics.liveBuffers, 0, name);
    assert.equal(snapshot.publicDeviceStatistics.liveBytes, 0, name);
    assert.equal(snapshot.gpuDeviceDestroyCalled, false, name);
  }
  const otherBuffers = native.buffers.filter((buffer) => buffer.id !== ringId);
  assert.ok(
    otherBuffers.every((buffer) => buffer.explicitlyDestroyed && buffer.explicitDestroyCalls > 0),
  );
  assert.equal(native.destroyedBuffers, otherBuffers.length);
  assert.equal(native.createdBuffers, native.destroyedBuffers + 1);
  assert.equal(native.gpuDeviceDestroy.called, true);
  assert.equal(native.gpuDeviceDestroy.calls, 1);
  assert.ok(Number.isSafeInteger(ring.logicalDeviceInvalidationSequence));
  assert.ok(ring.logicalDeviceInvalidationSequence > ring.creationSequence);
  assert.deepEqual(native.gpuDeviceDestroy.logicalInvalidations, [
    { id: ringId, sequence: ring.logicalDeviceInvalidationSequence },
  ]);
  const afterDevice = native.snapshots.find(
    (snapshot) => snapshot.name === 'after-gpu-device-destroy',
  );
  assert.ok(afterDevice);
  assert.equal(afterDevice.gpuDeviceDestroyCalled, true);
  assert.deepEqual(afterDevice.retainedWithoutExplicitBufferDestroyIds, [ringId]);
  assert.deepEqual(afterDevice.logicallyLiveBufferIds, []);
}

test('strict original witnesses and existing-feature provenance use actual WebGPU measurements', async () => {
  const input = strictWitnessFixture(),
    hardware = process.env.TYPECV_HARDWARE_GPU === '1';
  const { result, errors } = await inBrowser(
    async ({ input, hardware }) => {
      const api = await import('/dist/gpu.js'),
        { runStrictWitnessCases } =
          await import('/tests/typeshade/observation-strict-witness-cases.mjs'),
        adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter');
      const description = [adapter.info.vendor, adapter.info.architecture, adapter.info.description]
        .filter(Boolean)
        .join(' ');
      if (
        hardware &&
        (/swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter)
      )
        throw new Error(`Hardware adapter requested, selected ${description}`);
      const gpu = await adapter.requestDevice(),
        validation = [],
        uncaptured = [],
        lost = [],
        buffers = new Map(),
        restore = [],
        native = {
          pipelines: 0,
          dispatches: 0,
          readbacks: 0,
          createdBuffers: 0,
          destroyedBuffers: 0,
          buffers: [],
          snapshots: [],
          gpuDeviceDestroy: { called: false, calls: 0, logicalInvalidations: [] },
        };
      let device,
        phase = 'before-shade-open',
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
        restore.push(() => {
          if (own) Object.defineProperty(object, name, own);
          else delete object[name];
        });
      };
      const snapshot = (name) => {
        const retained = [...buffers.values()],
          statistics = device?.statistics;
        native.snapshots.push({
          name,
          retainedWithoutExplicitBufferDestroyIds: retained.map((record) => record.id),
          retainedWithoutExplicitBufferDestroyBytes: retained.reduce(
            (sum, record) => sum + record.size,
            0,
          ),
          logicallyLiveBufferIds: retained
            .filter((record) => record.logicalDeviceInvalidationSequence === null)
            .map((record) => record.id),
          publicDeviceStatistics: statistics
            ? {
                liveBuffers: statistics.liveBuffers,
                liveBytes: statistics.liveBytes,
                pooledBytes: statistics.pooledBytes,
              }
            : null,
          gpuDeviceDestroyCalled: native.gpuDeviceDestroy.called,
        });
      };
      for (const method of ['createComputePipeline', 'createComputePipelineAsync'])
        wrap(gpu, method, (fn, self, args) => {
          native.pipelines++;
          return Reflect.apply(fn, self, args);
        });
      wrap(gpu, 'createBuffer', (fn, self, args) => {
        const descriptor = args[0],
          creationStack = new Error('GPUDevice.createBuffer creation stack').stack,
          buffer = Reflect.apply(fn, self, args),
          record = {
            id: `buffer-${String(++native.createdBuffers).padStart(4, '0')}`,
            size: Number(descriptor.size),
            usage: Number(descriptor.usage),
            label: descriptor.label ?? '',
            mappedAtCreation: descriptor.mappedAtCreation ?? false,
            creationStack,
            createdInPhase: phase,
            creationSequence: ++sequence,
            explicitDestroyCalls: 0,
            explicitlyDestroyed: false,
            logicalDeviceInvalidationSequence: null,
          };
        native.buffers.push(record);
        buffers.set(buffer, record);
        wrap(buffer, 'destroy', (fn, self, args) => {
          record.explicitDestroyCalls++;
          const result = Reflect.apply(fn, self, args);
          if (buffers.delete(buffer)) {
            record.explicitlyDestroyed = true;
            native.destroyedBuffers++;
          }
          return result;
        });
        return buffer;
      });
      wrap(gpu, 'destroy', (fn, self, args) => {
        native.gpuDeviceDestroy.called = true;
        native.gpuDeviceDestroy.calls++;
        const destroySequence = ++sequence,
          result = Reflect.apply(fn, self, args);
        // Device invalidation is distinct from explicit buffer.destroy() calls. This
        // records logical object lifetime, not a physical driver freeing timestamp.
        for (const record of buffers.values()) {
          record.logicalDeviceInvalidationSequence = destroySequence;
          native.gpuDeviceDestroy.logicalInvalidations.push({
            id: record.id,
            sequence: destroySequence,
          });
        }
        return result;
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
      try {
        phase = 'shade-open';
        device = await api.ShadeDevice.open({ backend: 'webgpu', gpuDevice: gpu, poolBytes: 0 });
        gpu.pushErrorScope('out-of-memory');
        gpu.pushErrorScope('internal');
        gpu.pushErrorScope('validation');
        let cases;
        try {
          phase = 'case:plain';
          cases = await runStrictWitnessCases(api, device, input, {
            onCase: async (name) => {
              await gpu.queue.onSubmittedWorkDone();
              snapshot(`after-case:${name}`);
              const names = Object.keys(input.cases),
                next = names[names.indexOf(name) + 1];
              phase = next ? `case:${next}` : 'after-all-cases';
            },
          });
          await gpu.queue.onSubmittedWorkDone();
        } finally {
          for (const scope of ['validation', 'internal', 'out-of-memory']) {
            const error = await gpu.popErrorScope();
            if (error) validation.push({ scope, message: error.message });
          }
        }
        return {
          ...cases,
          backend: device.backend,
          adapter: { description, fallback: adapter.isFallbackAdapter },
          validation,
          uncaptured,
          lost,
          native,
        };
      } finally {
        phase = 'before-shade-dispose';
        snapshot('before-shade-dispose');
        await device?.dispose();
        phase = 'after-shade-dispose';
        snapshot('after-shade-dispose');
        phase = 'gpu-device-destroy';
        gpu.destroy();
        phase = 'after-gpu-device-destroy';
        snapshot('after-gpu-device-destroy');
        for (const reset of restore.reverse()) reset();
      }
    },
    { input, hardware },
    { timeoutMs: 180000 },
  );
  if (process.env.TYPECV_STRICT_WITNESS_RESULT)
    await writeFile(
      process.env.TYPECV_STRICT_WITNESS_RESULT,
      JSON.stringify({ result, errors }, null, 2) + '\n',
    );
  assert.deepEqual(errors, []);
  assert.equal(result.backend, 'webgpu');
  assert.deepEqual(result.validation, []);
  assert.deepEqual(result.uncaptured, []);
  assert.deepEqual(result.lost, []);
  assertStrictWitnessNativeLifetime(result.native, Object.keys(input.cases));
  assert.ok(result.native.pipelines > 0);
  assert.ok(result.native.dispatches > 0);
  assert.ok(result.native.readbacks > 0);
  assertStrictWitnessCases(result, input);
  console.log(
    JSON.stringify({
      strictWitnessWebGPU: {
        cases: Object.fromEntries(
          Object.entries(result.cases).map(([name, value]) => [name, value.newEdges.length]),
        ),
        adapter: result.adapter,
        native: {
          createdBuffers: result.native.createdBuffers,
          destroyedBuffers: result.native.destroyedBuffers,
          pipelines: result.native.pipelines,
          dispatches: result.native.dispatches,
          readbacks: result.native.readbacks,
          ringId: result.native.snapshots[0].retainedWithoutExplicitBufferDestroyIds[0],
          gpuDeviceDestroy: result.native.gpuDeviceDestroy,
        },
      },
    }),
  );
});
