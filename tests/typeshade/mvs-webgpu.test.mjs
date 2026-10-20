// Verifies: docs/dense-mvs.md step 3 (sections 5.1, 5.2 and 7.2): the photometric pass on WebGPU
// agrees with the CPU tier, and the SwiftShader rate of each chunk key is recorded. Verifies:
// docs/dense-mvs.md step 5 (sections 4.2, 7.2 and 7.5): the full `reconstructDenseGpu` path, the
// geometric pass, the filter and the fusion included, agrees with the CPU tier, and the memory
// formula holds on WebGPU.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  decodeMvsNormal,
  mvsPhotometricGpu,
  planMvsViews,
} from '../../dist/internal/gpu.js';
import { ShadeDevice as PublicDevice, reconstructDenseGpu } from '../../dist/gpu.js';
import { depthMetrics, fusionMetrics, memoryFormula, mvsScene } from './mvs-cases.mjs';
import { inBrowser } from './browser.mjs';

const REFERENCE_ID = 2;
/** The units of work of one chunk unit (section 5.1): window samples per pixel and source. */
const SAMPLES_PER_UNIT = { mvsInitialize: 36, mvsPropagate: 10 * 36 };
/**
 * The configuration of the full path: 2 iterations of each pass, so that the CPU tier takes
 * about half a minute. The test checks that the kernels agree, not the accuracy of 7.1.
 */
const DENSE = { photometricIterations: 2, geometricIterations: 2 };

test('mvsInitialize and mvsPropagate on WebGPU: agreement with the CPU tier and the chunk rates', async () => {
  const scene = mvsScene();
  const plan = planMvsViews(scene);
  const reference = plan.views[REFERENCE_ID].index;
  const range = plan.views[REFERENCE_ID].depthRange;
  const sources = plan.sources[reference * 7];

  const cpuDevice = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  let cpu;
  try {
    cpu = await mvsPhotometricGpu(cpuDevice, scene, plan, scene.images, reference, {
      snapshots: true,
    });
  } finally {
    await cpuDevice.dispose();
  }

  // The scene is built again in the page from the same module, so that only numbers cross.
  const { result, errors } = await inBrowser(
    async (referenceId) => {
      const { ShadeDevice, mvsPhotometricGpu, planMvsViews } =
        await import('/dist/internal/gpu.js');
      const { mvsScene } = await import('/tests/typeshade/mvs-cases.mjs');
      const scene = mvsScene();
      const plan = planMvsViews(scene);
      const reference = plan.views[referenceId].index;
      const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      const plain = (s) => ({
        depth: Array.from(s.depth),
        normal: Array.from(s.normal),
        cost: Array.from(s.cost),
      });
      try {
        const started = performance.now();
        const result = await mvsPhotometricGpu(device, scene, plan, scene.images, reference, {
          snapshots: true,
        });
        const ms = performance.now() - started;
        const calls = (name) =>
          result.report.groups.flatMap((group) => group.calls).filter((c) => c === name).length;
        return {
          ms,
          final: plain(result),
          // The initialization and the first iteration: before the tiers' choices can part.
          early: result.snapshots.slice(0, 2).map(plain),
          rates: device.chunkRates,
          timeBudgetMs: device.timeBudgetMs,
          calls: { mvsInitialize: calls('mvsInitialize'), mvsPropagate: calls('mvsPropagate') },
          liveBytes: device.statistics.liveBytes,
        };
      } finally {
        await device.dispose();
      }
    },
    REFERENCE_ID,
    { timeoutMs: 180000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.liveBytes, 0);

  // Agreement on the pixels kept on both tiers ("kept" as in 7.1: the photometric rules of 3.9).
  const kept = (r, i) => r.cost[i] <= 0.9 && r.depth[i] >= range[0] && r.depth[i] <= range[1];
  const agreement = (gpu, host) => {
    const relative = [],
      costs = [];
    for (let i = 0; i < host.depth.length; i++) {
      if (!(kept(gpu, i) && kept(host, i))) continue;
      const e = Math.abs(gpu.depth[i] - host.depth[i]) / host.depth[i];
      relative.push(e);
      if (e > 1e-3) costs.push(Math.abs(gpu.cost[i] - host.cost[i]));
    }
    relative.sort((a, b) => a - b);
    costs.sort((a, b) => a - b);
    const share = (bound) => relative.filter((e) => e <= bound).length / relative.length;
    return {
      keptBoth: relative.length,
      within1e3: share(1e-3),
      within1e2: share(1e-2),
      median: relative[Math.floor(relative.length / 2)],
      max: relative[relative.length - 1],
      // The cost difference of the pixels that part by more than 1e-3.
      partedMedianCostDifference: costs.length ? costs[Math.floor(costs.length / 2)] : 0,
    };
  };
  const [init, first] = result.early.map((s, t) => agreement(s, cpu.snapshots[t]));
  const final = agreement(result.final, cpu);
  const truth = (r) =>
    depthMetrics(r, scene.truth[REFERENCE_ID], plan.normalization.scale, range, decodeMvsNormal);
  const accuracy = { webgpu: truth(result.final), cpu: truth(cpu) };
  const rates = Object.fromEntries(
    Object.entries(result.rates).map(([key, rate]) => [
      key,
      {
        ...rate,
        msPerUnit: +rate.msPerUnit.toPrecision(4),
        samplesPerSecond: Math.round((SAMPLES_PER_UNIT[key] / rate.msPerUnit) * 1000),
      },
    ]),
  );
  console.log(
    JSON.stringify({
      ms: Math.round(result.ms),
      sources,
      timeBudgetMs: result.timeBudgetMs,
      agreement: { init, first, final },
      accuracy,
      calls: result.calls,
      rates,
    }),
  );

  // Section 7.2, as measured in step 3 (docs/dense-mvs.md, "Step 3 record"). Transcendentals and
  // fused multiply-adds round differently on the GPU, so the tiers agree within 1e-3 before
  // propagation can part them, and every pixel kept on both stays within 1e-3 at the
  // initialization. After one iteration at least 99.9 % do.
  assert.ok(init.max <= 1e-3, `initialization: largest relative difference ${init.max}`);
  assert.ok(first.within1e3 >= 0.999, `iteration 1: ${first.within1e3} within 1e-3`);
  // A last-bit difference in a cost can flip the choice between two candidates of near-equal
  // cost, and the propagation carries the choice to the neighbours. After 8 iterations the tiers
  // still agree within 1e-2 on at least 99 % of the pixels, and within 1e-3 on at least 95 %.
  assert.ok(final.within1e2 >= 0.99, `final: ${final.within1e2} within 1e-2`);
  assert.ok(final.within1e3 >= 0.95, `final: ${final.within1e3} within 1e-3`);
  // The parted pixels hold hypotheses of near-equal cost, not a wrong kernel.
  assert.ok(final.partedMedianCostDifference <= 0.01);
  // The accuracy bounds of 7.1 hold on WebGPU, and the two tiers' accuracy is the same.
  assert.ok(accuracy.webgpu.median <= 0.01, `median ${accuracy.webgpu.median}`);
  assert.ok(accuracy.webgpu.kept >= 0.8, `kept ${accuracy.webgpu.kept}`);
  assert.ok(accuracy.webgpu.medianNormalDeg <= 16, `normal ${accuracy.webgpu.medianNormalDeg}°`);
  assert.ok(Math.abs(accuracy.webgpu.median - accuracy.cpu.median) <= 0.0005);
  assert.ok(Math.abs(accuracy.webgpu.kept - accuracy.cpu.kept) <= 0.01);

  // Section 5.1: both keys ran as timed chunks, more than one a pass.
  for (const [key, passes] of [
    ['mvsInitialize', 1],
    ['mvsPropagate', 16],
  ]) {
    const rate = result.rates[key];
    assert.ok(rate && rate.chunks >= 2 && rate.msPerUnit > 0, `${key} has a rate`);
    assert.ok(result.calls[key] > passes + 1, `${key} ran as more than one chunk a pass`);
  }
  // The warm step ran once, before the first timed chunk: one call of the kernel more than its
  // chunks, all of them timed while fewer than `chunkTimingSamples` (8).
  const initialize = result.rates.mvsInitialize;
  if (initialize.chunks < 8) assert.equal(result.calls.mvsInitialize, initialize.chunks + 1);
  // The CPU tier runs each pass as one chunk.
  const cpuCalls = (name) =>
    cpu.report.groups.flatMap((group) => group.calls).filter((c) => c === name).length;
  assert.equal(cpuCalls('mvsInitialize'), 1);
  assert.equal(cpuCalls('mvsPropagate'), 16);
});

/**
 * The agreement of two point clouds, reference by reference: the share of `a`'s points with a
 * point of `b` of the same reference within 1e-3 and within 1e-2 of the point's depth in that
 * reference.
 */
function cloudAgreement(scene, a, b) {
  const views = new Map(scene.views.map((view) => [view.id, view]));
  const shares = { within1e3: 0, within1e2: 0 };
  for (let v = 0; v < a.views.length; v++) {
    const view = views.get(a.views[v].id),
      { R, t } = view.pose;
    const depth = (p) => R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2];
    const slice = (cloud) =>
      Array.from({ length: cloud.views[v].points }, (_, k) =>
        Array.from(
          cloud.points.positions.slice(
            3 * (cloud.views[v].firstPoint + k),
            3 * (cloud.views[v].firstPoint + k) + 3,
          ),
        ),
      );
    const ours = slice(a),
      theirs = slice(b);
    // Bins of 0.05 units are wider than 1e-2 of a depth (the depths are about 3.3), so a
    // 3 × 3 × 3 block holds every candidate.
    const cell = 0.05;
    const key = (p) => p.map((c) => Math.floor(c / cell)).join(',');
    const bins = new Map();
    for (const p of theirs) {
      const k = key(p);
      if (!bins.has(k)) bins.set(k, []);
      bins.get(k).push(p);
    }
    for (const p of ours) {
      const [i, j, k] = p.map((c) => Math.floor(c / cell));
      let nearest = Infinity;
      for (let di = -1; di <= 1; di++)
        for (let dj = -1; dj <= 1; dj++)
          for (let dk = -1; dk <= 1; dk++)
            for (const q of bins.get(`${i + di},${j + dj},${k + dk}`) ?? [])
              nearest = Math.min(nearest, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
      const relative = nearest / depth(p);
      if (relative <= 1e-3) shares.within1e3++;
      if (relative <= 1e-2) shares.within1e2++;
    }
  }
  return {
    within1e3: shares.within1e3 / a.points.count,
    within1e2: shares.within1e2 / a.points.count,
  };
}

test('reconstructDenseGpu on WebGPU: the full path agrees with the CPU tier, and memory follows 4.2', async () => {
  const scene = mvsScene();
  const plan = planMvsViews(scene);
  const P = 128 * 96;
  const cpuDevice = await PublicDevice.open({ backend: 'cpu', poolBytes: 0 });
  let cpu, cpuMs;
  try {
    const started = performance.now();
    cpu = await reconstructDenseGpu(cpuDevice, scene, scene.images, DENSE);
    cpuMs = performance.now() - started;
  } finally {
    await cpuDevice.dispose();
  }

  const { result, errors } = await inBrowser(
    async (options) => {
      const { ShadeDevice, reconstructDenseGpu } = await import('/dist/gpu.js');
      const { mvsScene } = await import('/tests/typeshade/mvs-cases.mjs');
      const scene = mvsScene();
      const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      const active = () => device.statistics.liveBytes - device.statistics.pooledBytes;
      const memory = { depth: [], points: [] };
      try {
        const started = performance.now();
        const result = await reconstructDenseGpu(device, scene, scene.images, {
          ...options,
          onView: (event) => memory[event.type].push(active()),
        });
        const ms = performance.now() - started;
        return {
          ms,
          views: result.views,
          points: {
            count: result.points.count,
            positions: Array.from(result.points.positions),
            normals: Array.from(result.points.normals),
            colors: Array.from(result.points.colors),
          },
          memory,
          rates: device.chunkRates,
          liveBytes: device.statistics.liveBytes,
        };
      } finally {
        await device.dispose();
      }
    },
    DENSE,
    { timeoutMs: 240000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.liveBytes, 0);
  const gpu = {
    views: result.views,
    points: {
      count: result.points.count,
      positions: Float32Array.from(result.points.positions),
      normals: Int16Array.from(result.points.normals),
      colors: Uint8Array.from(result.points.colors),
    },
  };
  const scale = plan.normalization.scale;
  const agreement = {
    gpuInCpu: cloudAgreement(scene, gpu, cpu),
    cpuInGpu: cloudAgreement(scene, cpu, gpu),
  };
  const accuracy = {
    webgpu: fusionMetrics(scene, gpu, scale, decodeMvsNormal),
    cpu: fusionMetrics(scene, cpu, scale, decodeMvsNormal),
  };
  const perView = gpu.views.map((view, v) => ({
    id: view.id,
    depthPixels: [view.depthPixels, cpu.views[v].depthPixels],
    points: [view.points, cpu.views[v].points],
  }));
  const memory = memoryFormula(plan, P);
  console.log(
    JSON.stringify({
      ms: Math.round(result.ms),
      cpuMs: Math.round(cpuMs),
      perView,
      agreement,
      accuracy,
      memory: { ...memory, webgpu: result.memory },
      rates: Object.fromEntries(
        Object.entries(result.rates).map(([key, rate]) => [
          key,
          { ...rate, msPerUnit: +rate.msPerUnit.toPrecision(4) },
        ]),
      ),
    }),
  );

  // 4.2 on WebGPU: the device holds the bytes of each stage's formula and its small arrays.
  for (const bytes of result.memory.depth) assert.equal(bytes, memory.depthStage);
  for (const bytes of result.memory.points) assert.equal(bytes, memory.fusionStage);
  // 7.2 for the full path, as measured in step 5 ("Step 5 record"): the kept pixels of the
  // filtered maps and the points of each reference agree within 2 % of their counts, and most
  // points of each tier have a point of the other within 1e-3 of their depth.
  for (const view of perView) {
    const [g, c] = view.depthPixels;
    assert.ok(Math.abs(g - c) <= 0.02 * c, `view ${view.id} depth pixels ${g} ${c}`);
  }
  assert.ok(Math.abs(gpu.points.count - cpu.points.count) <= 0.02 * cpu.points.count);
  for (const share of [agreement.gpuInCpu, agreement.cpuInGpu]) {
    assert.ok(share.within1e3 >= 0.9, `within 1e-3: ${share.within1e3}`);
    assert.ok(share.within1e2 >= 0.98, `within 1e-2: ${share.within1e2}`);
  }
  // The two tiers' accuracy is the same.
  assert.ok(Math.abs(accuracy.webgpu.within1pct - accuracy.cpu.within1pct) <= 0.02);
  assert.ok(Math.abs(accuracy.webgpu.duplicates - accuracy.cpu.duplicates) <= 0.02);
  assert.ok(Math.abs(accuracy.webgpu.medianNormalDeg - accuracy.cpu.medianNormalDeg) <= 0.5);
  assert.ok(accuracy.webgpu.colorMatch >= 0.99);
  // Section 5.1: the keys of the geometric pass, the filter and the fusion ran as timed chunks.
  for (const key of ['mvsGeometricInitialize', 'mvsPropagateGeometric', 'mvsFilter', 'mvsFuse'])
    assert.ok(result.rates[key]?.msPerUnit > 0, `${key} has a rate`);
});
