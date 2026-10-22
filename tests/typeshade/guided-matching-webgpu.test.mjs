import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { buildGuidedProbe } from './guided-probe-build.mjs';
import { floatDescriptorFixture } from './float-matching-cases.mjs';

const NONE = 0xffffffff;
// Lines through the epipole (320, 180): F = [e]x, and Fᵀ = -F maps the other way.
const radial = [0, -1, 180, 1, 0, -320, -180, 320, 0];
const transpose = (m) => [0, 3, 6, 1, 4, 7, 2, 5, 8].map((i) => m[i]);
// A pure horizontal shift: the epipolar lines are the image rows.
const rows = [0, 0, 0, 0, 0, -1, 0, 1, 0];
// Vertical lines, across a horizontal shift.
const columns = [0, 0, -1, 0, 0, 0, 1, 0, 0];

function points(count, seed) {
  let state = seed >>> 0;
  const next = () => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32;
  return Array.from({ length: count }, () => [Math.round(next() * 2560) / 4, next() * 360]).flat();
}

/**
 * Raw fixtures for the probe: query and candidate counts that are not multiples of the 8 × 8
 * tile, channel counts that do not fill a slab, duplicated candidates (exact ties), a
 * query at the epipole (a degenerate line), an infinite channel on one candidate, and 200
 * channels, which take two 128-channel slabs.
 */
function randomFixtures() {
  const fixtures = [];
  for (const [n, m, dimensions, seed] of [
    [37, 45, 128, 3],
    [5, 3, 33, 5],
    [64, 61, 17, 7],
    [9, 1, 128, 11],
    [70, 13, 1, 13],
    [11, 19, 200, 17],
  ]) {
    const queries = floatDescriptorFixture(n, dimensions, seed),
      candidates = floatDescriptorFixture(m, dimensions, seed + 100),
      queryPoints = points(n, seed + 200),
      candidatePoints = points(m, seed + 300);
    // Near correspondences, so that queries have a clear nearest candidate.
    for (let i = 0; i < Math.min(n, m); i += 2)
      for (let d = 0; d < dimensions; d++)
        candidates[i * dimensions + d] = queries[i * dimensions + d] + 0.01;
    // Candidate 1 repeats candidate 0 at the same point: an exact tie, broken by the lower index.
    if (m > 2) {
      candidates.copyWithin(dimensions, 0, dimensions);
      candidatePoints[2] = candidatePoints[0];
      candidatePoints[3] = candidatePoints[1];
    }
    queryPoints[0] = 320;
    queryPoints[1] = 180;
    for (const infinite of m > 4 ? [false, true] : [false]) {
      const c = Float32Array.from(candidates);
      if (infinite) c[4 * dimensions + (dimensions >> 1)] = Infinity;
      for (const [name, matrix] of [
        ['radial', radial],
        ['radialT', transpose(radial)],
        ['rows', rows],
      ])
        for (const maxLinePx of [0.5, 2, 8, 1e4])
          fixtures.push({
            name: `${n}x${m}x${dimensions} ${name} ${maxLinePx}px${infinite ? ' inf' : ''}`,
            queries: [...queries],
            candidates: [...c],
            dimensions,
            queryPoints,
            candidatePoints,
            matrix,
            maxLinePx,
          });
    }
  }
  // Candidates exactly on the edge of the band count: the gate is `distance <= maxLinePx`.
  const offsets = [0, 0.5, -0.5, 2, -2, 8, -8, 0.25, 3, -9, 12];
  for (const maxLinePx of [0.5, 2, 8])
    fixtures.push({
      name: `band edge ${maxLinePx}px`,
      queries: [0, 0.5, 3],
      candidates: offsets.map((_, j) => j * 0.375),
      dimensions: 1,
      queryPoints: [10, 100, 300, 100, 600, 100],
      candidatePoints: offsets.flatMap((offset, j) => [j * 50, 100 + offset]),
      matrix: rows,
      maxLinePx,
    });
  return fixtures;
}

function summarize(fixture, scalar) {
  const n = fixture.queries.length / fixture.dimensions,
    m = fixture.candidates.length / fixture.dimensions;
  const count = (f) => Array.from({ length: n }, (_, i) => f(i)).filter(Boolean).length;
  return {
    matched: count((i) => scalar.indices[i * 3] !== NONE),
    sole: count((i) => scalar.indices[i * 3 + 1] === m),
    invalid: count((i) => scalar.indices[i * 3 + 2] !== 0),
  };
}

test('WebGPU guided tiles equal the scalar guided kernel bit for bit, and SIFT guided matching holds', async () => {
  const probe = await buildGuidedProbe();
  const fixtures = randomFixtures();
  const { result, errors } = await inBrowser(
    async ({ fixtures, hardware, rows, columns }) => {
      const probe = await import('/probe/guided-probe.js');
      const { ShadeDevice, extractSiftGpuResident, matchFloatDescriptorsGpu } =
        await import('/dist/gpu.js');
      const { texture, warped } = await import('/tests/typeshade/sift-cases.mjs');
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('WebGPU adapter unavailable');
      const info = adapter.info;
      const adapterDescription = [info?.vendor, info?.architecture, info?.device, info?.description]
        .filter(Boolean)
        .join(' ');
      const software =
        /swiftshader|software|llvmpipe/i.test(adapterDescription) ||
        adapter.isFallbackAdapter === true;
      if (hardware && software) throw new Error('Hardware requested but software adapter selected');
      const gpu = await adapter.requestDevice(),
        gpuErrors = [];
      gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      const kernels = await probe.ShadeDevice.open({ gpuDevice: gpu });
      const run = async (fixture) => ({
        scalar: await probe.nearestGuided(kernels, 'scalar', fixture),
        tiles: await probe.nearestGuided(kernels, 'tiles', fixture),
      });
      const random = [];
      for (const fixture of fixtures) random.push(await run(fixture));
      // SIFT on a 14 px horizontal shift, as the CPU-tier test in sift.test.mjs.
      const device = await ShadeDevice.open({ gpuDevice: gpu });
      const width = 256,
        height = 192,
        gray = texture(width, height),
        shift = 14;
      const first = await extractSiftGpuResident(
        device,
        warped(gray, width, height, width, height, 1, 0, 0, 0),
        width,
        height,
        { maxFeatures: 1024 },
      );
      const second = await extractSiftGpuResident(
        device,
        warped(gray, width, height, width, height, 1, 0, shift, 0),
        width,
        height,
        { maxFeatures: 1024 },
      );
      const a = await first.readKeypoints(),
        b = await second.readKeypoints();
      const [firstSnapshot, secondSnapshot] = await device.enqueue(() =>
        Promise.all([
          device.read(first.descriptors.descriptors, first.descriptors.elements),
          device.read(second.descriptors.descriptors, second.descriptors.elements),
        ]),
      );
      // Resident.read returns the full allocation; only the descriptor's live prefix is input.
      const da = firstSnapshot.subarray(0, first.descriptors.elements),
        db = secondSnapshot.subarray(0, second.descriptors.elements);
      const sift = [];
      for (const [name, matrix, maxLinePx] of [
        ['rows', rows, 2],
        ['rows unbounded', rows, 1e4],
        ['columns', columns, 2],
      ])
        for (const backward of [false, true]) {
          const fixture = {
            queries: Array.from(backward ? db : da),
            candidates: Array.from(backward ? da : db),
            dimensions: 128,
            queryPoints: Array.from(backward ? b : a),
            candidatePoints: Array.from(backward ? a : b),
            matrix: backward ? [0, 3, 6, 1, 4, 7, 2, 5, 8].map((i) => matrix[i]) : matrix,
            maxLinePx,
          };
          const raw = await run(fixture);
          const specialized = await probe.measureGuidedWorkgroups(gpu, fixture, {
            samples: 1,
            warmup: 1,
          });
          sift.push({
            name: `${name}${backward ? ' backward' : ''}`,
            queries: fixture.queries.length / 128,
            candidates: fixture.candidates.length / 128,
            ...raw,
            specialized,
          });
        }
      const match = (options) =>
        matchFloatDescriptorsGpu(device, first.descriptors, second.descriptors, {
          ratio: 0.8,
          ...options,
        });
      const epipolar = { fundamental: rows, firstPoints: a, secondPoints: b, maxLinePx: 2 };
      const plain = await match({}),
        guided = await match({ epipolar }),
        tiledKernel = await match({ epipolar: { ...epipolar, kernel: 'tiled' } }),
        unbounded = await match({ epipolar: { ...epipolar, maxLinePx: 1e4 } }),
        wrong = await match({ epipolar: { ...epipolar, fundamental: columns } });
      const out = (r) => ({
        matches: [...r.matches],
        distances: [...r.distances],
        tier: r.report.tier,
      });
      const matches = {
        plain: out(plain),
        guided: out(guided),
        tiledKernel: out(tiledKernel),
        unbounded: out(unbounded),
        wrong: out(wrong),
      };
      await first.dispose();
      await second.dispose();
      await device.dispose();
      await kernels.dispose();
      gpu.destroy();
      return {
        random,
        sift,
        matches,
        a: [...a],
        b: [...b],
        adapterDescription,
        software,
        gpuErrors,
        liveBuffers: device.statistics.liveBuffers + kernels.statistics.liveBuffers,
      };
    },
    { fixtures, hardware: process.env.TYPECV_HARDWARE_GPU === '1', rows, columns },
    { timeoutMs: 300000, serve: { '/probe/guided-probe.js': probe } },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  assert.equal(result.liveBuffers, 0);
  // The tiled kernel holds every index, invalid flag and distance bit of the scalar kernel.
  const totals = { matched: 0, sole: 0, invalid: 0, degenerate: 0 };
  result.random.forEach(({ scalar, tiles }, i) => {
    const fixture = fixtures[i];
    assert.equal(scalar.tier, 'webgpu');
    assert.equal(tiles.tier, 'webgpu');
    assert.deepEqual(tiles.indices, scalar.indices, fixture.name);
    assert.deepEqual(tiles.distanceBits, scalar.distanceBits, fixture.name);
    const summary = summarize(fixture, scalar);
    totals.matched += summary.matched;
    totals.sole += summary.sole;
    totals.invalid += summary.invalid;
    // Query 0 sits at the epipole of the radial guides: no candidate, no invalid flag.
    if (fixture.name.includes('radial')) {
      assert.deepEqual(scalar.indices.slice(0, 3), [NONE, NONE, 0], fixture.name);
      totals.degenerate++;
    }
  });
  // The fixtures reach every rule they are meant to test.
  assert.ok(totals.matched > 0 && totals.sole > 0 && totals.invalid > 0, JSON.stringify(totals));
  const siftRules = { matched: 0, sole: 0 };
  for (const { name, scalar, tiles, queries, candidates, specialized } of result.sift) {
    assert.ok(queries > 40 && candidates > 40, `${name}: ${queries} x ${candidates}`);
    assert.deepEqual(tiles.indices, scalar.indices, name);
    assert.deepEqual(tiles.distanceBits, scalar.distanceBits, name);
    assert.equal(specialized.bitExact, true, name);
    assert.deepEqual(
      specialized.variants.map((v) => v.size),
      [64, 128, 256],
      name,
    );
    for (let i = 0; i < queries; i++) {
      if (scalar.indices[i * 3] !== NONE) siftRules.matched++;
      if (scalar.indices[i * 3 + 1] === candidates) siftRules.sole++;
    }
  }
  assert.ok(siftRules.matched > 0 && siftRules.sole > 0, JSON.stringify(siftRules));
  // The public matcher on WebGPU: the properties sift.test.mjs holds on the CPU tier.
  const { plain, guided, tiledKernel, unbounded, wrong } = result.matches,
    { a, b } = result;
  for (const r of [plain, guided, unbounded, wrong]) assert.equal(r.tier, 'webgpu');
  for (let i = 0; i < guided.matches.length; i += 2) {
    const p = guided.matches[i] * 2,
      q = guided.matches[i + 1] * 2;
    assert.ok(Math.abs(a[p + 1] - b[q + 1]) <= 2 + 1e-3, 'match off its epipolar line');
  }
  const onShift = (m) => {
    let n = 0;
    for (let i = 0; i < m.length; i += 2) {
      const p = m[i] * 2,
        q = m[i + 1] * 2;
      if (Math.hypot(b[q] - a[p] - 14, b[q + 1] - a[p + 1]) <= 2) n++;
    }
    return n;
  };
  assert.ok(onShift(guided.matches) >= onShift(plain.matches), 'guided lost correct matches');
  // `kernel: 'tiled'` is the opt-in: the same matches and distance bits as the scalar default.
  assert.deepEqual(tiledKernel.matches, guided.matches);
  assert.deepEqual(tiledKernel.distances, guided.distances);
  assert.deepEqual(unbounded.matches, plain.matches);
  assert.deepEqual(unbounded.distances, plain.distances);
  assert.ok(onShift(wrong.matches) < onShift(plain.matches) / 4, 'a wrong guide still matched');
  console.log(
    JSON.stringify({
      adapter: result.adapterDescription,
      software: result.software,
      fixtures: fixtures.length,
      ...totals,
      sift: siftRules,
      plain: plain.matches.length / 2,
      plainCorrect: onShift(plain.matches),
      guided: guided.matches.length / 2,
      guidedCorrect: onShift(guided.matches),
      wrong: wrong.matches.length / 2,
    }),
  );
});
