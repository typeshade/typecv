import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { descriptorPair, hammingOracle } from './hamming-oracle.mjs';

// Verifies the tiled Hamming kernel on WebGPU (SwiftShader in CI): it equals the untiled kernel
// on the CPU tier and the JavaScript oracle, and it runs past the former pair cap of 4,000,000.
test('the tiled Hamming matcher equals the untiled one and the oracle', async () => {
  const cases = [
    [1, 1, 0.8, 80],
    [63, 65, 0.8, 256],
    [64, 64, 0.8, 256],
    [130, 200, 0.9, 256],
    [300, 129, 0.8, 100],
    [1000, 777, 0.85, 256],
    [4500, 4500, 0.85, 256],
  ].map(([nf, ns, ratio, maxDistance]) => ({
    nf,
    ns,
    ratio,
    maxDistance,
    ...descriptorPair(nf, ns, 11 + nf),
  }));
  const { result, errors } = await inBrowser(
    async ({ cases }) => {
      const { ShadeDevice, matchFeaturesGpu } = await import('/dist/gpu.js');
      const adapter = await navigator.gpu.requestAdapter(),
        gpu = await adapter.requestDevice(),
        device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0 }),
        cpu = await ShadeDevice.open({ backend: 'cpu' }),
        revive = (set) => ({
          descriptors: Uint32Array.from(set.descriptors),
          scores: Float32Array.from(set.scores),
          keypoints: Float32Array.from(set.keypoints),
        });
      try {
        const out = [];
        for (const c of cases) {
          const first = revive(c.first),
            second = revive(c.second),
            options = { ratio: c.ratio, maxDistance: c.maxDistance },
            started = performance.now(),
            tiled = await matchFeaturesGpu(device, first, second, options),
            tiledMs = performance.now() - started,
            // The untiled kernel on the CPU tier, for the smaller cases (the large one is too slow).
            untiled =
              c.nf * c.ns <= 1_000_000
                ? await matchFeaturesGpu(cpu, first, second, options)
                : undefined;
          out.push({
            tier: tiled.report.tier,
            tiledMs,
            matches: Array.from(tiled.matches),
            distances: Array.from(tiled.distances),
            untiled: untiled && {
              matches: Array.from(untiled.matches),
              distances: Array.from(untiled.distances),
            },
          });
        }
        return out;
      } finally {
        await device.dispose();
        await cpu.dispose();
      }
    },
    {
      cases: cases.map((c) => ({
        ...c,
        first: {
          ...c.first,
          descriptors: Array.from(c.first.descriptors),
          scores: Array.from(c.first.scores),
          keypoints: Array.from(c.first.keypoints),
        },
        second: {
          ...c.second,
          descriptors: Array.from(c.second.descriptors),
          scores: Array.from(c.second.scores),
          keypoints: Array.from(c.second.keypoints),
        },
      })),
    },
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.ok(
    result.slice(3).every((r) => r.matches.length >= 20),
    'the larger cases have matches',
  );
  cases.forEach((c, k) => {
    const got = result[k],
      want = hammingOracle(c.first, c.second, c.ratio, c.maxDistance);
    assert.equal(got.tier, 'webgpu');
    assert.deepEqual(got.matches, Array.from(want.matches), `${c.nf} x ${c.ns} matches`);
    assert.deepEqual(got.distances, Array.from(want.distances), `${c.nf} x ${c.ns} distances`);
    if (got.untiled) {
      assert.deepEqual(got.untiled.matches, got.matches, `${c.nf} x ${c.ns} tiled equals untiled`);
      assert.deepEqual(got.untiled.distances, got.distances);
    }
    process.stderr.write(
      `# Hamming ${c.nf} x ${c.ns}: ${got.matches.length / 2} matches, ${got.tiledMs.toFixed(0)} ms (SwiftShader)\n`,
    );
  });
});
