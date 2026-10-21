import test from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { inBrowser } from './browser.mjs';

// The lab's COLMAP-SIFT run lost the device (DXGI_ERROR_DEVICE_HUNG, NVIDIA Blackwell,
// 2026-10-19) in its first triangulation: 1672 tracks of two observations each. The fixture
// is that job. Each variant runs in its own browser and device, from the simplest to the
// full kernel, so that a hang names the loop it is in. On the CPU tier and on SwiftShader
// every variant completes; `TYPECV_HARDWARE_GPU=1` runs them on the physical adapter.
const FIXTURE = resolve(import.meta.dirname, '../fixtures/south-building-pair-triangulation.json');
const VARIANTS = [
  ['no refinement, one pair hypothesis', { refinementIterations: 0, maxPairHypotheses: 1 }],
  ['no refinement', { refinementIterations: 0 }],
  ['the lab job', {}],
];

for (const [name, override] of VARIANTS)
  test(`south-building pair triangulation on WebGPU: ${name}`, async () => {
    const { result, errors } = await inBrowser(
      async ({ override }) => {
        const { ShadeDevice, triangulateTracksGpu } = await import('/dist/gpu.js');
        const fixture = await (await fetch('/fixture.json')).json();
        const tracks = fixture.pixels.map((pixels) =>
          fixture.views.map((view, k) => ({
            camera: view.camera,
            pose: { R: Float64Array.from(view.pose.R), t: Float64Array.from(view.pose.t) },
            pixel: [pixels[2 * k], pixels[2 * k + 1]],
          })),
        );
        const d = await ShadeDevice.open();
        const adapter = d.adapterInfo ?? null;
        try {
          const started = performance.now();
          const r = await triangulateTracksGpu(d, tracks, { ...fixture.options, ...override });
          const status = [...r.status],
            counts = {};
          for (const s of status) counts[s] = (counts[s] ?? 0) + 1;
          return {
            adapter,
            elapsedMs: performance.now() - started,
            counts,
            accepted: status.filter((s) => s === 0).length,
            expectedAccepted: fixture.expectedStatus.filter((s) => s === 0).length,
            tracks: tracks.length,
            dispatches: r.execution.report.dispatches,
            calls: r.execution.report.groups.flatMap((group) => group.calls),
            chunks: r.execution.report.chunks?.triangulateTracks ?? null,
            tier: r.execution.report.tier,
          };
        } finally {
          await d.dispose();
        }
      },
      { override },
      { timeoutMs: 240000, serve: { '/fixture.json': FIXTURE } },
    );
    console.log(name, JSON.stringify(result));
    assert.deepEqual(errors, []);
    assert.equal(result.tier, 'webgpu');
    assert.equal(result.tracks, 1672);
    assert.ok(result.accepted >= 1600, `${result.accepted} accepted`);
    // `triangulateTracks` runs as chunks of tracks under the time budget: no single dispatch
    // covers the 1672 tracks. The first chunk is one workgroup of 16 tracks, and each chunk
    // grows at most fourfold toward the occupancy floor: 16 + 64 + 256 + 1024 = 1360 tracks, so
    // the job takes at least five chunks. Before them run the prepare dispatch and, on the fresh
    // device, the warm-up: the kernel over zero workgroups, so that the first timed chunk
    // excludes the creation of its pipeline. The rate counts the timed chunks, at most eight
    // (`chunkTimingSamples`).
    const [prepare, warm, ...chunks] = result.calls;
    assert.equal(prepare, 'prepareTriangulation');
    assert.equal(warm, 'triangulateTracks');
    assert.ok(chunks.length >= 5 && chunks.every((call) => call === 'triangulateTracks'));
    assert.equal(result.chunks?.chunks, Math.min(8, chunks.length), JSON.stringify(result.chunks));
    assert.equal(result.dispatches, 2 + chunks.length);
    if (name === 'the lab job')
      assert.ok(
        Math.abs(result.accepted - result.expectedAccepted) <= 8,
        `${result.accepted} accepted on the device, ${result.expectedAccepted} on the CPU tier`,
      );
  });

// docs/gpu-throughput.md, item 2: a key that has settled plans a call of N tracks as
// `ceil(N / floor)` dispatches, the floor being 1024 workgroups of 16 tracks.
test('a settled triangulation key plans ceil(N / floor) chunks', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, triangulateTracksGpu } = await import('/dist/gpu.js');
      const fixture = await (await fetch('/fixture.json')).json();
      const one = fixture.pixels.map((pixels) =>
        fixture.views.map((view, k) => ({
          camera: view.camera,
          pose: { R: Float64Array.from(view.pose.R), t: Float64Array.from(view.pose.t) },
          pixel: [pixels[2 * k], pixels[2 * k + 1]],
        })),
      );
      const options = { ...fixture.options, refinementIterations: 0, maxPairHypotheses: 1 };
      // One timed chunk settles the key: the later chunks are sized by the floor.
      const d = await ShadeDevice.open({ chunkTimingSamples: 1 });
      try {
        const warmup = await triangulateTracksGpu(d, one, options);
        const small = await triangulateTracksGpu(d, one, options);
        const large = await triangulateTracksGpu(
          d,
          Array.from({ length: 10 }, () => one).flat(),
          options,
        );
        return {
          warmup: warmup.execution.report.dispatches,
          small: small.execution.report.dispatches,
          large: large.execution.report.dispatches,
          tracks: one.length,
          largeTracks: large.status.length,
        };
      } finally {
        await d.dispose();
      }
    },
    {},
    { timeoutMs: 300000, serve: { '/fixture.json': FIXTURE } },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.tracks, 1672);
  assert.ok(result.warmup >= 4, `${result.warmup} dispatches on the fresh key`);
  // One prepare dispatch and one chunk.
  assert.equal(result.small, 1 + Math.ceil(result.tracks / (16 * 1024)));
  assert.equal(result.largeTracks, 16720);
  assert.equal(result.large, 1 + Math.ceil(result.largeTracks / (16 * 1024)));
});
