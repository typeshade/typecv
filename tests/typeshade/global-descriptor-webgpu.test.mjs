import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { inBrowser } from './browser.mjs';

// Verifies: docs/multi-video.md 7.4 (kernel against reference, WebGPU).

const vocabularyFile = fileURLToPath(
  new URL('../../src/gpu/vocabulary/rootsift-512-v1.bin', import.meta.url),
);
const vladFile = fileURLToPath(
  new URL('../../src/gpu/vocabulary/rootsift-32-v1.bin', import.meta.url),
);

test('WebGPU globalAssign agrees with the reference: words and descriptor cosine', async () => {
  const { result, errors } = await inBrowser(
    async ({ hardware }) => {
      const api = await import('/dist/gpu.js');
      const cases = await import('/tests/typeshade/global-descriptor-cases.mjs');
      const sift = await import('/tests/typeshade/sift-cases.mjs');
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('WebGPU adapter unavailable');
      const info = adapter.info;
      const description = [info?.vendor, info?.architecture, info?.device, info?.description]
        .filter(Boolean)
        .join(' ');
      const software =
        /swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter === true;
      if (hardware && software) throw new Error('Hardware requested but software adapter selected');
      const gpu = await adapter.requestDevice(),
        gpuErrors = [];
      gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      const device = await api.ShadeDevice.open({ gpuDevice: gpu });
      const vocabulary = api.parseVocabulary(
        await (await fetch('/vocabulary.bin')).arrayBuffer(),
        512,
      );
      const vlad = api.parseVocabulary(await (await fetch('/vlad.bin')).arrayBuffer());
      const out = { gpuErrors, software, description, groups: [] };
      try {
        // Host descriptors: random, near a word, and exactly on a word.
        const host = new Float32Array([
          ...cases.rootSiftFixture(400, 31),
          ...cases.nearWords(vocabulary, 400, 32),
          ...vocabulary.subarray(0, 50 * 128),
        ]);
        // Resident descriptors of a real SIFT run on a synthetic texture.
        const width = 256,
          height = 192,
          rgba = sift.warped(sift.texture(width, height), width, height, width, height, 1, 0, 0, 0);
        const features = await api.extractSiftGpuResident(device, rgba, width, height, {
          maxFeatures: 1024,
        });
        const siftHost = await device.enqueue(() =>
          device.read(features.descriptors.descriptors, features.count * 128),
        );
        const inputs = [
          ['host', host, host],
          [
            'sift',
            features.descriptors,
            Float32Array.from(siftHost.subarray(0, features.count * 128)),
          ],
        ];
        for (const [name, input, values] of inputs) {
          const expected = api.assignWordsReference(values, vocabulary);
          const words = await api.assignWordsGpu(device, input, vocabulary);
          const chunked = await api.assignWordsGpu(device, input, vocabulary, {
            chunkDescriptors: 97,
          });
          const idf = api.inverseDocumentFrequency([
            api.wordHistogram(expected),
            api.wordHistogram(api.assignWordsReference(cases.rootSiftFixture(60, 4), vocabulary)),
          ]);
          const bag = { kind: 'bag-of-words', idf };
          const reference = api.globalDescriptorReference(values, vocabulary, bag);
          const device_ = await api.globalDescriptorGpu(device, input, vocabulary, bag);
          // VLAD, the default, with the 32-word vocabulary.
          const vladExpected = api.assignWordsReference(values, vlad);
          const vladWords = await api.assignWordsGpu(device, input, vlad);
          const vladReference = api.globalDescriptorReference(values, vlad);
          const vladDevice = await api.globalDescriptorGpu(device, input, vlad);
          const vladChunked = await api.globalDescriptorGpu(device, input, vlad, {
            chunkDescriptors: 97,
          });
          out.groups.push({
            name,
            count: expected.length,
            equal: expected.reduce((n, w, i) => n + (w === words[i] ? 1 : 0), 0),
            chunkedEqual: words.every((w, i) => w === chunked[i]),
            cosine: api.cosineSimilarity(reference, device_),
            vladDimension: vladDevice.length,
            vladEqual: vladExpected.reduce((n, w, i) => n + (w === vladWords[i] ? 1 : 0), 0),
            vladCosine: api.cosineSimilarity(vladReference, vladDevice),
            vladChunkedCosine: api.cosineSimilarity(vladReference, vladChunked),
          });
        }
        await features.dispose();
      } finally {
        await device.dispose();
        gpu.destroy();
      }
      out.liveBuffers = device.statistics.liveBuffers;
      return out;
    },
    { hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
    { timeoutMs: 900000, serve: { '/vocabulary.bin': vocabularyFile, '/vlad.bin': vladFile } },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  assert.equal(result.liveBuffers, 0);
  console.log(JSON.stringify(result.groups));
  assert.equal(result.groups.length, 2);
  for (const g of result.groups) {
    assert.ok(g.count >= 40, `${g.name}: only ${g.count} descriptors`);
    assert.ok(g.equal / g.count >= 0.999, `${g.name}: ${g.equal} of ${g.count} words equal`);
    assert.ok(g.cosine >= 0.9999, `${g.name}: cosine ${g.cosine}`);
    assert.equal(g.chunkedEqual, true, `${g.name}: chunk size changed the words`);
    assert.equal(g.vladDimension, 4096);
    assert.ok(g.vladEqual / g.count >= 0.999, `${g.name}: ${g.vladEqual} of ${g.count} VLAD words`);
    assert.ok(g.vladCosine >= 0.9999, `${g.name}: VLAD cosine ${g.vladCosine}`);
    assert.ok(
      g.vladChunkedCosine >= 0.9999,
      `${g.name}: chunked VLAD cosine ${g.vladChunkedCosine}`,
    );
  }
});
