import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  ShadeDevice,
  assignWordsGpu,
  assignWordsReference,
  bagOfWordsDescriptor,
  bagOfWordsDescriptorGpu,
  globalDescriptorGpu,
  globalDescriptorReference,
  inverseDocumentFrequency,
  parseVocabulary,
  vladDescriptorGpu,
  vladDescriptorReference,
  wordHistogram,
} from '../../dist/gpu.js';
import { nearWords, rootSiftFixture } from './global-descriptor-cases.mjs';

// Verifies: docs/multi-video.md 7.4 (kernel against reference, CPU tier; chunk invariance).

// The bag of words uses the 512-word vocabulary. VLAD (the default) uses the 32-word one.
const vocabulary = parseVocabulary(
  readFileSync(new URL('../../src/gpu/vocabulary/rootsift-512-v1.bin', import.meta.url)),
  512,
);
const vladVocabulary = parseVocabulary(
  readFileSync(new URL('../../src/gpu/vocabulary/rootsift-32-v1.bin', import.meta.url)),
);

async function withDevice(body, options = {}) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, ...options });
  try {
    return await body(device);
  } finally {
    await device.dispose();
  }
}

test('globalAssign gives the word indices of the reference, and the descriptors equal to 1e-6', async () => {
  // Random descriptors, descriptors near a word, and descriptors exactly on a word.
  const descriptors = new Float32Array([
    ...rootSiftFixture(120, 5),
    ...nearWords(vocabulary, 120, 6),
    ...vocabulary.subarray(0, 40 * 128),
  ]);
  const expected = assignWordsReference(descriptors, vocabulary);
  await withDevice(async (device) => {
    const words = await assignWordsGpu(device, descriptors, vocabulary);
    assert.deepEqual([...words], [...expected]);
    // A descriptor that is a word is assigned to a word at distance 0: itself, or a lower twin.
    for (let w = 0; w < 40; w++) assert.equal(words[240 + w], w);
    const hist = wordHistogram(expected);
    const idf = inverseDocumentFrequency([
      hist,
      wordHistogram(assignWordsReference(rootSiftFixture(50, 9), vocabulary)),
    ]);
    const reference = globalDescriptorReference(descriptors, vocabulary, {
      kind: 'bag-of-words',
      idf,
    });
    const device_ = await globalDescriptorGpu(device, descriptors, vocabulary, {
      kind: 'bag-of-words',
      idf,
    });
    assert.deepEqual(
      [...(await bagOfWordsDescriptorGpu(device, descriptors, vocabulary, idf))],
      [...device_],
    );
    assert.equal(device_.length, 512);
    for (let w = 0; w < 512; w++)
      assert.ok(Math.abs(device_[w] - reference[w]) <= 1e-6, `bin ${w}`);
    assert.deepEqual([...reference], [...bagOfWordsDescriptor(hist, idf)]);
  });
});

test('the words do not depend on the chunk size or on the time budget', async () => {
  const descriptors = new Float32Array([
    ...rootSiftFixture(150, 11),
    ...nearWords(vocabulary, 150, 12),
  ]);
  const expected = [...assignWordsReference(descriptors, vocabulary)];
  await withDevice(async (device) => {
    // 1 and 63 and 64 and 65 cut the 300 descriptors at and around the 64 lanes of a group.
    for (const chunkDescriptors of [1, 63, 64, 65, 127, 300, 1000])
      assert.deepEqual(
        [...(await assignWordsGpu(device, descriptors, vocabulary, { chunkDescriptors }))],
        expected,
        `chunk of ${chunkDescriptors}`,
      );
  });
  for (const timeBudgetMs of [1, 250])
    await withDevice(
      async (device) =>
        assert.deepEqual([...(await assignWordsGpu(device, descriptors, vocabulary))], expected),
      { timeBudgetMs, chunkTimingSamples: 1 },
    );
});

test('a tie between two words goes to the lower word index, and a resident input works', async () => {
  const words = 6,
    twins = new Float32Array(words * 128);
  const base = rootSiftFixture(3, 21);
  // Words 1 and 4 are the same vector, as are 2 and 5: the lower index must win.
  twins.set(base.subarray(0, 128), 0);
  twins.set(base.subarray(128, 256), 128);
  twins.set(base.subarray(256, 384), 256);
  twins.set(base.subarray(128, 256), 3 * 128);
  twins.set(base.subarray(128, 256), 4 * 128);
  twins.set(base.subarray(256, 384), 5 * 128);
  const queries = new Float32Array([...base.subarray(128, 384), ...base.subarray(0, 128)]);
  assert.deepEqual([...assignWordsReference(queries, twins)], [1, 2, 0]);
  await withDevice(async (device) => {
    assert.deepEqual([...(await assignWordsGpu(device, queries, twins))], [1, 2, 0]);
    const resident = device.allocate(queries.length, queries);
    try {
      const words_ = await assignWordsGpu(
        device,
        { descriptors: resident, count: 3, dimensions: 128, elements: queries.length },
        twins,
      );
      assert.deepEqual([...words_], [1, 2, 0]);
    } finally {
      device.drop(resident);
    }
  });
});

test('assignWordsGpu rejects bad shapes and nonfinite descriptors and accepts none', async () => {
  await withDevice(async (device) => {
    assert.deepEqual([...(await assignWordsGpu(device, new Float32Array(0), vocabulary))], []);
    await assert.rejects(
      async () => assignWordsGpu(device, new Float32Array(100), vocabulary),
      RangeError,
    );
    await assert.rejects(
      async () => assignWordsGpu(device, new Float32Array(128), new Float32Array(100)),
      RangeError,
    );
    const bad = rootSiftFixture(2, 3);
    bad[130] = NaN;
    await assert.rejects(async () => assignWordsGpu(device, bad, vocabulary), RangeError);
    await assert.rejects(
      async () =>
        assignWordsGpu(device, rootSiftFixture(2, 3), vocabulary, { chunkDescriptors: 0 }),
      RangeError,
    );
    // A failed job leaves the device usable.
    const ok = rootSiftFixture(4, 8);
    assert.deepEqual(
      [...(await assignWordsGpu(device, ok, vocabulary))],
      [...assignWordsReference(ok, vocabulary)],
    );
  });
});

test('VLAD on the CPU tier equals the reference to 1e-6, with the 32-word vocabulary', async () => {
  const descriptors = new Float32Array([
    ...rootSiftFixture(150, 41),
    ...nearWords(vladVocabulary, 150, 42),
    ...vladVocabulary.subarray(0, 32 * 128),
  ]);
  const reference = vladDescriptorReference(descriptors, vladVocabulary);
  assert.equal(reference.length, 4096);
  await withDevice(async (device) => {
    const direct = await vladDescriptorGpu(device, descriptors, vladVocabulary);
    const byDefault = await globalDescriptorGpu(device, descriptors, vladVocabulary);
    assert.equal(direct.length, 4096);
    for (let i = 0; i < 4096; i++)
      assert.ok(Math.abs(direct[i] - reference[i]) <= 1e-6, `value ${i}`);
    assert.deepEqual([...byDefault], [...direct]);
    // The words are the same on the CPU tier, so the descriptor is equal bit for bit.
    assert.deepEqual([...direct], [...reference]);
    await assert.rejects(
      globalDescriptorGpu(device, descriptors, vladVocabulary, { idf: [1] }),
      RangeError,
    );
  });
});

test('VLAD does not depend on the chunk size, and a resident input is read back', async () => {
  const descriptors = new Float32Array([
    ...rootSiftFixture(130, 51),
    ...nearWords(vladVocabulary, 130, 52),
  ]);
  const reference = [...vladDescriptorReference(descriptors, vladVocabulary)];
  await withDevice(async (device) => {
    for (const chunkDescriptors of [1, 63, 64, 65, 260, 1000])
      assert.deepEqual(
        [...(await vladDescriptorGpu(device, descriptors, vladVocabulary, { chunkDescriptors }))],
        reference,
        `chunk of ${chunkDescriptors}`,
      );
    const resident = device.allocate(descriptors.length, descriptors);
    try {
      const got = await vladDescriptorGpu(
        device,
        { descriptors: resident, count: 260, dimensions: 128, elements: descriptors.length },
        vladVocabulary,
      );
      assert.deepEqual([...got], reference);
    } finally {
      device.drop(resident);
    }
    assert.equal(
      (await vladDescriptorGpu(device, new Float32Array(0), vladVocabulary)).length,
      4096,
    );
  });
  for (const timeBudgetMs of [1, 250])
    await withDevice(
      async (device) =>
        assert.deepEqual(
          [...(await vladDescriptorGpu(device, descriptors, vladVocabulary))],
          reference,
        ),
      { timeBudgetMs, chunkTimingSamples: 1 },
    );
});
