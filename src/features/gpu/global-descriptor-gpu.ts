/**
 * The kernel `globalAssign` on a device (`docs/multi-video.md`, 3.2). The host steps and the
 * CPU reference are in `global-descriptor.ts`; this file holds the part that needs a TypeShade
 * device, so that the reference loads in plain Node.
 */
import type { Resident } from 'typeshade/runtime';
import type { ShadeDevice } from '../../device/gpu/device.ts';
import type { GpuResidentFloatDescriptors } from './float-matching.ts';
import { globalAssign } from './global-descriptor.shade.ts';
import {
  GLOBAL_DESCRIPTOR_KIND,
  GLOBAL_DIMENSIONS,
  bagOfWordsDescriptor,
  checkShape,
  vladFromWords,
  wordHistogram,
} from './global-descriptor.ts';
import type { GlobalDescriptorOptions } from './global-descriptor.ts';

const LANES = 64;

export interface GlobalAssignOptions {
  /**
   * Descriptors per dispatch. The default sizes each chunk from the time budget of the device
   * (`ShadeDeviceOptions.timeBudgetMs`). A fixed size is for tests: the words do not depend on it.
   */
  readonly chunkDescriptors?: number;
}

/**
 * The word of each descriptor on `device`, with the kernel `globalAssign`. `descriptors` is a
 * resident array (the result of SIFT) or a host array of `count × 128` values; it must be finite.
 * `vocabulary` is a host array of `words × 128` values. The result is in descriptor order.
 *
 * The CPU tier gives the same indices as `assignWordsReference`. On WebGPU a floating-point
 * near-tie can flip a word, so a small share of indices can differ (section 7.4).
 */
export function assignWordsGpu(
  device: ShadeDevice,
  descriptors: GpuResidentFloatDescriptors | Float32Array,
  vocabulary: Float32Array,
  options: GlobalAssignOptions = {},
): Promise<Uint32Array> {
  const dimensions = GLOBAL_DIMENSIONS,
    host = descriptors instanceof Float32Array ? descriptors : undefined,
    resident = descriptors instanceof Float32Array ? undefined : descriptors;
  const count = host ? host.length / dimensions : resident!.count;
  if (resident && (resident.dimensions !== dimensions || resident.elements !== count * dimensions))
    throw new RangeError('The descriptors must have 128 dimensions');
  checkShape(host ?? new Float32Array(count * dimensions), vocabulary, dimensions);
  if (!Number.isSafeInteger(count) || count < 0) throw new RangeError('Invalid descriptor count');
  if (host)
    for (let i = 0; i < host.length; i++)
      if (!Number.isFinite(host[i])) throw new RangeError('Nonfinite descriptor');
  const words = vocabulary.length / dimensions,
    fixed = options.chunkDescriptors;
  if (fixed !== undefined && !(Number.isSafeInteger(fixed) && fixed >= 1))
    throw new RangeError('Invalid chunk size');
  if (
    Math.max(count * dimensions, vocabulary.length) * 4 > device.bindingBytes ||
    4 * (count * dimensions + vocabulary.length + count) > device.workingBytes
  )
    throw new RangeError('The word assignment exceeds the storage or working budget');
  if (count === 0) return Promise.resolve(new Uint32Array(0));
  return device.enqueue(async () => {
    const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
    let success = false;
    try {
      const source = host ? device.allocate(host.length, host) : resident!.descriptors;
      if (host) owned.push(source);
      const vocab = device.allocate(vocabulary.length, vocabulary),
        out = device.allocateU32(count);
      owned.push(vocab, out);
      const control = { words, dimensions };
      const chunks = function* () {
        const key = 'globalAssign';
        const run = (range: { first: number; end: number }, groups: number) =>
          globalAssign(
            {
              globalControl: control,
              globalRange: range,
              globalDescriptors: source,
              globalVocabulary: vocab,
              globalWords: out,
            },
            groups,
          );
        const warm = { id: key, run: () => run({ first: 0, end: 0 }, 0) };
        for (let first = 0; first < count;) {
          const remaining = count - first;
          let n =
            fixed ??
            Math.min(remaining, device.chunkSize(key, remaining, Math.min(remaining, LANES)));
          n = Math.min(Math.max(1, n), remaining);
          if (fixed === undefined && n < remaining)
            n = Math.min(remaining, Math.max(LANES, n - (n % LANES)));
          const range = { first, end: first + n };
          yield [
            'globalAssign',
            () => run(range, Math.ceil(n / LANES)),
            undefined,
            { key, units: n, warm },
          ] as const;
          first += n;
        }
      };
      await device.planSteps(chunks());
      const words32 = await device.read(out, count);
      success = true;
      return Uint32Array.from(words32.subarray(0, count));
    } finally {
      for (const buffer of owned) success ? device.recycle(buffer) : device.drop(buffer);
    }
  });
}

/** The descriptors on the host: the array itself, or a read-back of the resident array. */
async function hostDescriptors(
  device: ShadeDevice,
  descriptors: GpuResidentFloatDescriptors | Float32Array,
): Promise<Float32Array> {
  if (descriptors instanceof Float32Array) return descriptors;
  const n = descriptors.count * GLOBAL_DIMENSIONS;
  if (n === 0) return new Float32Array(0);
  const values = await device.enqueue(() => device.read(descriptors.descriptors, n));
  return Float32Array.from(values.subarray(0, n));
}

/**
 * The VLAD descriptor of one keyframe on `device`: `assignWordsGpu` for the words, then the
 * residual sums and the intra-normalization on the host (`vladFromWords`). A resident input is
 * read back once for the residuals. The vocabulary is `rootsift-32-v1`.
 */
export async function vladDescriptorGpu(
  device: ShadeDevice,
  descriptors: GpuResidentFloatDescriptors | Float32Array,
  vocabulary: Float32Array,
  options: GlobalAssignOptions = {},
): Promise<Float32Array> {
  const assigned = await assignWordsGpu(device, descriptors, vocabulary, options);
  return vladFromWords(await hostDescriptors(device, descriptors), vocabulary, assigned);
}

/**
 * The bag-of-words descriptor of one keyframe on `device`: `assignWordsGpu`, then the host steps.
 * The vocabulary is `rootsift-512-v1`, and `idf` gives the weights of the run.
 */
export async function bagOfWordsDescriptorGpu(
  device: ShadeDevice,
  descriptors: GpuResidentFloatDescriptors | Float32Array,
  vocabulary: Float32Array,
  idf?: ArrayLike<number>,
  options: GlobalAssignOptions = {},
): Promise<Float32Array> {
  const assigned = await assignWordsGpu(device, descriptors, vocabulary, options);
  return bagOfWordsDescriptor(wordHistogram(assigned, vocabulary.length / GLOBAL_DIMENSIONS), idf);
}

/**
 * The global descriptor of one keyframe on `device`. The default kind is VLAD (`vocabulary` is
 * `rootsift-32-v1`). With `kind: 'bag-of-words'` it is the bag of words (`rootsift-512-v1`),
 * and `idf` gives the weights. `idf` with VLAD throws.
 */
export function globalDescriptorGpu(
  device: ShadeDevice,
  descriptors: GpuResidentFloatDescriptors | Float32Array,
  vocabulary: Float32Array,
  options: GlobalDescriptorOptions & GlobalAssignOptions = {},
): Promise<Float32Array> {
  const kind = options.kind ?? GLOBAL_DESCRIPTOR_KIND;
  if (kind === 'bag-of-words')
    return bagOfWordsDescriptorGpu(device, descriptors, vocabulary, options.idf, options);
  if (options.idf) return Promise.reject(new RangeError('VLAD takes no idf'));
  return vladDescriptorGpu(device, descriptors, vocabulary, options);
}
