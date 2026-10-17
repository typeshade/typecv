import { createRuntime, configure, resident } from 'typeshade/runtime';
import { fuse } from 'typeshade/fuse';
import {
  l2RowNorms,
  normalizeL2,
  normalizeDescriptors,
  gatherNormalizedPatch3x3,
  gridSampleNormalizedNHWC,
} from '../../src/gpu/normalized-sampling.shade.ts';
import {
  legacyNormalizeL2,
  legacyNormalizeDescriptors,
  unpad,
  gatherPatch3x3,
  gridSampleNHWC,
} from '../../experimental/typeshade/normalized-sampling-reference.shade.ts';

const assert = (value, message) => {
  if (!value) throw new Error(message);
};
const words = (value) => new Uint32Array(value.buffer, value.byteOffset, value.length);

export async function checkNormalizedSampling({ hardware = false } = {}) {
  const adapter = await navigator.gpu?.requestAdapter();
  assert(adapter, 'WebGPU adapter required');
  const info = adapter.info;
  assert(
    !hardware ||
      (!adapter.isFallbackAdapter &&
        !/swiftshader|software|llvmpipe/i.test(
          [info.vendor, info.architecture, info.description].join(' '),
        )),
    'Physical WebGPU requested but software adapter selected',
  );
  const device = await adapter.requestDevice();
  const validationErrors = [];
  device.addEventListener('uncapturederror', (event) => validationErrors.push(event.error.message));
  const cases = [];
  let runtime;
  try {
    runtime = await createRuntime({ device });
    configure({ runtime, prefer: ['webgpu'] });
    // No timing claim: intentionally includes small, uneven and signed-zero cases.
    for (const channels of [1, 2, 3, 7, 31, 127, 128, 129, 255]) {
      for (const rows of [1, 3, 65]) {
        const values = new Float32Array(rows * channels);
        for (let row = 0; row < rows; row++) {
          for (let channel = 0; channel < channels; channel++) {
            const i = row * channels + channel;
            const pattern = row % 7;
            values[i] =
              pattern === 0
                ? channel % 2
                  ? -0
                  : 0
                : pattern === 1
                  ? (channel % 2 ? -1 : 1) * 1e-14
                  : pattern === 2
                    ? (channel % 2 ? -1 : 1) * 1e-20
                    : pattern === 3
                      ? Math.sin(i * 0.73) * 3.7
                      : pattern === 4
                        ? (channel % 2 ? -1 : 1) * 1e12
                        : pattern === 5
                          ? channel === 0
                            ? 1
                            : 1e-5
                          : (((i * 37) % 257) - 128) / 31;
          }
        }
        const source = resident(values);
        const expected = resident(new Float32Array(values.length));
        const actual = resident(new Float32Array(values.length));
        const norms = resident(new Float32Array(rows));
        try {
          for (const descriptor of channels === 128 ? [false, true] : [false]) {
            const plan = await fuse(() => {
              if (descriptor) legacyNormalizeDescriptors(expected, source, rows);
              else legacyNormalizeL2(expected, source, channels);
              l2RowNorms(norms, source, channels);
              if (descriptor) normalizeDescriptors(actual, source, norms, rows);
              else normalizeL2(actual, source, norms, channels);
            });
            const [left, right] = await Promise.all([expected.read(), actual.read()]);
            const a = words(left),
              b = words(right);
            assert(
              plan.groups.length > 0 &&
                plan.groups.every((group) => group.workgroupShapes?.length > 0),
              'Every kernel must execute on WebGPU',
            );
            for (let i = 0; i < a.length; i++) {
              assert(
                Number.isFinite(right[i]),
                `Non-finite output: channels=${channels}, rows=${rows}, i=${i}`,
              );
              assert(
                a[i] === b[i],
                `Bit mismatch: channels=${channels}, rows=${rows}, descriptors=${descriptor}, i=${i}, old=${left[i]} [${a[i]}], new=${right[i]} [${b[i]}]`,
              );
            }
            cases.push({
              channels,
              rows,
              descriptor,
              elements: a.length,
              bitwiseEqual: true,
              scratchBytes: rows * 4,
              dispatches: plan.dispatches,
              submissions: plan.submissions,
              allGroupsWebgpu: true,
            });
          }
        } finally {
          source.destroy();
          expected.destroy();
          actual.destroy();
          norms.destroy();
        }
      }
    }
    // Compare materialize-then-sample with direct normalized sampling, including
    // asymmetric padding, image edges, both alignment policies and nonfinite grids.
    const sampleCases = [];
    const shapes = [
      { height: 4, width: 4, paddedHeight: 4, paddedWidth: 4, padTop: 0, padLeft: 0 },
      { height: 5, width: 7, paddedHeight: 8, paddedWidth: 12, padTop: 1, padLeft: 2 },
      { height: 31, width: 35, paddedHeight: 32, paddedWidth: 64, padTop: 0, padLeft: 14 },
    ];
    for (const shape of shapes) {
      for (const channels of [1, 3, 128]) {
        for (const alignCorners of [0, 1]) {
          const { height, width, paddedHeight, paddedWidth, padTop, padLeft } = shape;
          const input = new Float32Array(paddedHeight * paddedWidth * channels);
          for (let i = 0; i < input.length; i++)
            input[i] =
              Math.floor(i / channels) % 11 === 0 ? 0 : Math.sin(i * 0.131) * ((i % 5) + 1);
          const positions = new Float32Array([-1, -1, 1, 1, 0, 0, -0.99, 0.98, -0.13, 0.29]);
          const gridValues = new Float32Array([
            -1,
            -1,
            1,
            1,
            0,
            0,
            -1.1,
            -0.98,
            0.9999,
            1.05,
            -1.5,
            2,
            1e30,
            -1e30,
            0.3,
            -0.2,
            NaN,
            0,
            0,
            Infinity,
            -Infinity,
            0,
          ]);
          const buffers = [];
          const keep = (value) => {
            const r = resident(value);
            buffers.push(r);
            return r;
          };
          try {
            const source = keep(input),
              norms = keep(new Float32Array(paddedHeight * paddedWidth));
            const materialized = keep(new Float32Array(input.length)),
              cropped = keep(new Float32Array(height * width * channels));
            const keypoints = keep(positions),
              grid = keep(gridValues);
            const expectedPatch = keep(new Float32Array((positions.length / 2) * 9 * channels)),
              actualPatch = keep(new Float32Array((positions.length / 2) * 9 * channels));
            const expectedGrid = keep(new Float32Array((gridValues.length / 2) * channels)),
              actualGrid = keep(new Float32Array((gridValues.length / 2) * channels));
            const plan = await fuse(() => {
              legacyNormalizeL2(materialized, source, channels);
              unpad(cropped, materialized, paddedWidth, width, channels, padTop, padLeft);
              gatherPatch3x3(expectedPatch, cropped, keypoints, height, width, channels);
              gridSampleNHWC(
                expectedGrid,
                cropped,
                grid,
                height,
                width,
                channels,
                gridValues.length / 2,
                1,
                alignCorners,
              );
              l2RowNorms(norms, source, channels);
              gatherNormalizedPatch3x3(
                actualPatch,
                source,
                norms,
                keypoints,
                height,
                width,
                channels,
                paddedWidth,
                padTop,
                padLeft,
              );
              gridSampleNormalizedNHWC(
                actualGrid,
                source,
                norms,
                grid,
                height,
                width,
                channels,
                gridValues.length / 2,
                1,
                alignCorners,
                paddedHeight,
                paddedWidth,
                padTop,
                padLeft,
              );
            });
            assert(
              plan.groups.length > 0 &&
                plan.groups.every((group) => group.workgroupShapes?.length > 0),
              'Every sampler must execute on WebGPU',
            );
            let comparedElements = 0;
            for (const [name, expected, actual] of [
              ['patch', expectedPatch, actualPatch],
              ['grid', expectedGrid, actualGrid],
            ]) {
              const [left, right] = await Promise.all([expected.read(), actual.read()]);
              const a = words(left),
                b = words(right);
              for (let i = 0; i < a.length; i++)
                assert(
                  a[i] === b[i],
                  `${name} bit mismatch shape=${JSON.stringify(shape)} channels=${channels} align=${alignCorners} at ${i}: old=${left[i]} [${a[i]}], new=${right[i]} [${b[i]}]`,
                );
              comparedElements += a.length;
            }
            sampleCases.push({
              ...shape,
              channels,
              alignCorners,
              comparedElements,
              bitwiseEqual: true,
              allGroupsWebgpu: true,
            });
          } finally {
            for (const buffer of buffers) buffer.destroy();
          }
        }
      }
    }
    // Batched grid sampling must use each batch's own padded storage and norms.
    // Build the materialized reference crop with byte-preserving CPU copies;
    // the original single-image unpad kernel is deliberately not reused here.
    const batchCases = [];
    for (const padded of [false, true]) {
      for (const channels of [3, 128]) {
        for (const alignCorners of [0, 1]) {
          const batches = 2,
            height = 5,
            width = 7,
            paddedHeight = padded ? 8 : height,
            paddedWidth = padded ? 12 : width,
            padTop = padded ? 1 : 0,
            padLeft = padded ? 2 : 0,
            outHeight = 2,
            outWidth = 3;
          const input = new Float32Array(batches * paddedHeight * paddedWidth * channels);
          for (let b = 0; b < batches; b++) {
            for (let i = 0; i < paddedHeight * paddedWidth * channels; i++) {
              input[b * paddedHeight * paddedWidth * channels + i] =
                b === 0 ? Math.sin(i * 0.191) * ((i % 7) + 1) : Math.cos(i * 0.373) - 0.7;
            }
          }
          const gridValues = new Float32Array([
            -1, -1, 1, 1, 0, 0, -0.7, 0.4, 1.1, 0.9, -0.2, -0.3, 1, -1, -1, 1, 0.25, -0.6, -0.8,
            0.8, -1.1, -0.5, 0.2, 0.3,
          ]);
          const buffers = [];
          const keep = (value) => {
            const buffer = resident(value);
            buffers.push(buffer);
            return buffer;
          };
          try {
            const source = keep(input),
              materialized = keep(new Float32Array(input.length)),
              norms = keep(new Float32Array(batches * paddedHeight * paddedWidth));
            const normalization = await fuse(() => {
              legacyNormalizeL2(materialized, source, channels);
              l2RowNorms(norms, source, channels);
            });
            const normalized = await materialized.read();
            const croppedValues = new Float32Array(batches * height * width * channels);
            for (let b = 0; b < batches; b++) {
              for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                  const from =
                    (b * paddedHeight * paddedWidth + (y + padTop) * paddedWidth + x + padLeft) *
                    channels;
                  const to = (b * height * width + y * width + x) * channels;
                  croppedValues.set(normalized.subarray(from, from + channels), to);
                }
              }
            }
            const cropped = keep(croppedValues),
              grid = keep(gridValues),
              expected = keep(new Float32Array(batches * outHeight * outWidth * channels)),
              actual = keep(new Float32Array(batches * outHeight * outWidth * channels));
            const sampling = await fuse(() => {
              gridSampleNHWC(
                expected,
                cropped,
                grid,
                height,
                width,
                channels,
                outHeight,
                outWidth,
                alignCorners,
              );
              gridSampleNormalizedNHWC(
                actual,
                source,
                norms,
                grid,
                height,
                width,
                channels,
                outHeight,
                outWidth,
                alignCorners,
                paddedHeight,
                paddedWidth,
                padTop,
                padLeft,
              );
            });
            assert(
              [normalization, sampling].every(
                (plan) =>
                  plan.groups.length > 0 &&
                  plan.groups.every((group) => group.workgroupShapes?.length > 0),
              ),
              'Every batched sampler operation must execute on WebGPU',
            );
            const [left, right] = await Promise.all([expected.read(), actual.read()]);
            const a = words(left),
              b = words(right);
            for (let i = 0; i < a.length; i++)
              assert(
                a[i] === b[i],
                `Batch bit mismatch padded=${padded} channels=${channels} align=${alignCorners} at ${i}: old=${left[i]} [${a[i]}], new=${right[i]} [${b[i]}]`,
              );
            const batchElements = outHeight * outWidth * channels;
            assert(
              a
                .subarray(0, batchElements)
                .some((value, index) => value !== a[batchElements + index]),
              'Batch fixture must distinguish the two images',
            );
            batchCases.push({
              batches,
              height,
              width,
              paddedHeight,
              paddedWidth,
              padTop,
              padLeft,
              channels,
              alignCorners,
              outHeight,
              outWidth,
              comparedElements: a.length,
              independentBatchValues: true,
              bitwiseEqual: true,
              allGroupsWebgpu: true,
            });
          } finally {
            for (const buffer of buffers) buffer.destroy();
          }
        }
      }
    }
    await device.queue.onSubmittedWorkDone();
    assert(validationErrors.length === 0, validationErrors.join('\n'));
    return {
      scope:
        'Exact same-device normalization and normalized-sampling regression; no speed or full-model claim',
      adapter: {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      },
      cases,
      sampleCases,
      batchCases,
      validationErrors,
    };
  } finally {
    runtime?.destroy();
    configure({ runtime: null });
    configure({});
    device.destroy();
  }
}
