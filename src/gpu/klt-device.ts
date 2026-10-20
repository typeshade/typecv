import type { Resident } from 'typeshade/runtime';
import type { ShadeDevice, ShadePlanReport } from './device.ts';
import { kltCheck, kltPyrDown, kltScharr, kltTrack } from './klt.shade.ts';
import { kltLayout, kltSeedRecords, type KltLayout, type KltOptions } from './klt.ts';

export interface KltGpuOptions {
  /** The tracks of one chunk of `kltTrack` and `kltCheck`. Default: all in one chunk. */
  readonly chunkTracks?: number;
}

export interface KltGpuResult {
  /** (x, y, forward-backward error, status) for each point, as `checkKltReference`. */
  readonly records: Float32Array;
  /** The forward result (x, y, NCC, status). */
  readonly forward: Float32Array;
  readonly back: Float32Array;
  /** The forward and backward pyramids as built on the device, for the tests. */
  readonly pyramids?: { gray: Float32Array[]; grad: Float32Array[] };
  readonly report: ShadePlanReport;
}

const LANES = 64;

function controlOf(layout: KltLayout, width: number, height: number, options: KltOptions) {
  return {
    width,
    height,
    levels: layout.levels,
    halfWindow: options.halfWindow,
    maxIterations: options.maxIterations,
    epsilon: options.epsilonPx,
    minEig: options.minEigThreshold,
    fbThreshold: options.fbThresholdPx,
    nccThreshold: options.nccThreshold,
    photometric: options.photometric,
  };
}

/** Record the pyramid and the gradient of a frame into an open plan body. */
function recordPyramid(
  layout: KltLayout,
  gray: Resident<Float32Array>,
  grad: Resident<Float32Array>,
): void {
  for (let l = 0; l + 1 < layout.levels; l++) {
    const level = {
      width: layout.widths[l + 1]!,
      height: layout.heights[l + 1]!,
      offset: layout.offsets[l + 1]!,
      srcWidth: layout.widths[l]!,
      srcHeight: layout.heights[l]!,
      srcOffset: layout.offsets[l]!,
    };
    kltPyrDown(
      { kltPyramid: gray, kltLevel: level },
      Math.ceil((level.width * level.height) / LANES),
    );
  }
  for (let l = 0; l < layout.levels; l++) {
    const level = {
      width: layout.widths[l]!,
      height: layout.heights[l]!,
      offset: layout.offsets[l]!,
      srcWidth: 0,
      srcHeight: 0,
      srcOffset: 0,
    };
    kltScharr(
      { kltGrayIn: gray, kltGradOut: grad, kltLevel: level },
      Math.ceil((level.width * level.height) / LANES),
    );
  }
}

/**
 * Track `points` (x, y pairs) from the gray frame `prev` to `cur` on the device, then back, and
 * check both: the pyramid, the gradient, `kltTrack` twice and `kltCheck`
 * (`docs/klt-tracking.md`, section 4.2). `trackKlt` of `klt.ts` is the oracle. Each of the
 * track kernels runs as chunks of `chunkTracks` tracks, and any split gives the same bits.
 */
export function trackKltGpu(
  device: ShadeDevice,
  prev: ArrayLike<number>,
  cur: ArrayLike<number>,
  width: number,
  height: number,
  points: ArrayLike<number>,
  options: KltOptions,
  gpu: KltGpuOptions & { readonly readPyramids?: boolean } = {},
): Promise<KltGpuResult> {
  const layout = kltLayout(width, height, options.levels),
    n = points.length >> 1,
    chunk = gpu.chunkTracks ?? Math.max(n, 1);
  if (prev.length !== width * height || cur.length !== width * height)
    throw new RangeError('Each frame must hold width x height values');
  if (!(chunk >= 1)) throw new RangeError('chunkTracks must be at least 1');
  const control = controlOf(layout, width, height, options),
    padded = (gray: ArrayLike<number>) => {
      const values = new Float32Array(layout.total);
      for (let i = 0; i < width * height; i++) values[i] = gray[i]!;
      return values;
    };
  return device.enqueue(async () => {
    const owned: Resident<Float32Array>[] = [],
      own = <T extends Resident<Float32Array>>(buffer: T) => (owned.push(buffer), buffer);
    try {
      const grayA = own(device.allocate(layout.total, padded(prev))),
        gradA = own(device.allocate(2 * layout.total)),
        grayB = own(device.allocate(layout.total, padded(cur))),
        gradB = own(device.allocate(2 * layout.total)),
        start = own(device.allocate(4 * Math.max(n, 1), kltSeedRecords(points))),
        forward = own(device.allocate(4 * Math.max(n, 1))),
        back = own(device.allocate(4 * Math.max(n, 1))),
        checked = own(device.allocate(4 * Math.max(n, 1)));
      /**
       * The chunks of one pass over the n tracks as plan steps. A forced `chunkTracks` fixes the
       * size. Otherwise the device sizes each chunk from the rate it measured under `key`
       * (section 4.4), and a chunk before the last fills its workgroups. The warm step runs the
       * dispatch over zero workgroups, so the pipeline is made before the first timed chunk.
       */
      function* chunks(
        key: string | undefined,
        run: (range: { first: number; end: number }, groups: number) => void,
      ) {
        const warm =
          key === undefined ? undefined : { id: key, run: () => run({ first: 0, end: 0 }, 0) };
        for (let first = 0; first < n;) {
          const remaining = n - first;
          let size = Math.min(remaining, chunk);
          if (gpu.chunkTracks === undefined && key !== undefined) {
            size = Math.min(
              remaining,
              device.chunkSize(key, remaining, Math.min(remaining, LANES)),
            );
            if (size < remaining) size = Math.max(LANES, size - (size % LANES));
          }
          const range = { first, end: first + size };
          const step = ['chunk', () => run(range, Math.ceil(size / LANES))] as const;
          yield key === undefined || warm === undefined
            ? step
            : ([...step, undefined, { key, units: size, warm }] as const);
          first += size;
        }
      }
      function* steps() {
        yield [
          'kltPyramid',
          () => {
            recordPyramid(layout, grayA, gradA);
            recordPyramid(layout, grayB, gradB);
          },
        ] as const;
        yield* chunks('kltTrack', (kltRange, groups) =>
          kltTrack(
            {
              kltControl: control,
              kltRange,
              kltGray: grayA,
              kltGrad: gradA,
              kltTarget: grayB,
              kltStart: start,
              kltNext: forward,
            },
            groups,
          ),
        );
        yield* chunks('kltTrack', (kltRange, groups) =>
          kltTrack(
            {
              kltControl: control,
              kltRange,
              kltGray: grayB,
              kltGrad: gradB,
              kltTarget: grayA,
              kltStart: forward,
              kltNext: back,
            },
            groups,
          ),
        );
        yield* chunks(undefined, (kltRange, groups) =>
          kltCheck(
            {
              kltControl: control,
              kltRange,
              kltStart: start,
              kltForward: forward,
              kltBack: back,
              kltChecked: checked,
            },
            groups,
          ),
        );
      }
      const report = await device.planSteps(steps());
      const read = async (buffer: Resident<Float32Array>, length: number) =>
        (await device.read(buffer, length)).slice(0, length);
      const pyramids = gpu.readPyramids
        ? {
            gray: [await read(grayA, layout.total), await read(grayB, layout.total)],
            grad: [await read(gradA, 2 * layout.total), await read(gradB, 2 * layout.total)],
          }
        : undefined;
      return {
        records: await read(checked, 4 * n),
        forward: await read(forward, 4 * n),
        back: await read(back, 4 * n),
        ...(pyramids ? { pyramids } : {}),
        report,
      };
    } finally {
      for (const buffer of owned) device.recycle(buffer);
    }
  });
}
