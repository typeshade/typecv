import type { Resident } from 'typeshade/runtime';
import type { ShadeDevice } from './device.ts';
import { rgbaToGray } from './features.shade.ts';
import { kltChunkSteps, kltControlOf, recordKltPyramid, KLT_LANES } from './klt-device.ts';
import {
  kltCheck,
  kltCompact,
  kltEigenMap,
  kltEigenMax,
  kltEigenMaxFinal,
  kltSelect,
  kltSuppress,
  kltTrack,
} from './klt.shade.ts';
import {
  KLT_END,
  kltAlive,
  kltCellGrid,
  kltOccupancy,
  kltLayout,
  kltOptionsFor,
  kltReason,
  type KltOptions,
} from './klt.ts';
import { planFlagScan, scanBlockCount } from './scan.ts';
import { packRgba, type GpuResidentSift } from './sift.ts';

/** The most tracks a tracker holds (section 4.1). The default capacity is lower (`klt.ts`). */
export const KLT_MAX_CAPACITY = 8192;

/** A frame the tracker takes: a gray image in [0, 1], RGBA bytes, or a SIFT result's gray. */
export type KltFrame =
  | { readonly gray: ArrayLike<number> }
  | { readonly rgba: Uint8Array | Uint8ClampedArray }
  | { readonly sift: GpuResidentSift };

export interface KltTrackerOptions {
  /** Overrides of `kltOptionsFor(width)`. */
  readonly options?: Partial<KltOptions>;
  /** A forced chunk of `kltTrack` and `kltEigenMap`, for the tests. */
  readonly chunkTracks?: number;
  /** The fewest tracks of a chunk that the device sizes (`KltGpuOptions.minChunkTracks`). */
  readonly minChunkTracks?: number;
}

export interface KltPushOptions {
  /**
   * Select new points on this frame (section 3.4). The first frame always selects. Step 5 sets
   * this at each keyframe.
   */
  readonly keyframe?: boolean;
}

export interface KltStep {
  /** The index of the frame, from 0. */
  readonly frame: number;
  /** The tracks that went into the step (alive before it). */
  readonly tracked: number;
  /** The tracks alive after the step, new points included. */
  readonly alive: number;
  /** Tracks that ended in this step, by end reason (index `KLT_END.*`). */
  readonly ended: readonly number[];
  /** Points that selection found, and the ones appended within the capacity. */
  readonly selected: number;
  readonly appended: number;
  /** The largest lambda-min of the frame, 0 when selection did not run. */
  readonly maxEigen: number;
  /** The tracks of each chunk of `kltTrack` and the pixels of each chunk of `kltEigenMap`. */
  readonly trackChunks: readonly number[];
  readonly eigenChunks: readonly number[];
  /** Wall time in ms: the pyramid, tracking and check; and selection. */
  readonly msTrack: number;
  readonly msSelect: number;
}

/** The alive tracks, as flat arrays in the frame of the last push. */
export interface KltTracks {
  readonly count: number;
  readonly id: Uint32Array;
  readonly x: Float32Array;
  readonly y: Float32Array;
  /** The frame index and the position at the seed. */
  readonly seedFrame: Uint32Array;
  readonly seedX: Float32Array;
  readonly seedY: Float32Array;
  /** The seed keypoint, or 0xFFFFFFFF for a fill-in track (section 4.1, word 5). */
  readonly seedKeypoint: Uint32Array;
  /** Sampled frames since the seed. */
  readonly age: Uint32Array;
}

export interface KltTracker {
  readonly width: number;
  readonly height: number;
  readonly options: KltOptions;
  /** The bytes of the arrays that the tracker holds on the device (`kltResidentBytes`). */
  readonly residentBytes: number;
  push(frame: KltFrame, push?: KltPushOptions): Promise<KltStep>;
  tracks(): KltTracks;
  /** The lambda-min map of the last frame that selection ran on, for the tests. */
  readEigenMap(): Promise<Float32Array>;
  dispose(): Promise<void>;
}

/**
 * The bytes of the device arrays of a tracker (section 4.1): two slots of gray and gradient
 * pyramids, the lambda-min map with its partial maxima, four track record arrays of `capacity`
 * records, and the selection scratch of the cells of `minDistancePx`.
 */
export function kltResidentBytes(width: number, height: number, options: KltOptions): number {
  const { total } = kltLayout(width, height, options.levels),
    { cellsX, cellsY } = kltCellGrid(width, height, options.minDistancePx),
    cells = cellsX * cellsY,
    words =
      2 * total + // kltGray, two slots
      2 * 2 * total + // kltGrad, two slots
      width * height + // kltEig
      Math.ceil((width * height) / 64) + // kltPartial
      1 + // kltMax
      4 * 4 * options.capacity + // start, forward, back and checked records
      2 * cells + // kltCells
      cells + // kltFlags
      2 * scanBlockCount(cells) + // block sums
      2 + // totals
      2 * 4 * cells; // kltPicks, kltPickList
  return 4 * words;
}

interface Slot {
  gray: Resident<Float32Array>;
  grad: Resident<Float32Array>;
}

const FILL_IN = 0xffffffff;

/**
 * A Lucas-Kanade tracker that keeps its frames and scratch on the device (`docs/klt-tracking.md`,
 * section 5.1). Each `push` builds the pyramid and the gradient of the frame, tracks the alive
 * tracks from the previous frame with the forward-backward check, and on the first frame and at a
 * keyframe selects new points by the Shi-Tomasi measure in the cells that hold no track. The
 * table of tracks is on the host between pushes: it is read once per push, with the records
 * the check wrote, and uploaded again as the start of the next push.
 */
export async function createKltTracker(
  device: ShadeDevice,
  width: number,
  height: number,
  tracker: KltTrackerOptions = {},
): Promise<KltTracker> {
  const options = kltOptionsFor(width, tracker.options),
    layout = kltLayout(width, height, options.levels),
    capacity = options.capacity,
    pixels = width * height,
    { cellsX, cellsY } = kltCellGrid(width, height, options.minDistancePx),
    cells = cellsX * cellsY,
    strips = Math.ceil(pixels / 64),
    blocks = scanBlockCount(cells);
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > KLT_MAX_CAPACITY)
    throw new RangeError(`capacity must be an integer from 1 to ${KLT_MAX_CAPACITY}`);
  if (!Number.isInteger(options.minDistancePx) || options.minDistancePx < 1)
    throw new RangeError('minDistancePx must be a positive integer');
  if (width < 2 * options.halfWindow + 2 || height < 2 * options.halfWindow + 2)
    throw new RangeError('The frame is too small for the tracking window');
  const control = kltControlOf(layout, width, height, options),
    selectControl = {
      width,
      height,
      margin: options.halfWindow,
      cell: options.minDistancePx,
      cellsX,
      cellsY,
      quality: options.qualityLevel,
      minEigBlock: Math.fround(Math.fround(options.minEigThreshold) * 9),
    },
    residentBytes = kltResidentBytes(width, height, options);

  const owned = new Set<Resident<Float32Array> | Resident<Uint32Array>>();
  const f32 = (length: number) => {
    const buffer = device.allocate(length);
    owned.add(buffer);
    return buffer;
  };
  const u32 = (length: number) => {
    const buffer = device.allocateU32(length);
    owned.add(buffer);
    return buffer;
  };
  const buffers = await device.enqueue(async () => {
    try {
      return {
        slots: [0, 1].map(() => ({ gray: f32(layout.total), grad: f32(2 * layout.total) })) as [
          Slot,
          Slot,
        ],
        eig: f32(pixels),
        partial: f32(strips),
        max: f32(1),
        start: f32(4 * capacity),
        forward: f32(4 * capacity),
        back: f32(4 * capacity),
        checked: f32(4 * capacity),
        occupied: f32(2 * cells),
        flags: u32(cells),
        blockSums: u32(2 * blocks),
        totals: u32(2),
        picks: f32(4 * cells),
        pickList: f32(4 * cells),
      };
    } catch (error) {
      for (const buffer of owned) device.drop(buffer);
      throw error;
    }
  });
  const { slots, eig, partial, max, start, forward, back, checked } = buffers;

  // The table on the host. Dead tracks are compacted away after each push.
  const table = {
    n: 0,
    id: new Uint32Array(capacity),
    x: new Float32Array(capacity),
    y: new Float32Array(capacity),
    seedFrame: new Uint32Array(capacity),
    seedX: new Float32Array(capacity),
    seedY: new Float32Array(capacity),
    seedKeypoint: new Uint32Array(capacity),
    age: new Uint32Array(capacity),
  };
  let frames = 0,
    nextId = 0,
    previous: Slot | undefined,
    disposed = false;

  const chunkPlan = (count: number, key: string) => ({
    count,
    key,
    ...(tracker.chunkTracks === undefined ? {} : { fixed: tracker.chunkTracks }),
    min: tracker.minChunkTracks ?? KLT_LANES,
  });

  /** Remove the dead tracks, and the aged ones, from the table. Returns the ends by reason. */
  function update(records: Float32Array, n: number): number[] {
    const ended = new Array<number>(8).fill(0);
    let out = 0;
    for (let k = 0; k < n; k++) {
      const status = records[4 * k + 3]!;
      let reason: number = kltAlive(status) ? KLT_END.alive : kltReason(status);
      const age = table.age[k]! + 1;
      if (reason === KLT_END.alive && age > options.maxAgeFrames) reason = KLT_END.age;
      if (reason !== KLT_END.alive) {
        ended[reason]!++;
        continue;
      }
      table.id[out] = table.id[k]!;
      table.x[out] = records[4 * k]!;
      table.y[out] = records[4 * k + 1]!;
      table.seedFrame[out] = table.seedFrame[k]!;
      table.seedX[out] = table.seedX[k]!;
      table.seedY[out] = table.seedY[k]!;
      table.seedKeypoint[out] = table.seedKeypoint[k]!;
      table.age[out] = age;
      out++;
    }
    table.n = out;
    return ended;
  }

  async function push(frame: KltFrame, pushOptions: KltPushOptions = {}): Promise<KltStep> {
    if (disposed) throw new Error('The KLT tracker is disposed');
    return device.enqueue(async () => {
      if (disposed) throw new Error('The KLT tracker is disposed');
      const index = frames,
        slot = slots[index % 2]!,
        n = table.n;
      let rgba: Resident<Uint32Array> | undefined;
      if ('sift' in frame) {
        const kept = frame.sift.gray;
        if (
          !kept ||
          kept.levels !== options.levels ||
          kept.width !== width ||
          kept.height !== height
        )
          throw new RangeError(
            'The SIFT result holds no gray of this tracker (GpuSiftOptions.grayForKlt)',
          );
        // The SIFT buffer becomes level 0 of this slot; the slot's own buffer goes back.
        const taken = kept.take();
        device.recycle(slot.gray);
        owned.delete(slot.gray);
        slot.gray = taken;
        owned.add(taken);
      } else if ('gray' in frame) {
        if (frame.gray.length !== pixels)
          throw new RangeError('The frame must hold width x height values');
        const padded = new Float32Array(layout.total);
        for (let i = 0; i < pixels; i++) padded[i] = frame.gray[i]!;
        slot.gray.write(padded);
      } else {
        if (frame.rgba.length !== 4 * pixels)
          throw new RangeError('The frame must hold width x height RGBA bytes');
        rgba = device.allocateU32(pixels, packRgba(frame.rgba, pixels));
      }

      // Phase 1: the pyramid, then the tracks of the previous frame and the check.
      const t0 = performance.now(),
        trackChunks: number[] = [];
      const tracking = previous !== undefined && n > 0;
      if (tracking) {
        const padded = new Float32Array(4 * capacity);
        for (let k = 0; k < n; k++) {
          padded[4 * k] = table.x[k]!;
          padded[4 * k + 1] = table.y[k]!;
          padded[4 * k + 3] = 1;
        }
        start.write(padded);
      }
      const prev = previous;
      function* tracking1() {
        yield [
          'kltPyramid',
          () => {
            if (rgba)
              rgbaToGray(
                { rgba, gray: slot.gray, pixelWindow: { offset: 0, count: pixels } },
                Math.ceil(pixels / KLT_LANES),
              );
            recordKltPyramid(layout, slot.gray, slot.grad);
          },
        ] as const;
        if (!prev || n === 0) return;
        yield* kltChunkSteps(
          device,
          chunkPlan(n, 'kltTrack'),
          (kltRange, groups) =>
            kltTrack(
              {
                kltControl: control,
                kltRange,
                kltGray: prev.gray,
                kltGrad: prev.grad,
                kltTarget: slot.gray,
                kltStart: start,
                kltNext: forward,
              },
              groups,
            ),
          (size) => trackChunks.push(size),
        );
        yield* kltChunkSteps(
          device,
          chunkPlan(n, 'kltTrack'),
          (kltRange, groups) =>
            kltTrack(
              {
                kltControl: control,
                kltRange,
                kltGray: slot.gray,
                kltGrad: slot.grad,
                kltTarget: prev.gray,
                kltStart: forward,
                kltNext: back,
              },
              groups,
            ),
          (size) => trackChunks.push(size),
        );
        yield* kltChunkSteps(device, { count: n }, (kltRange, groups) =>
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
      try {
        await device.planSteps(tracking1());
      } finally {
        if (rgba) device.recycle(rgba);
      }
      let ended = new Array<number>(8).fill(0);
      if (tracking) ended = update(await device.read(checked, 4 * n), n);
      else for (let k = 0; k < table.n; k++) table.age[k] = table.age[k]! + 1;
      const msTrack = performance.now() - t0;

      // Phase 2: select new points in the cells that hold no track.
      let selected = 0,
        appended = 0,
        maxEigen = 0,
        msSelect = 0;
      const eigenChunks: number[] = [];
      if (index === 0 || pushOptions.keyframe) {
        const t1 = performance.now(),
          occupied = kltOccupancy(table.x, table.y, table.n, width, height, options.minDistancePx);
        buffers.occupied.write(occupied);
        function* selection() {
          yield* kltChunkSteps(
            device,
            chunkPlan(pixels, 'kltEigenMap'),
            (kltRange, groups) =>
              kltEigenMap(
                { kltSelectControl: selectControl, kltRange, kltGrad: slot.grad, kltEig: eig },
                groups,
              ),
            (size) => eigenChunks.push(size),
          );
          yield [
            'kltSelect',
            () => {
              kltEigenMax(
                { kltSelectControl: selectControl, kltEig: eig, kltPartial: partial },
                Math.ceil(strips / KLT_LANES),
              );
              kltEigenMaxFinal(
                { kltSelectControl: selectControl, kltPartial: partial, kltMax: max },
                1,
              );
              kltSelect(
                {
                  kltSelectControl: selectControl,
                  kltEig: eig,
                  kltMax: max,
                  kltCells: buffers.occupied,
                  kltPicks: buffers.picks,
                },
                Math.ceil(cells / KLT_LANES),
              );
              kltSuppress(
                {
                  kltSelectControl: selectControl,
                  kltCells: buffers.occupied,
                  kltPicks: buffers.picks,
                  kltFlags: buffers.flags,
                },
                Math.ceil(cells / KLT_LANES),
              );
              planFlagScan(
                device,
                { flags: buffers.flags, blockSums: buffers.blockSums, totals: buffers.totals },
                cells,
              );
              kltCompact(
                {
                  kltSelectControl: selectControl,
                  kltPicks: buffers.picks,
                  kltFlags: buffers.flags,
                  kltBlockSums: buffers.blockSums,
                  kltPickList: buffers.pickList,
                },
                Math.ceil(cells / KLT_LANES),
              );
            },
          ] as const;
        }
        await device.planSteps(selection());
        const count = (await device.read(buffers.totals, 2))[0]!,
          list = (await device.read(buffers.pickList, 4 * Math.max(count, 1))).slice(0, 4 * count);
        maxEigen = (await device.read(max, 1))[0]!;
        selected = count;
        const room = Math.max(capacity - table.n, 0),
          order = Array.from({ length: count }, (_, i) => i);
        // The strongest first when the capacity cuts the list; the cell order otherwise.
        if (count > room) order.sort((a, b) => list[4 * b + 2]! - list[4 * a + 2]! || a - b);
        for (const i of order.slice(0, room)) {
          const k = table.n++;
          table.id[k] = nextId++;
          table.x[k] = table.seedX[k] = list[4 * i]!;
          table.y[k] = table.seedY[k] = list[4 * i + 1]!;
          table.seedFrame[k] = index;
          table.seedKeypoint[k] = FILL_IN;
          table.age[k] = 0;
          appended++;
        }
        msSelect = performance.now() - t1;
      }
      previous = slot;
      frames++;
      return {
        frame: index,
        tracked: tracking ? n : 0,
        alive: table.n,
        ended,
        selected,
        appended,
        maxEigen,
        trackChunks,
        eigenChunks,
        msTrack,
        msSelect,
      };
    });
  }

  return {
    width,
    height,
    options,
    residentBytes,
    push,
    readEigenMap: () =>
      device.enqueue(async () => (await device.read(eig, pixels)).slice(0, pixels)),
    tracks: () => ({
      count: table.n,
      id: table.id.slice(0, table.n),
      x: table.x.slice(0, table.n),
      y: table.y.slice(0, table.n),
      seedFrame: table.seedFrame.slice(0, table.n),
      seedX: table.seedX.slice(0, table.n),
      seedY: table.seedY.slice(0, table.n),
      seedKeypoint: table.seedKeypoint.slice(0, table.n),
      age: table.age.slice(0, table.n),
    }),
    dispose: async () => {
      if (disposed) return;
      disposed = true;
      await device.enqueue(async () => {
        for (const buffer of owned) device.recycle(buffer);
        owned.clear();
      });
    },
  };
}
