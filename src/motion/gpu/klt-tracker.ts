import type { Resident } from 'typeshade/runtime';
import type { ShadeDevice } from '../../device/gpu/device.ts';
import { rgbaToGray } from '../../features/gpu/features.shade.ts';
import {
  kltChunkSteps,
  kltControlOf,
  kltRoundtripSteps,
  recordKltPyramid,
  KLT_LANES,
} from './klt-device.ts';
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
  KLT_MAX_CAPACITY,
  KltPointGrid,
  kltAlive,
  kltAssignNearest,
  kltCellGrid,
  kltOccupancy,
  kltOptionsFor,
  kltReason,
  type KltOptions,
} from './klt.ts';
import { kltLayout } from '../../image/gpu/pyramid-layout.ts';
import { planFlagScan, scanBlockCount } from '../../device/gpu/scan.ts';
import type { GpuResidentSift } from '../../features/gpu/sift.ts';
import { packRgba } from '../../image/gpu/pack-rgba.ts';

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
  /** Experimental: submit forward, backward and check together for each tracking chunk. Default false. */
  readonly batchPasses?: boolean;
  /**
   * Select fill-in points around the seeds at a join (section 3.4). True by default. False leaves
   * the join to the SIFT seeds alone, which is how step 6 measures the refresh.
   */
  readonly fillIn?: boolean;
}

export interface KltPushOptions {
  /**
   * Select new points on this frame (section 3.4). The first frame always selects. A caller that
   * joins the frame to SIFT keypoints (`join`) leaves this out and sets `select: false`: `join`
   * seeds the keypoints first and then selects.
   */
  readonly keyframe?: boolean;
  /** Overrides the rule above: true selects on this frame, false does not. */
  readonly select?: boolean;
  /** The id of the frame in the caller's numbering (a view id). The default is the frame index. */
  readonly frameId?: number;
}

/** What `join` takes: the SIFT keypoints of the frame of the last `push` (section 3.3). */
export interface KltJoinInput {
  /** The id given to that `push`. */
  readonly frameId: number;
  /** The keypoints as x, y pairs in pixels of the tracker's frame. */
  readonly keypoints: ArrayLike<number>;
  /**
   * The ids of the earlier keyframes that can take a match. A track keeps its anchor in any other
   * frame out of the pairs, and forgets it. All frames when absent.
   */
  readonly views?: Iterable<number>;
}

/** The matches of one earlier keyframe against the joined one, in the shape of a mapper pair. */
export interface KltJoinPair {
  readonly first: number;
  readonly second: number;
  /** Feature index pairs (first index, second index), in the order of the second index. */
  readonly matches: Uint32Array;
  /** The track id of each match. */
  readonly tracks: Uint32Array;
}

export interface KltJoin {
  readonly frameId: number;
  readonly pairs: readonly KltJoinPair[];
  /** Tracks that joined a keypoint, SIFT seeds added, fill-in points added, and picks that the host dropped. */
  readonly joined: number;
  readonly seeded: number;
  readonly filledIn: number;
  readonly filtered: number;
  /** The tracks alive after the join. */
  readonly alive: number;
  /** Wall time in ms of the join, selection included. */
  readonly ms: number;
}

/** The totals of a tracker's life, for the run record (section 5.2). */
export interface KltStats {
  readonly pushes: number;
  /** Tracks that ended, by end reason (index `KLT_END.*`), `epipolar` for the rejected ones. */
  readonly ended: readonly number[];
  readonly joins: number;
  readonly joined: number;
  readonly seeded: number;
  readonly filledIn: number;
  readonly filtered: number;
  /** Matches the joins emitted, summed over the pairs. */
  readonly matches: number;
  /** Tracks removed by `reject`. */
  readonly rejected: number;
  /** Tracks alive now. */
  readonly alive: number;
  /**
   * Tracks by frames lived since the seed, ended and alive: fewer than 2, 2, 3 to 4, 5 to 8, 9 to
   * 16, and 17 or more (`KLT_LENGTH_EDGES`).
   */
  readonly lengths: readonly number[];
  /** The mean of the frames lived since the seed, over the same tracks as `lengths`. */
  readonly meanLived: number;
  readonly msPush: number;
  readonly msSelect: number;
  readonly msJoin: number;
}

/** The lower edges of the bins of `KltStats.lengths`, after the first. */
export const KLT_LENGTH_EDGES = [2, 3, 5, 9, 17] as const;

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
  /** Picks that the host dropped for an alive track closer than `minDistancePx` (Step 5 record). */
  readonly filtered?: number;
  /** The largest lambda-min of the frame, 0 when selection did not run. */
  readonly maxEigen: number;
  /** Forward then backward chunk sizes (batched mode executes each pair together), and eigen pixels. */
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
  /**
   * Join the tracks to the SIFT keypoints of the frame of the last `push` (section 3.3), seed the
   * unclaimed keypoints, and select fill-in points around them. Returns the matches by pair.
   */
  join(keyframe: KltJoinInput): Promise<KltJoin>;
  /**
   * End the tracks with these ids as epipolar outliers (section 3.5, item 6). `kltReject` of
   * `src/motion/gpu/klt.ts` makes the list from the verdict of a pair verification. Returns the number
   * of tracks removed.
   */
  reject(trackIds: Iterable<number>): Promise<number>;
  /** The totals since the tracker was made. */
  stats(): KltStats;
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
    /** The frame index of the first seed of the track. */
    born: new Uint32Array(capacity),
    /** The keypoints the track joined, as flat (frame id, keypoint index) pairs, the newest last. */
    anchors: new Array<number[]>(capacity),
  };
  let frames = 0,
    nextId = 0,
    previous: Slot | undefined,
    lastFrameId = -1,
    disposed = false;
  let disposal: Promise<void> | undefined;
  const totals = {
    pushes: 0,
    ended: new Array<number>(8).fill(0),
    joins: 0,
    joined: 0,
    seeded: 0,
    filledIn: 0,
    filtered: 0,
    matches: 0,
    rejected: 0,
    lengths: new Array<number>(KLT_LENGTH_EDGES.length + 1).fill(0),
    livedSum: 0,
    livedCount: 0,
    msPush: 0,
    msSelect: 0,
    msJoin: 0,
  };
  const lengthBin = (lived: number) => {
    let bin = 0;
    while (bin < KLT_LENGTH_EDGES.length && lived >= KLT_LENGTH_EDGES[bin]!) bin++;
    return bin;
  };
  /** A track that ended, after `lived` frames since its seed. */
  const finished = (lived: number) => {
    totals.lengths[lengthBin(lived)]!++;
    totals.livedSum += lived;
    totals.livedCount++;
  };

  const chunkPlan = (count: number, key: string) => ({
    count,
    key,
    ...(tracker.chunkTracks === undefined ? {} : { fixed: tracker.chunkTracks }),
    min: tracker.minChunkTracks ?? (key === 'kltRoundtrip' ? 2 * KLT_LANES : KLT_LANES),
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
        totals.ended[reason]!++;
        // The step that failed is not a frame lived.
        finished(Math.max(frames - table.born[k]! - 1, 0));
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
      table.born[out] = table.born[k]!;
      table.anchors[out] = table.anchors[k]!;
      out++;
    }
    table.n = out;
    return ended;
  }

  /** The lambda-min map and its maximum for the frame of `slot`, as steps for `planSteps`. */
  function* eigenSteps(slot: Slot, eigenChunks: number[]) {
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
      'kltEigenMax',
      () => {
        kltEigenMax(
          { kltSelectControl: selectControl, kltEig: eig, kltPartial: partial },
          Math.ceil(strips / KLT_LANES),
        );
        kltEigenMaxFinal({ kltSelectControl: selectControl, kltPartial: partial, kltMax: max }, 1);
      },
    ] as const;
  }

  /** One pick of each empty cell, the suppression, the scan and the compaction. */
  function* pickSteps() {
    yield [
      'kltSelect',
      () => {
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

  /**
   * Selection from the map that `eigenSteps` made: the occupancy of the table goes up, the picks
   * come back, and the strongest fill the room. A pick closer than `minDistancePx` to an alive
   * track is dropped on the host. The occupancy holds one position for each cell, so the second
   * track of a cell (a SIFT seed) is out of its reach (Step 5 record, decision 1). Each new track
   * takes the keypoint it lies within the join radius of, if `keypoints` has one that `taken` leaves.
   */
  async function selectPicks(
    index: number,
    frameId: number,
    keypoints?: ArrayLike<number>,
    taken?: Uint8Array,
  ) {
    buffers.occupied.write(
      kltOccupancy(table.x, table.y, table.n, width, height, options.minDistancePx),
    );
    await device.planSteps(pickSteps());
    const count = (await device.read(buffers.totals, 2))[0]!,
      list = (await device.read(buffers.pickList, 4 * Math.max(count, 1))).slice(0, 4 * count);
    const room = Math.max(capacity - table.n, 0),
      order = Array.from({ length: count }, (_, i) => i);
    // The strongest first when the capacity cuts the list; the cell order otherwise.
    if (count > room) order.sort((a, b) => list[4 * b + 2]! - list[4 * a + 2]! || a - b);
    const grid = new KltPointGrid(options.minDistancePx);
    for (let k = 0; k < table.n; k++) grid.add(table.x[k]!, table.y[k]!);
    const chosen: number[] = [];
    let filtered = 0;
    for (const i of order) {
      if (chosen.length >= room) break;
      if (grid.near(list[4 * i]!, list[4 * i + 1]!, options.minDistancePx)) {
        filtered++;
        continue;
      }
      grid.add(list[4 * i]!, list[4 * i + 1]!);
      chosen.push(i);
    }
    const anchor = keypoints
      ? kltAssignNearest(
          chosen.map((i) => list[4 * i]!),
          chosen.map((i) => list[4 * i + 1]!),
          chosen.length,
          keypoints,
          options.joinRadiusPx,
          taken,
        )
      : undefined;
    chosen.forEach((i, c) => {
      const k = table.n++,
        j = anchor ? anchor[c]! : -1;
      table.id[k] = nextId++;
      table.x[k] = table.seedX[k] = list[4 * i]!;
      table.y[k] = table.seedY[k] = list[4 * i + 1]!;
      table.seedFrame[k] = frameId;
      table.seedKeypoint[k] = j >= 0 ? j : FILL_IN;
      table.age[k] = 0;
      table.born[k] = index;
      table.anchors[k] = j >= 0 ? [frameId, j] : [];
    });
    return { selected: count, appended: chosen.length, filtered };
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
        const passes: Parameters<typeof kltRoundtripSteps>[2] = [
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
          (kltRange, groups) =>
            kltCheck(
              {
                kltControl: control,
                kltCheckOutput: { nccOffset: 0 },
                kltRange,
                kltStart: start,
                kltForward: forward,
                kltBack: back,
                kltChecked: checked,
              },
              groups,
            ),
        ];
        if (tracker.batchPasses === true)
          yield* kltRoundtripSteps(device, chunkPlan(n, 'kltRoundtrip'), passes, trackChunks);
        else {
          yield* kltChunkSteps(device, chunkPlan(n, 'kltTrack'), passes[0], (size) =>
            trackChunks.push(size),
          );
          yield* kltChunkSteps(device, chunkPlan(n, 'kltTrack'), passes[1], (size) =>
            trackChunks.push(size),
          );
          yield* kltChunkSteps(device, { count: n }, passes[2]);
        }
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
        filtered = 0,
        maxEigen = 0,
        msSelect = 0;
      const eigenChunks: number[] = [];
      const frameId = pushOptions.frameId ?? index;
      if (pushOptions.select ?? (index === 0 || pushOptions.keyframe === true)) {
        const t1 = performance.now();
        await device.planSteps(eigenSteps(slot, eigenChunks));
        maxEigen = (await device.read(max, 1))[0]!;
        const picked = await selectPicks(index, frameId);
        selected = picked.selected;
        appended = picked.appended;
        filtered = picked.filtered;
        msSelect = performance.now() - t1;
      }
      totals.pushes++;
      totals.msPush += msTrack;
      totals.msSelect += msSelect;
      totals.filtered += filtered;
      lastFrameId = frameId;
      previous = slot;
      frames++;
      return {
        frame: index,
        tracked: tracking ? n : 0,
        alive: table.n,
        ended,
        selected,
        appended,
        filtered,
        maxEigen,
        trackChunks,
        eigenChunks,
        msTrack,
        msSelect,
      };
    });
  }

  const MAX_ANCHORS = 8;

  async function join(keyframe: KltJoinInput): Promise<KltJoin> {
    if (disposed) throw new Error('The KLT tracker is disposed');
    return device.enqueue(async () => {
      if (disposed) throw new Error('The KLT tracker is disposed');
      if (!previous || lastFrameId !== keyframe.frameId)
        throw new RangeError('join() takes the keypoints of the frame of the last push()');
      const started = performance.now(),
        { frameId } = keyframe,
        keypoints = keyframe.keypoints,
        total = keypoints.length >> 1,
        index = frames - 1,
        radius = options.joinRadiusPx,
        views = keyframe.views === undefined ? undefined : new Set(keyframe.views),
        usable = (id: number) => id !== frameId && (!views || views.has(id));
      // The lambda-min map of the frame: the test of the seeds and of the new position.
      const mapStarted = performance.now();
      await device.planSteps(eigenSteps(previous, []));
      const map = (await device.read(eig, pixels)).slice(0, pixels),
        floor = selectControl.minEigBlock,
        mapMs = performance.now() - mapStarted,
        eigAt = (x: number, y: number) =>
          map[
            Math.min(Math.max(Math.round(y), 0), height - 1) * width +
              Math.min(Math.max(Math.round(x), 0), width - 1)
          ]!;

      // Steps 1 and 2 of section 3.3: the nearest keypoint within the radius, if no other track has it.
      const n = table.n,
        assigned = kltAssignNearest(table.x, table.y, n, keypoints, radius),
        claimed = new Uint8Array(total),
        groups = new Map<number, { rows: [number, number, number][] }>();
      let joined = 0;
      for (let k = 0; k < n; k++) {
        const j = assigned[k]!;
        // Anchors in frames that cannot take a pair are forgotten, whether or not the track joins.
        const kept = table.anchors[k]!.filter((_, i) => usable(table.anchors[k]![i - (i % 2)]!));
        table.anchors[k] = kept;
        if (j < 0) continue;
        claimed[j] = 1;
        joined++;
        for (let i = 0; i < kept.length; i += 2) {
          const group = groups.get(kept[i]!) ?? { rows: [] };
          group.rows.push([j, kept[i + 1]!, table.id[k]!]);
          groups.set(kept[i]!, group);
        }
        // Re-anchor: the track follows the keypoint if the keypoint is a place to track.
        const x = keypoints[2 * j]!,
          y = keypoints[2 * j + 1]!;
        if (eigAt(x, y) >= floor) {
          table.x[k] = x;
          table.y[k] = y;
        }
        table.seedFrame[k] = frameId;
        table.seedX[k] = x;
        table.seedY[k] = y;
        table.seedKeypoint[k] = j;
        table.age[k] = 0;
        kept.push(frameId, j);
        if (kept.length > 2 * MAX_ANCHORS) kept.splice(0, kept.length - 2 * MAX_ANCHORS);
      }

      // Step 4: a keypoint that no track claimed becomes a seed. Strongest first, one for each
      // join radius, within the room the capacity leaves.
      const grid = new KltPointGrid(Math.max(radius, 1));
      for (let k = 0; k < table.n; k++) grid.add(table.x[k]!, table.y[k]!);
      const candidates: number[] = [];
      for (let j = 0; j < total; j++)
        if (!claimed[j] && eigAt(keypoints[2 * j]!, keypoints[2 * j + 1]!) >= floor)
          candidates.push(j);
      candidates.sort(
        (a, b) =>
          eigAt(keypoints[2 * b]!, keypoints[2 * b + 1]!) -
            eigAt(keypoints[2 * a]!, keypoints[2 * a + 1]!) || a - b,
      );
      let seeded = 0;
      for (const j of candidates) {
        if (table.n >= capacity) break;
        const x = keypoints[2 * j]!,
          y = keypoints[2 * j + 1]!;
        if (grid.near(x, y, radius)) continue;
        grid.add(x, y);
        claimed[j] = 1;
        const k = table.n++;
        table.id[k] = nextId++;
        table.x[k] = table.seedX[k] = x;
        table.y[k] = table.seedY[k] = y;
        table.seedFrame[k] = frameId;
        table.seedKeypoint[k] = j;
        table.age[k] = 0;
        table.born[k] = index;
        table.anchors[k] = [frameId, j];
        seeded++;
      }

      // Fill-in points in the cells that no track holds, each anchored to a free keypoint near it.
      const pickStarted = performance.now(),
        picked =
          tracker.fillIn === false
            ? { appended: 0, filtered: 0 }
            : await selectPicks(index, frameId, keypoints, claimed);
      totals.msSelect += mapMs + (performance.now() - pickStarted);

      const pairs: KltJoinPair[] = [];
      let matches = 0;
      for (const first of [...groups.keys()].sort((a, b) => a - b)) {
        const rows = groups.get(first)!.rows.sort((a, b) => a[0] - b[0] || a[1] - b[1]),
          pair = new Uint32Array(2 * rows.length),
          ids = new Uint32Array(rows.length);
        rows.forEach(([second, firstIndex, id], r) => {
          pair[2 * r] = firstIndex;
          pair[2 * r + 1] = second;
          ids[r] = id;
        });
        pairs.push({ first, second: frameId, matches: pair, tracks: ids });
        matches += rows.length;
      }
      const ms = performance.now() - started;
      totals.joins++;
      totals.joined += joined;
      totals.seeded += seeded;
      totals.filledIn += picked.appended;
      totals.filtered += picked.filtered;
      totals.matches += matches;
      totals.msJoin += ms;
      return {
        frameId,
        pairs,
        joined,
        seeded,
        filledIn: picked.appended,
        filtered: picked.filtered,
        alive: table.n,
        ms,
      };
    });
  }

  async function reject(trackIds: Iterable<number>): Promise<number> {
    const ids = new Set(trackIds);
    if (ids.size === 0) return 0;
    return device.enqueue(async () => {
      let out = 0;
      for (let k = 0; k < table.n; k++) {
        if (ids.has(table.id[k]!)) {
          totals.ended[KLT_END.epipolar]!++;
          totals.rejected++;
          finished(Math.max(frames - 1 - table.born[k]!, 0));
          continue;
        }
        if (out !== k) {
          table.id[out] = table.id[k]!;
          table.x[out] = table.x[k]!;
          table.y[out] = table.y[k]!;
          table.seedFrame[out] = table.seedFrame[k]!;
          table.seedX[out] = table.seedX[k]!;
          table.seedY[out] = table.seedY[k]!;
          table.seedKeypoint[out] = table.seedKeypoint[k]!;
          table.age[out] = table.age[k]!;
          table.born[out] = table.born[k]!;
          table.anchors[out] = table.anchors[k]!;
        }
        out++;
      }
      const removed = table.n - out;
      table.n = out;
      return removed;
    });
  }

  function stats(): KltStats {
    const lengths = totals.lengths.slice();
    let sum = totals.livedSum,
      count = totals.livedCount;
    for (let k = 0; k < table.n; k++) {
      const lived = Math.max(frames - 1 - table.born[k]!, 0);
      lengths[lengthBin(lived)]!++;
      sum += lived;
      count++;
    }
    const { livedSum, livedCount, ...rest } = totals;
    void livedSum;
    void livedCount;
    return {
      ...rest,
      ended: totals.ended.slice(),
      alive: table.n,
      lengths,
      meanLived: count ? sum / count : 0,
    };
  }

  return {
    width,
    height,
    options,
    residentBytes,
    push,
    join,
    reject,
    stats,
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
    dispose: () => {
      if (disposal) return disposal;
      disposed = true;
      disposal = device.enqueue(async () => {
        for (const buffer of owned) device.recycle(buffer);
        owned.clear();
      });
      return disposal;
    },
  };
}
