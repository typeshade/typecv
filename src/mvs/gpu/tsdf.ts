/**
 * The TSDF fusion on the device (`docs/dense-mvs.md`, section 10.4): the kernels of
 * `tsdf.shade.ts` over the filtered depth maps, within the device's working budget. The host plans
 * the volume (`planTsdfVolume`), counts the blocks with an allocation-only dry run, and cuts the
 * volume into boxes that fit the budget (`planTsdfBoxes`). Each box reuses its discovered keys,
 * integrates and counts free space from every view, then extracts its points to the host before
 * the next box. A box extracts only its own blocks, so no point repeats, and its margin holds the
 * neighbours its voxels read. The budget changes the speed and not the points.
 */
import type { Resident } from 'typeshade/runtime';
import { readTogether } from '../../device/gpu/readback.ts';
import { tsdfBoxTable } from './tsdf-box-table.ts';
import { TsdfViewChannels } from './tsdf-view-channels.ts';
import { TsdfUploadWindow, type TsdfUploadWindowReport } from './tsdf-upload-window.ts';
import type { GpuStep, ShadeDevice, ShadePlanReport } from '../../device/gpu/device.ts';
import {
  tsdfAllocate,
  tsdfAssign,
  tsdfCarve,
  tsdfExtract,
  tsdfIntegrate,
  tsdfViewCarve,
  tsdfResetVotes,
} from './tsdf.shade.ts';
import {
  TSDF_BLOCK_VOXELS,
  TSDF_BLOCK_WORDS,
  TSDF_HEADER,
  TSDF_POINT_WORDS,
  TSDF_WEIGHT_SCALE,
  TSDF_CARVE_SCALE,
  tsdfVisibilityGroups,
  TSDF_WHOLE,
  checkedTsdfOptions,
  planTsdfBoxes,
  planTsdfVolume,
  sortOwnedTsdfPoints,
  type TsdfBox,
  type TsdfInput,
  type TsdfOptions,
  type TsdfPoints,
  type TsdfVolume,
} from '../tsdf.ts';

export interface TsdfGpuOptions extends TsdfOptions {
  /** The most bytes the fusion keeps on the device. Default: the device's free working bytes. */
  readonly residentBytes?: number;
  /** @internal Cut the volume into passes of at most this many blocks (a test of the boxes). */
  readonly maxBlocks?: number;
  /** Logical view payload allowed between confirmed completions; default 64 MiB, at least one view. */
  readonly maxQueuedViewBytes?: number;
  /** @internal Exact old reset-kernel sequence retained only as a differential-test oracle. */
  readonly standaloneVoteReset?: boolean;
  /** Work submitted by each chunk; readback marks completion. No extra GPU fence is added. */
  readonly onProgress?: (event: TsdfProgress) => void;
  /** Each extracted box, before the final sort. Own arrays: the callback may transfer them. */
  readonly onPart?: (part: TsdfPart) => void | Promise<void>;
  readonly signal?: AbortSignal;
}

export type TsdfPhase =
  'allocate' | 'assign' | 'integrate' | 'carve' | 'count' | 'extract' | 'readback' | 'sort';
export interface TsdfProgress {
  readonly phase: TsdfPhase;
  /** 0 while discovering the volume; otherwise the current box, starting at 1. */
  readonly pass: number;
  /** 0 until the volume's boxes are known. */
  readonly passes: number;
  /** Input view index, or -1 for work on the volume. */
  readonly view: number;
  readonly views: number;
  /** Units submitted within this operation, not an estimate of the entire run's time. */
  readonly done: number;
  readonly total: number;
  /** Submitted units across all operations of this phase in the current region. */
  readonly phaseDone?: number;
  readonly phaseTotal?: number;
  readonly points: number;
}
export interface TsdfPart {
  readonly positions: Float32Array;
  /** Packed normal, color, weight, free-space count and view tally per point. */
  readonly data: Uint32Array;
}

/** What the fusion did: the volume, the passes and the bytes. */
export interface TsdfReport {
  /** v_0 in the frame of the input, and the levels. */
  readonly voxel: number;
  readonly levels: number;
  /** The blocks the dry run claimed, and the passes (boxes) the volume was cut into. */
  readonly blocks: number;
  readonly passes: number;
  /** The slots of the dry run's table. */
  readonly tableSlots: number;
  /** The most bytes the fusion held on the device at once. */
  readonly peakBytes: number;
  /** View bytes uploaded during discovery, integration and visibility checks. */
  readonly uploadBytes: number;
  /** Additional TypeCV host channel storage; borrowed input arrays and runtime snapshots excluded. */
  readonly hostViewStagingBytes?: number;
  /** Pixels visited by discovery, excluding box tables reused from its complete key set. */
  readonly allocationPixels?: number;
  /** Box tables built from complete discovery keys without replaying all input pixels. */
  readonly reusedBoxTables?: number;
  readonly inlineVoteResetGroups?: number;
  /** Submitted view-payload watermark; excludes native allocations, other buffers and process RSS. */
  readonly queuedViewUploads?: TsdfUploadWindowReport;
  /** Synchronous wall sections, not a total-time partition; excludes awaited GPU execution. */
  readonly hostTimings?: {
    readonly volumePlanningMs: number;
    readonly boxPlanningMs: number;
    readonly boxTableMs: number;
    /** Includes the runtime's input snapshots and upload enqueue, without waiting for execution. */
    readonly viewUploadMs: number;
    readonly sortingMs: number;
  };
  readonly report: ShadePlanReport;
}

const EMPTY = 0xffffffff;
/** Bytes of one block: its voxels, and two slots of the table at three words each. */
const BLOCK_BYTES = 4 * TSDF_BLOCK_WORDS + 2 * 3 * 4;
/** The points one block is expected to give (the extraction falls back to smaller ranges). */
const POINTS_PER_BLOCK = 96;
const POINT_BYTES = 12 + 4 * TSDF_POINT_WORDS;

/**
 * The occupancy floor of a TSDF chunk: 16,384 pixels, slots or voxels (256 workgroups), the
 * `minUnits` of `ShadeDevice.chunkSize` (GPU throughput item 2). One element is a walk of at most a
 * few hundred cells, so the floor stays far below `GPU_CEILING_MS`, and the device halves it after
 * a dispatch that overruns the ceiling. A software adapter honors no occupancy floor
 * (`ShadeDeviceOptions.occupancyFloors`), and there the fenced time of a small chunk is mostly the
 * fence: the rate learned from it held every chunk small and bound the fusion by its dispatches
 * (1,030 s and 127,000 dispatches on `pipes`, the step 3 record). On such a device the floor is the
 * chunk's `min` instead. A software adapter has no GPU watchdog to stay below.
 */
export const TSDF_OCCUPANCY_FLOOR = 16384;
const LANES = 64;

/** The most elements of one dispatch: the adapter's workgroups per dimension, LANES lanes each. */
export function maxLanes(device: Pick<ShadeDevice, 'maxKernelElements'>): number {
  return Math.max(LANES, Math.floor(device.maxKernelElements / LANES) * LANES);
}

/** The chunks of one pass over `total` elements under `key`, with the TSDF's occupancy floor. */
function* chunkedPass(
  device: ShadeDevice,
  key: string,
  total: number,
  dispatch: (range: { first: number; end: number }, groups: number) => void,
) {
  const warm = { id: key, run: () => dispatch({ first: 0, end: 0 }, 0) };
  for (let first = 0; first < total;) {
    // One dispatch runs at most `maxComputeWorkgroupsPerDimension` workgroups of LANES lanes.
    const remaining = Math.min(total - first, maxLanes(device));
    const least = Math.min(remaining, device.occupancyFloors ? LANES : TSDF_OCCUPANCY_FLOOR);
    let n = Math.min(remaining, device.chunkSize(key, remaining, least, TSDF_OCCUPANCY_FLOOR));
    if (n < remaining) n = Math.max(least, n - (n % LANES));
    const range = { first, end: first + n };
    yield [
      key,
      () => dispatch(range, Math.ceil(n / LANES)),
      undefined,
      { key, units: n, minUnits: TSDF_OCCUPANCY_FLOOR, warm },
    ] as const;
    first += n;
  }
}

function nextPow2(n: number): number {
  let p = 16;
  while (p < n) p *= 2;
  return p;
}

/** The uniform of one view of a pass. */
function controlOf(
  volume: TsdfVolume,
  input: TsdfInput,
  v: number,
  mask: number,
  box: TsdfBox,
  extra: {
    minWeight: number;
    minViews: number;
    carveSpan: number;
    carveRatio: number;
    write: number;
    capacity: number;
    visibility: number;
    resetVotes: number;
  },
) {
  const view = input.views[v];
  const R = view?.rotation ?? [1, 0, 0, 0, 1, 0, 0, 0, 1],
    C = view?.center ?? [0, 0, 0];
  const k1 = view?.k1 ?? 0,
    k2 = view?.k2 ?? 0;
  return {
    width: input.width,
    height: input.height,
    r0: R[0]!,
    r1: R[1]!,
    r2: R[2]!,
    r3: R[3]!,
    r4: R[4]!,
    r5: R[5]!,
    r6: R[6]!,
    r7: R[7]!,
    r8: R[8]!,
    centerX: C[0]!,
    centerY: C[1]!,
    centerZ: C[2]!,
    fx: view?.fx ?? 1,
    fy: view?.fy ?? 1,
    cx: view?.cx ?? 0,
    cy: view?.cy ?? 0,
    distorted: k1 !== 0 || k2 !== 0 ? 1 : 0,
    k1,
    k2,
    radialLimit: Math.min(1e30, view?.radialLimit ?? 1e30),
    footprint: volume.footprints[v] ?? 0,
    originX: volume.origin[0],
    originY: volume.origin[1],
    originZ: volume.origin[2],
    voxel: volume.voxel,
    inverse: volume.inverse,
    levels: volume.levels,
    truncation: volume.truncation,
    mask,
    boxLoX: box.lo[0],
    boxLoY: box.lo[1],
    boxLoZ: box.lo[2],
    boxHiX: box.hi[0],
    boxHiY: box.hi[1],
    boxHiZ: box.hi[2],
    view: v,
    ...extra,
  };
}

/**
 * The TSDF fusion of section 10.4 on `device`: the points of the surface in the frame of the
 * input, sorted by x, y, z (`sortTsdfPoints`), and the report.
 * Input channels are borrowed read-only; keep their bytes unchanged until this promise settles.
 */
export async function fuseTsdfGpu(
  device: ShadeDevice,
  input: TsdfInput,
  options: TsdfGpuOptions = {},
): Promise<TsdfPoints & { readonly tsdf: TsdfReport }> {
  options.signal?.throwIfAborted();
  options.onProgress?.({
    phase: 'allocate',
    pass: 0,
    passes: 0,
    view: -1,
    views: input.views.length,
    done: 0,
    total: 0,
    points: 0,
  });
  const o = checkedTsdfOptions(options);
  if (
    options.maxQueuedViewBytes !== undefined &&
    (!Number.isSafeInteger(options.maxQueuedViewBytes) || options.maxQueuedViewBytes < 1)
  )
    throw new RangeError('maxQueuedViewBytes must be a positive safe integer');
  const volumeStart = performance.now();
  const volume = planTsdfVolume(input, o);
  const hostTimings = {
    volumePlanningMs: performance.now() - volumeStart,
    boxPlanningMs: 0,
    boxTableMs: 0,
    viewUploadMs: 0,
    sortingMs: 0,
  };
  const reports: ShadePlanReport[] = [];
  const empty = (): TsdfPoints & { readonly tsdf: TsdfReport } => ({
    count: 0,
    positions: new Float32Array(0),
    normals: new Uint32Array(0),
    colors: new Uint32Array(0),
    weights: new Uint32Array(0),
    free: new Uint32Array(0),
    tallies: new Uint32Array(0),
    ...(o.visibility === 'view-weighted' ? { freeWeights: new Float32Array(0) } : {}),
    tsdf: {
      voxel: volume?.voxel ?? 0,
      levels: o.levels,
      blocks: 0,
      passes: 0,
      tableSlots: 0,
      peakBytes: 0,
      uploadBytes: 0,
      hostViewStagingBytes: 0,
      allocationPixels: 0,
      reusedBoxTables: 0,
      inlineVoteResetGroups: 0,
      queuedViewUploads: {
        limitBytes: options.maxQueuedViewBytes ?? 64 * 1024 ** 2,
        peakSubmittedBytes: 0,
        drainCount: 0,
      },
      hostTimings,
      report: { tier: device.backend, dispatches: 0, submissions: 0, groups: [] },
    },
  });
  if (!volume) return empty();
  const P = input.width * input.height;
  const N = input.views.length;
  const visibilityGroups = o.visibility === 'view-weighted' ? tsdfVisibilityGroups(input) : [];
  const s = device.statistics;
  const available = Math.max(0, device.workingBytes - (s.liveBytes - s.pooledBytes));
  const budget = Math.min(options.residentBytes ?? available, available);
  const minWeight = Math.floor(o.minWeight * TSDF_WEIGHT_SCALE + 0.5);
  const extraOf = (write = 0, capacity = 0, resetVotes = 0) => ({
    minWeight,
    visibility: o.visibility === 'view-weighted' ? 1 : 0,
    minViews: o.minViews,
    carveSpan: o.carveSpan,
    carveRatio: o.carveRatio,
    write,
    capacity,
    resetVotes,
  });
  // The arrays of the fusion with their lengths: the bytes held now and at most.
  const owned = new Map<Resident<Uint32Array> | Resident<Float32Array>, number>();
  let live = 0,
    peakBytes = 0,
    uploadBytes = 0;
  const hold = <T extends Resident<Uint32Array> | Resident<Float32Array>>(r: T, n: number) => {
    owned.set(r, n);
    live += 4 * n;
    peakBytes = Math.max(peakBytes, live);
    return r;
  };
  const allocU32 = (n: number, values?: Uint32Array) => hold(device.allocateU32(n, values), n);
  const allocF32 = (n: number) => hold(device.allocate(n), n);
  const free = (r: Resident<Uint32Array> | Resident<Float32Array>) => {
    live -= 4 * (owned.get(r) ?? 0);
    owned.delete(r);
    device.recycle(r);
  };
  let pass = 0,
    passes = 0,
    emitted = 0;
  const progress = (
    phase: TsdfPhase,
    view: number,
    done: number,
    total: number,
    operation?: number,
    operations?: number,
  ) =>
    options.onProgress?.({
      phase,
      pass,
      passes,
      view,
      views: N,
      done,
      total,
      ...(operation === undefined || operations === undefined
        ? {}
        : { phaseDone: operation * total + done, phaseTotal: operations * total }),
      points: emitted,
    });
  const uploadWindow = new TsdfUploadWindow(
    Math.max(options.maxQueuedViewBytes ?? 64 * 1024 ** 2, 16 * P),
    () => device.waitForSubmitted(),
    options.signal,
  );
  const steps = async (
    list: Iterable<GpuStep>,
    phase: TsdfPhase,
    view: number,
    total: number,
    operation = 0,
    operations = 1,
  ) => {
    options.signal?.throwIfAborted();
    progress(phase, view, 0, total, operation, operations);
    // Chunk steps already run alone. Adding an after hook changes no plan or numerical readback.
    function* observed(): Generator<GpuStep> {
      let done = 0;
      for (const [key, run, , chunk] of list) {
        options.signal?.throwIfAborted();
        yield [
          key,
          run,
          async () => {
            done += chunk!.units;
            progress(phase, view, done, total, operation, operations);
            options.signal?.throwIfAborted();
          },
          chunk,
        ] as const;
      }
    }
    const report = await device.planSteps(observed(), { reportCompletion: true });
    if (report.completed) uploadWindow.completed();
    reports.push(report);
  };
  const channels = new TsdfViewChannels(P);
  let allocationPixels = 0,
    reusedBoxTables = 0,
    inlineVoteResetGroups = 0;
  let successful = false;
  try {
    const tsdfDepth = allocU32(P),
      tsdfNormal = allocU32(P),
      tsdfConfidence = allocU32(P),
      tsdfColor = allocU32(P);
    const viewBindings = { tsdfDepth, tsdfNormal, tsdfConfidence, tsdfColor };
    const tsdfGrid = allocU32(volume.grid.length, volume.grid);
    const upload = (v: number, color = true) =>
      uploadWindow.run((color ? 16 : 12) * P, () => {
        const start = performance.now();
        const source = channels.of(input.views[v]!);
        tsdfDepth.write(source.tsdfDepth);
        tsdfNormal.write(source.tsdfNormal);
        tsdfConfidence.write(source.tsdfConfidence);
        if (color) tsdfColor.write(source.tsdfColor);
        uploadBytes += (color ? 16 : 12) * P;
        hostTimings.viewUploadMs += performance.now() - start;
      });
    /** Claim the blocks of `box` from every view into `table` of `T` slots. */
    const claim = async (table: Resident<Uint32Array>, T: number, box: TsdfBox) => {
      allocationPixels += N * P;
      for (let v = 0; v < N; v++) {
        await upload(v);
        const tsdfControl = controlOf(volume, input, v, T - 1, box, extraOf());
        await steps(
          chunkedPass(device, 'tsdfAllocate', P, (tsdfRange, groups) =>
            tsdfAllocate(
              { tsdfControl, tsdfRange, ...viewBindings, tsdfGrid, tsdfTable: table },
              groups,
            ),
          ),
          'allocate',
          v,
          P,
          v,
          N,
        );
      }
      const result = await device.read(table, TSDF_HEADER + 3 * T);
      uploadWindow.completed();
      return result;
    };
    const tableOf = (T: number) => {
      const words = new Uint32Array(TSDF_HEADER + 3 * T);
      words.fill(EMPTY, TSDF_HEADER, TSDF_HEADER + T);
      return allocU32(words.length, words);
    };
    // The dry run: the blocks of every view, in a table that grows until it holds them at half load.
    let T = nextPow2(Math.max(4096, (2 * volume.samples) / 16));
    let table: Resident<Uint32Array>;
    let words: Uint32Array;
    for (;;) {
      table = tableOf(T);
      words = await claim(table, T, TSDF_WHOLE);
      if (words[2] === 0 && 2 * words[0]! <= T) break;
      free(table);
      T *= 4;
    }
    const dryBlocks = words[0]!;
    const tableSlots = T;
    const keys: number[] = [];
    for (let k = 0; k < T; k++) {
      const key = words[TSDF_HEADER + k]!;
      if (key !== EMPTY) keys.push(key);
    }
    // The passes: as many blocks as the budget holds beside the view, the grid and the points.
    const fixed = live - 4 * (TSDF_HEADER + 3 * T);
    const perBlock =
      BLOCK_BYTES +
      POINTS_PER_BLOCK * POINT_BYTES +
      (o.visibility === 'view-weighted' ? 4 * TSDF_BLOCK_VOXELS : 0);
    const fitBlocks = Math.max(
      1,
      Math.min(
        Math.floor((budget - fixed) / perBlock),
        Math.floor(device.bindingBytes / (4 * TSDF_BLOCK_WORDS)),
      ),
    );
    const maxBlocks = Math.max(1, Math.min(options.maxBlocks ?? fitBlocks, fitBlocks));
    const boxStart = performance.now();
    const boxes = planTsdfBoxes(volume, keys, maxBlocks);
    hostTimings.boxPlanningMs = performance.now() - boxStart;
    passes = boxes.length;
    const parts: { positions: Float32Array; data: Uint32Array }[] = [];
    for (const box of boxes) {
      pass++;
      let passTable: Resident<Uint32Array>, passT: number, blocks: number;
      if (boxes.length === 1) {
        passTable = table!;
        passT = T;
        blocks = dryBlocks;
      } else {
        if (owned.has(table!)) free(table!);
        // Only a successful, complete discovery reaches here. Its box-independent ray walk
        // already found every key: reuse the exact subset with the same neighbour margin.
        const tableStart = performance.now();
        const staged = tsdfBoxTable(volume, keys, box);
        passT = staged.slots;
        passTable = allocU32(staged.words.length, staged.words);
        blocks = staged.blocks;
        reusedBoxTables++;
        hostTimings.boxTableMs += performance.now() - tableStart;
      }
      if (blocks === 0) {
        free(passTable);
        continue;
      }
      // Blocks for the claimed slots, then the voxels, zeroed by the device.
      const assignControl = controlOf(volume, input, 0, passT - 1, box, extraOf());
      await steps(
        chunkedPass(device, 'tsdfAssign', passT, (tsdfRange, groups) =>
          tsdfAssign({ tsdfControl: assignControl, tsdfRange, tsdfTable: passTable }, groups),
        ),
        'assign',
        -1,
        passT,
      );
      const tsdfVoxels = allocU32(blocks * TSDF_BLOCK_WORDS);
      for (let v = 0; v < N; v++) {
        await upload(v);
        const tsdfControl = controlOf(volume, input, v, passT - 1, box, extraOf());
        const bindings = {
          tsdfControl,
          ...viewBindings,
          tsdfGrid,
          tsdfTable: passTable,
          tsdfVoxels,
        };
        await steps(
          chunkedPass(device, 'tsdfIntegrate', P, (tsdfRange, groups) =>
            tsdfIntegrate({ ...bindings, tsdfRange }, groups),
          ),
          'integrate',
          v,
          P,
          v,
          N,
        );
        if (o.visibility === 'ray-count')
          await steps(
            chunkedPass(device, 'tsdfCarve', P, (tsdfRange, groups) =>
              tsdfCarve({ ...bindings, tsdfRange }, groups),
            ),
            'carve',
            v,
            P,
            v,
            N,
          );
      }
      if (o.visibility === 'view-weighted') {
        const tsdfVotes = allocU32(blocks * TSDF_BLOCK_VOXELS);
        let operation = 0;
        const operations = N + (options.standaloneVoteReset ? visibilityGroups.length : 0);
        for (const group of visibilityGroups) {
          if (options.standaloneVoteReset)
            await steps(
              chunkedPass(
                device,
                'tsdfResetVotes',
                blocks * TSDF_BLOCK_VOXELS,
                (tsdfRange, groups) => tsdfResetVotes({ tsdfRange, tsdfVotes }, groups),
              ),
              'carve',
              -1,
              blocks * TSDF_BLOCK_VOXELS,
              operation++,
              operations,
            );
          let firstView = true;
          for (const v of group) {
            // View-weighted visibility reads depth, normal and confidence, but never color.
            await upload(v, false);
            const tsdfControl = controlOf(
              volume,
              input,
              v,
              passT - 1,
              box,
              extraOf(0, 0, firstView && !options.standaloneVoteReset ? 1 : 0),
            );
            if (firstView && !options.standaloneVoteReset) inlineVoteResetGroups++;
            firstView = false;
            await steps(
              chunkedPass(
                device,
                'tsdfViewCarve',
                blocks * TSDF_BLOCK_VOXELS,
                (tsdfRange, groups) =>
                  tsdfViewCarve(
                    {
                      tsdfControl,
                      tsdfRange,
                      tsdfDepth,
                      tsdfNormal,
                      tsdfConfidence,
                      tsdfGrid,
                      tsdfTable: passTable,
                      tsdfVoxels,
                      tsdfVotes,
                    },
                    groups,
                  ),
              ),
              'carve',
              v,
              blocks * TSDF_BLOCK_VOXELS,
              operation++,
              operations,
            );
          }
        }
        free(tsdfVotes);
      }
      // The extraction: count, then write into arrays of that many points.
      const voxels = blocks * TSDF_BLOCK_VOXELS;
      const spare = allocF32(3);
      const spareData = allocU32(TSDF_POINT_WORDS);
      const countControl = controlOf(volume, input, 0, passT - 1, box, extraOf(0, 0));
      await steps(
        chunkedPass(device, 'tsdfExtract', voxels, (tsdfRange, groups) =>
          tsdfExtract(
            {
              tsdfControl: countControl,
              tsdfRange,
              tsdfGrid,
              tsdfTable: passTable,
              tsdfVoxels,
              tsdfPoints: spare,
              tsdfPointData: spareData,
            },
            groups,
          ),
        ),
        'count',
        -1,
        voxels,
      );
      progress('readback', -1, 0, 1);
      const counted = (await device.read(passTable, TSDF_HEADER + 3 * passT))[3]!;
      uploadWindow.completed();
      free(spare);
      free(spareData);
      if (counted > 0) {
        const tsdfPoints = allocF32(3 * counted);
        const tsdfPointData = allocU32(TSDF_POINT_WORDS * counted);
        const writeControl = controlOf(volume, input, 0, passT - 1, box, extraOf(1, counted));
        await steps(
          chunkedPass(device, 'tsdfExtract', voxels, (tsdfRange, groups) =>
            tsdfExtract(
              {
                tsdfControl: writeControl,
                tsdfRange,
                tsdfGrid,
                tsdfTable: passTable,
                tsdfVoxels,
                tsdfPoints,
                tsdfPointData,
              },
              groups,
            ),
          ),
          'extract',
          -1,
          voxels,
        );
        progress('readback', -1, 0, 1);
        const [positions, data] = await readTogether([
          device.read(tsdfPoints, 3 * counted).then((a) => a.slice(0, 3 * counted)),
          device
            .read(tsdfPointData, TSDF_POINT_WORDS * counted)
            .then((a) => a.slice(0, TSDF_POINT_WORDS * counted)),
        ]);
        uploadWindow.completed();
        const part = { positions, data };
        parts.push(part);
        emitted += counted;
        progress('readback', -1, 1, 1);
        options.signal?.throwIfAborted();
        await options.onPart?.({ positions: part.positions.slice(), data: part.data.slice() });
        free(tsdfPoints);
        free(tsdfPointData);
      }
      free(tsdfVoxels);
      free(passTable);
    }
    const W = TSDF_POINT_WORDS;
    progress('sort', -1, 0, 1);
    const count = parts.reduce((n, p) => n + p.data.length / W, 0);
    const positions = new Float32Array(3 * count),
      normals = new Uint32Array(count),
      colors = new Uint32Array(count),
      weights = new Uint32Array(count),
      frees = new Uint32Array(count),
      tallies = new Uint32Array(count);
    const freeWeights = o.visibility === 'view-weighted' ? new Float32Array(count) : undefined;
    let at = 0;
    for (const p of parts) {
      positions.set(p.positions, 3 * at);
      for (let k = 0; k < p.data.length / W; k++) {
        normals[at + k] = p.data[W * k]!;
        colors[at + k] = p.data[W * k + 1]!;
        weights[at + k] = p.data[W * k + 2]!;
        const free = p.data[W * k + 3]!;
        frees[at + k] = freeWeights ? free >>> 20 : free;
        if (freeWeights) freeWeights[at + k] = (free & 0xfffff) / TSDF_CARVE_SCALE;
        tallies[at + k] = p.data[W * k + 4]!;
      }
      at += p.data.length / W;
    }
    // Only the concatenated arrays are needed now; avoid retaining extracted parts while sorting.
    parts.length = 0;
    options.signal?.throwIfAborted();
    const sortStart = performance.now();
    const sorted = sortOwnedTsdfPoints(
      positions,
      normals,
      colors,
      weights,
      frees,
      tallies,
      freeWeights,
    );
    hostTimings.sortingMs = performance.now() - sortStart;
    progress('sort', -1, 1, 1);
    successful = true;
    return {
      ...sorted,
      tsdf: {
        voxel: volume.voxel,
        levels: volume.levels,
        blocks: dryBlocks,
        passes: boxes.length,
        tableSlots,
        peakBytes,
        uploadBytes,
        hostViewStagingBytes: channels.hostBytes,
        allocationPixels,
        reusedBoxTables,
        inlineVoteResetGroups,
        queuedViewUploads: uploadWindow.report,
        hostTimings,
        report: {
          tier: device.backend,
          dispatches: reports.reduce((n, r) => n + r.dispatches, 0),
          submissions: reports.reduce((n, r) => n + r.submissions, 0),
          groups: reports.flatMap((r) => r.groups),
        },
      },
    };
  } finally {
    for (const r of owned.keys()) successful ? device.recycle(r) : device.drop(r);
  }
}
