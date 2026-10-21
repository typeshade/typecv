/**
 * The TSDF fusion on the device (`docs/dense-mvs.md`, section 10.4): the kernels of
 * `tsdf.shade.ts` over the filtered depth maps, within the device's working budget. The host plans
 * the volume (`planTsdfVolume`), counts the blocks with an allocation-only dry run, and cuts the
 * volume into boxes that fit the budget (`planTsdfBoxes`). Each box runs the claim, the
 * integration and the free-space counts of every view, then the extraction, and its points go to
 * the host before the next box. A box extracts only its own blocks, so no point repeats, and its
 * margin holds the neighbours its voxels read. The budget changes the speed and not the points.
 */
import type { Resident } from 'typeshade/runtime';
import type { ShadeDevice, ShadePlanReport } from './device.ts';
import { tsdfAllocate, tsdfAssign, tsdfCarve, tsdfExtract, tsdfIntegrate } from './tsdf.shade.ts';
import {
  TSDF_BLOCK_VOXELS,
  TSDF_BLOCK_WORDS,
  TSDF_HEADER,
  TSDF_WEIGHT_SCALE,
  TSDF_WHOLE,
  checkedTsdfOptions,
  planTsdfBoxes,
  planTsdfVolume,
  sortTsdfPoints,
  tsdfInBox,
  tsdfKeyBlock,
  type TsdfBox,
  type TsdfInput,
  type TsdfOptions,
  type TsdfPoints,
  type TsdfVolume,
} from '../reconstruction/tsdf.ts';

export interface TsdfGpuOptions extends TsdfOptions {
  /** The most bytes the fusion keeps on the device. Default: the device's free working bytes. */
  readonly residentBytes?: number;
  /** @internal Cut the volume into passes of at most this many blocks (a test of the boxes). */
  readonly maxBlocks?: number;
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
  /** The bytes uploaded to the device: the views, once per claim and once per integration. */
  readonly uploadBytes: number;
  readonly report: ShadePlanReport;
}

const EMPTY = 0xffffffff;
/** Bytes of one block: its voxels, and two slots of the table at three words each. */
const BLOCK_BYTES = 4 * TSDF_BLOCK_WORDS + 2 * 3 * 4;
/** The points one block is expected to give (the extraction falls back to smaller ranges). */
const POINTS_PER_BLOCK = 96;
const POINT_BYTES = 20;

/**
 * The least elements of a chunk: 16,384 pixels, slots or voxels (256 workgroups). The device sizes
 * a chunk from the time its key's chunks took, and a fenced chunk on a loaded host can take most of
 * its time in the fence: a rate learned from small chunks then holds every chunk small, and the
 * dispatches, not the kernels, bound the fusion (the Desert arch record of 10.8). One element of a
 * TSDF kernel is a walk of at most a few hundred cells, so this floor keeps a chunk far below a
 * GPU watchdog's limit.
 */
export const TSDF_MIN_CHUNK = 16384;
const LANES = 64;

/** The chunks of one pass over `total` elements under `key`, at least `TSDF_MIN_CHUNK` each. */
function* chunkedPass(
  device: ShadeDevice,
  key: string,
  total: number,
  dispatch: (range: { first: number; end: number }, groups: number) => void,
) {
  const warm = { id: key, run: () => dispatch({ first: 0, end: 0 }, 0) };
  for (let first = 0; first < total;) {
    const remaining = total - first;
    const least = Math.min(remaining, TSDF_MIN_CHUNK);
    let n = Math.min(remaining, device.chunkSize(key, remaining, least));
    if (n < remaining) n = Math.max(least, n - (n % LANES));
    const range = { first, end: first + n };
    yield [
      key,
      () => dispatch(range, Math.ceil(n / LANES)),
      undefined,
      { key, units: n, warm },
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
  extra: { minWeight: number; carveRatio: number; write: number; capacity: number },
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
    ...extra,
  };
}

/** One view as `tsdfView` holds it: depth bits, normal, confidence bits, RGBA per pixel. */
function packView(input: TsdfInput, v: number): Uint32Array {
  const view = input.views[v]!;
  const P = input.width * input.height;
  const words = new Uint32Array(4 * P);
  const floats = new Float32Array(words.buffer);
  for (let i = 0; i < P; i++) {
    floats[4 * i] = view.depth[i]!;
    words[4 * i + 1] = view.normal[i]!;
    floats[4 * i + 2] = view.confidence ? view.confidence[i]! : 1;
    words[4 * i + 3] = view.rgba[i]!;
  }
  return words;
}

/**
 * The TSDF fusion of section 10.4 on `device`: the points of the surface in the frame of the
 * input, sorted by x, y, z (`sortTsdfPoints`), and the report.
 */
export async function fuseTsdfGpu(
  device: ShadeDevice,
  input: TsdfInput,
  options: TsdfGpuOptions = {},
): Promise<TsdfPoints & { readonly tsdf: TsdfReport }> {
  const o = checkedTsdfOptions(options);
  const volume = planTsdfVolume(input, o);
  const reports: ShadePlanReport[] = [];
  const empty = (): TsdfPoints & { readonly tsdf: TsdfReport } => ({
    count: 0,
    positions: new Float32Array(0),
    normals: new Uint32Array(0),
    colors: new Uint32Array(0),
    tsdf: {
      voxel: volume?.voxel ?? 0,
      levels: o.levels,
      blocks: 0,
      passes: 0,
      tableSlots: 0,
      peakBytes: 0,
      uploadBytes: 0,
      report: { tier: device.backend, dispatches: 0, submissions: 0, groups: [] },
    },
  });
  if (!volume) return empty();
  const P = input.width * input.height;
  const N = input.views.length;
  const s = device.statistics;
  const available = Math.max(0, device.workingBytes - (s.liveBytes - s.pooledBytes));
  const budget = Math.min(options.residentBytes ?? available, available);
  const viewBytes = 16 * P;
  const minWeight = Math.floor(o.minWeight * TSDF_WEIGHT_SCALE + 0.5);
  const extraOf = (write = 0, capacity = 0) => ({
    minWeight,
    carveRatio: o.carveRatio,
    write,
    capacity,
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
  const steps = async (list: Iterable<readonly [string, () => void, ...unknown[]]>) => {
    reports.push(await device.planSteps(list as never));
  };
  const packed: (Uint32Array | undefined)[] = new Array(N);
  const viewWords = (v: number) => (packed[v] ??= packView(input, v));
  let successful = false;
  try {
    const tsdfView = allocU32(4 * P);
    const tsdfGrid = allocU32(volume.grid.length, volume.grid);
    const upload = (v: number) => {
      tsdfView.write(viewWords(v));
      uploadBytes += viewBytes;
    };
    /** Claim the blocks of `box` from every view into `table` of `T` slots. */
    const claim = async (table: Resident<Uint32Array>, T: number, box: TsdfBox) => {
      for (let v = 0; v < N; v++) {
        upload(v);
        const tsdfControl = controlOf(volume, input, v, T - 1, box, extraOf());
        await steps(
          chunkedPass(device, 'tsdfAllocate', P, (tsdfRange, groups) =>
            tsdfAllocate({ tsdfControl, tsdfRange, tsdfView, tsdfGrid, tsdfTable: table }, groups),
          ),
        );
      }
      return device.read(table, TSDF_HEADER + 3 * T);
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
    const perBlock = BLOCK_BYTES + POINTS_PER_BLOCK * POINT_BYTES;
    const fitBlocks = Math.max(
      1,
      Math.min(
        Math.floor((budget - fixed) / perBlock),
        Math.floor(device.bindingBytes / (4 * TSDF_BLOCK_WORDS)),
      ),
    );
    const maxBlocks = Math.max(1, Math.min(options.maxBlocks ?? fitBlocks, fitBlocks));
    const boxes = planTsdfBoxes(volume, keys, maxBlocks);
    const parts: { positions: Float32Array; data: Uint32Array }[] = [];
    for (const box of boxes) {
      let passTable: Resident<Uint32Array>, passT: number, blocks: number;
      if (boxes.length === 1) {
        passTable = table!;
        passT = T;
        blocks = dryBlocks;
      } else {
        if (owned.has(table!)) free(table!);
        // A table of at least twice the slots of the box's blocks, from the dry run's keys.
        const count = keys.filter((key) => {
          const [level, ...b] = tsdfKeyBlock(volume, key);
          return tsdfInBox(volume, box, level, b, true);
        }).length;
        passT = nextPow2(Math.max(64, 2 * count));
        passTable = tableOf(passT);
        const claimed = await claim(passTable, passT, box);
        blocks = claimed[0]!;
        if (claimed[2] !== 0) throw new Error('The TSDF table overflowed');
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
      );
      const tsdfVoxels = allocU32(blocks * TSDF_BLOCK_WORDS);
      for (let v = 0; v < N; v++) {
        upload(v);
        const tsdfControl = controlOf(volume, input, v, passT - 1, box, extraOf());
        const bindings = { tsdfControl, tsdfView, tsdfGrid, tsdfTable: passTable, tsdfVoxels };
        await steps([
          ...chunkedPass(device, 'tsdfIntegrate', P, (tsdfRange, groups) =>
            tsdfIntegrate({ ...bindings, tsdfRange }, groups),
          ),
          ...chunkedPass(device, 'tsdfCarve', P, (tsdfRange, groups) =>
            tsdfCarve({ ...bindings, tsdfRange }, groups),
          ),
        ]);
      }
      // The extraction: count, then write into arrays of that many points.
      const voxels = blocks * TSDF_BLOCK_VOXELS;
      const spare = allocF32(3);
      const spareData = allocU32(2);
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
      );
      const counted = (await device.read(passTable, TSDF_HEADER + 3 * passT))[3]!;
      free(spare);
      free(spareData);
      if (counted > 0) {
        const tsdfPoints = allocF32(3 * counted);
        const tsdfPointData = allocU32(2 * counted);
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
        );
        parts.push({
          positions: (await device.read(tsdfPoints, 3 * counted)).slice(0, 3 * counted),
          data: (await device.read(tsdfPointData, 2 * counted)).slice(0, 2 * counted),
        });
        free(tsdfPoints);
        free(tsdfPointData);
      }
      free(tsdfVoxels);
      free(passTable);
    }
    const count = parts.reduce((n, p) => n + p.data.length / 2, 0);
    const positions = new Float32Array(3 * count),
      normals = new Uint32Array(count),
      colors = new Uint32Array(count);
    let at = 0;
    for (const p of parts) {
      positions.set(p.positions, 3 * at);
      for (let k = 0; k < p.data.length / 2; k++) {
        normals[at + k] = p.data[2 * k]!;
        colors[at + k] = p.data[2 * k + 1]!;
      }
      at += p.data.length / 2;
    }
    successful = true;
    return {
      ...sortTsdfPoints(positions, normals, colors),
      tsdf: {
        voxel: volume.voxel,
        levels: volume.levels,
        blocks: dryBlocks,
        passes: boxes.length,
        tableSlots,
        peakBytes,
        uploadBytes,
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
