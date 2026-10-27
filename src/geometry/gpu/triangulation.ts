import type { Resident } from 'typeshade/runtime';
import type { TrackTriangulation, TriangulationObservation } from '../multiview.ts';
import { depthSensitivityControls, type DepthSensitivityOptions } from '../multiview.ts';
import { validatePose } from '../pose.ts';
import { ShadeDevice, type ShadePlanReport } from '../../device/gpu/device.ts';
import { planTriangulationChunks } from '../../device/gpu/budget-plan.ts';
import { finiteGpuArray, validateGpuCamera } from './bounds.ts';
import { geometryExecution, mergedReport, type GpuGeometryExecution } from './execution.ts';
import { cameraRecord, poseRecord } from '../../contracts/camera-record.ts';
import { prepareTriangulation, triangulateTracks } from './triangulation.shade.ts';
import { TRIANGULATION_DEFAULTS } from '../operator-defaults.ts';
import { TriangulationStatus } from '../../contracts/triangulation-status.ts';

export interface GpuTriangulationOptions extends DepthSensitivityOptions {
  readonly maxErrorPx?: number;
  readonly minAngleDeg?: number;
  readonly maxPairHypotheses?: number;
  readonly refinementIterations?: number;
}
export interface GpuTriangulationResult {
  readonly tracks: readonly (TrackTriangulation | undefined)[];
  /** 0 accepted, 1 insufficient observations, 3 no consensus, 4 low parallax, 5 depth sensitivity. */
  readonly status: Uint32Array;
  readonly execution: GpuGeometryExecution;
}

/** The workgroup of `triangulateTracks`: one invocation per track. The least chunk of the time
 *  budget is one workgroup of tracks. */
const TRACKS_PER_WORKGROUP = 16;
/**
 * The tracks a chunk holds at least, to fill the device: 1024 workgroups of 16 lanes. One track is
 * a serial solve in one lane, so a chunk of 118 tracks and one of 16384 cost about the same until
 * the device is full (`docs/gpu-throughput.md`, item 2).
 */
const TRIANGULATION_FLOOR = TRACKS_PER_WORKGROUP * 1024;

/** Batched robust DLT/point-LM triangulation. Every numerical stage executes in TypeShade. */
export function triangulateTracksGpu(
  device: ShadeDevice,
  tracks: readonly (readonly TriangulationObservation[])[],
  options: GpuTriangulationOptions = {},
): Promise<GpuTriangulationResult> {
  const depth = depthSensitivityControls(options);
  const maxError = options.maxErrorPx ?? TRIANGULATION_DEFAULTS.maxErrorPx,
    minAngle = options.minAngleDeg ?? TRIANGULATION_DEFAULTS.minAngleDeg,
    maxPairs = options.maxPairHypotheses ?? TRIANGULATION_DEFAULTS.maxPairHypotheses,
    iterations = options.refinementIterations ?? 10;
  if (
    !finiteGpuArray(Float64Array.of(depth.maxRelativeDepthStdDev, depth.pixelNoiseStdDevPx)) ||
    !tracks.length ||
    tracks.some((t) => t.length > 64) ||
    !Number.isFinite(maxError) ||
    maxError < 1e-12 ||
    maxError > 1e6 ||
    !Number.isFinite(minAngle) ||
    minAngle < 0 ||
    minAngle >= 180 ||
    !Number.isSafeInteger(maxPairs) ||
    maxPairs < 1 ||
    maxPairs > 64 ||
    !Number.isSafeInteger(iterations) ||
    iterations < 0 ||
    iterations > 100
  )
    throw new RangeError('Invalid GPU triangulation batch/options (at most 64 observations/track)');
  // docs/gpu-budget.md, section 4: the tracks are cut into chunks that fit the budget. A track
  // reads only its own observations, so a chunk gives the results of one dispatch over all tracks.
  // The observations are validated and packed on the host first, so a bad one throws before any
  // GPU work.
  const offsets = new Uint32Array(tracks.length + 1);
  const count = tracks.reduce((n, t) => n + t.length, 0);
  const raw = new Float64Array(Math.max(1, count * 20));
  let at = 0;
  for (let h = 0; h < tracks.length; h++) {
    offsets[h] = at;
    for (const o of tracks[h]!) {
      const k = validateGpuCamera({ ...o.camera });
      validatePose(o.pose);
      const record = Float64Array.of(...poseRecord(o.pose), ...cameraRecord(k), ...o.pixel);
      if (o.pixel.length !== 2 || !finiteGpuArray(record))
        throw new RangeError('Invalid GPU triangulation observation');
      raw.set(record, at * 20);
      at++;
    }
  }
  offsets[tracks.length] = at;
  return device.enqueue(async () => {
    const before = device.statistics,
      chunks = planTriangulationChunks(
        tracks.map((t) => t.length),
        {
          budgetBytes: Math.max(0, device.workingBytes - (before.liveBytes - before.pooledBytes)),
          bindingBytes: device.bindingBytes,
          maxKernelElements: device.maxKernelElements,
        },
      );
    const output: (TrackTriangulation | undefined)[] = [],
      statuses = new Uint32Array(tracks.length),
      reports: ShadePlanReport[] = [];
    for (const chunk of chunks) {
      const base = offsets[chunk.first]!,
        local = new Uint32Array(chunk.end - chunk.first + 1);
      for (let h = chunk.first; h <= chunk.end; h++) local[h - chunk.first] = offsets[h]! - base;
      const part = await triangulateChunk(
        device,
        raw.slice(base * 20, Math.max(base + 1, offsets[chunk.end]!) * 20),
        local,
        { maxError, minAngle, maxPairs, iterations, ...depth },
      );
      output.push(...part.tracks);
      statuses.set(part.status, chunk.first);
      reports.push(part.report);
    }
    const after = device.statistics;
    const report: ShadePlanReport =
      reports.length === 1 ? reports[0]! : mergedReport(device.backend, reports);
    return {
      tracks: output,
      status: statuses,
      execution: geometryExecution(device, report, before, after),
    };
  });
}

/** One chunk of tracks: its arrays are allocated, run and released here (not queued). */
async function triangulateChunk(
  device: ShadeDevice,
  raw: Float64Array,
  offsets: Uint32Array,
  options: {
    maxError: number;
    minAngle: number;
    maxPairs: number;
    iterations: number;
    maxRelativeDepthStdDev: number;
    pixelNoiseStdDevPx: number;
  },
): Promise<{
  tracks: (TrackTriangulation | undefined)[];
  status: Uint32Array;
  report: ShadePlanReport;
}> {
  const trackCount = offsets.length - 1,
    count = offsets[trackCount]!;
  {
    const owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
    let successful = false;
    const wide = (n: number, values?: Float64Array) => {
      const r = device.allocateF64(n, values);
      owned.push(r);
      return r;
    };
    const integer = (n: number, values?: Uint32Array) => {
      const r = device.allocateU32(n, values);
      owned.push(r);
      return r;
    };
    try {
      const input = wide(raw.length, raw),
        index = integer(offsets.length, offsets),
        geometry = wide(Math.max(1, count * 6)),
        result = wide(trackCount * 5),
        flags = integer(Math.max(1, count)),
        statuses = integer(trackCount);
      const control = { ...options },
        total = trackCount;
      // Named steps, so the device reports each kernel before it starts (a lost device names it).
      // `triangulateTracks` runs under the time budget (`docs/gpu-sfm.md`, "Triangulation"): a
      // chunk is a contiguous range of tracks, a plan of its own that the TypeShade device fences
      // and times (key `triangulateTracks`, one unit a track). The generator sizes each chunk
      // after the previous one ran, from the rate it measured. A track's work reads the prepared
      // geometry of its own observations and writes only its own slots, so the chunks give the
      // results of one dispatch over all tracks, bit for bit. The CPU tier runs one chunk.
      function* steps() {
        yield [
          'prepareTriangulation',
          () => {
            if (count) prepareTriangulation({ raw: input, geometry }, Math.ceil(count / 64));
          },
        ] as const;
        for (let first = 0; first < total;) {
          const remaining = total - first;
          let n = Math.min(
            remaining,
            device.chunkSize(
              'triangulateTracks',
              remaining,
              TRACKS_PER_WORKGROUP,
              TRIANGULATION_FLOOR,
            ),
          );
          // A chunk before the last fills its workgroups.
          if (n < remaining) n = Math.max(TRACKS_PER_WORKGROUP, n - (n % TRACKS_PER_WORKGROUP));
          const range = { first, end: first + n };
          yield [
            'triangulateTracks',
            () =>
              triangulateTracks(
                { control, range, raw: input, offsets: index, geometry, result, flags, statuses },
                Math.ceil(n / TRACKS_PER_WORKGROUP),
              ),
            undefined,
            {
              key: 'triangulateTracks',
              units: n,
              minUnits: TRIANGULATION_FLOOR,
              // Zero workgroups: the pipeline is made before the first timed chunk.
              warm: {
                id: 'triangulation',
                run: () =>
                  triangulateTracks(
                    {
                      control,
                      range: { first: 0, end: 0 },
                      raw: input,
                      offsets: index,
                      geometry,
                      result,
                      flags,
                      statuses,
                    },
                    0,
                  ),
              },
            },
          ] as const;
          first += n;
        }
      }
      const report = await device.planSteps(steps());
      // The reads share one round trip: the plan is complete before the first of them returns.
      const [values, masks, statusRead] = await Promise.all([
        device.read(result, trackCount * 5),
        count ? device.read(flags, count) : new Uint32Array(0),
        device.read(statuses, trackCount),
      ]);
      const status = statusRead.slice();
      const output = Array.from({ length: trackCount }, (_, h): TrackTriangulation | undefined => {
        if (status[h] !== TriangulationStatus.accepted) return undefined;
        return {
          xyz: values.slice(h * 5, h * 5 + 3),
          inliers: Uint8Array.from(masks.subarray(offsets[h]!, offsets[h + 1]!)),
          maxAngleDeg: values[h * 5 + 3]!,
          meanErrorPx: values[h * 5 + 4]!,
        };
      });
      successful = true;
      return { tracks: output, status, report };
    } finally {
      for (const b of owned) successful ? device.recycle(b) : device.drop(b);
    }
  }
}
