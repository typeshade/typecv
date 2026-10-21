import type { Resident } from 'typeshade/runtime';
import type { TrackTriangulation, TriangulationObservation } from '../geometry/multiview.ts';
import { validatePose } from '../geometry/pose.ts';
import { ShadeDevice } from './device.ts';
import { finiteGpuArray, validateGpuCamera, type GpuGeometryExecution } from './pnp.ts';
import { prepareTriangulation, triangulateTracks } from './triangulation.shade.ts';

export interface GpuTriangulationOptions {
  readonly maxErrorPx?: number;
  readonly minAngleDeg?: number;
  readonly maxPairHypotheses?: number;
  readonly refinementIterations?: number;
}
export interface GpuTriangulationResult {
  readonly tracks: readonly (TrackTriangulation | undefined)[];
  /** 0 accepted, 1 fewer than two observations, 3 no validated consensus, 4 low parallax. */
  readonly status: Uint32Array;
  readonly execution: GpuGeometryExecution;
}

/** The workgroup of `triangulateTracks`: one invocation per track. The least chunk of the time
 *  budget is one workgroup of tracks. */
const TRACKS_PER_WORKGROUP = 16;

/** Batched robust DLT/point-LM triangulation. Every numerical stage executes in TypeShade. */
export function triangulateTracksGpu(
  device: ShadeDevice,
  tracks: readonly (readonly TriangulationObservation[])[],
  options: GpuTriangulationOptions = {},
): Promise<GpuTriangulationResult> {
  const maxError = options.maxErrorPx ?? 3,
    minAngle = options.minAngleDeg ?? 0.5,
    maxPairs = options.maxPairHypotheses ?? 32,
    iterations = options.refinementIterations ?? 10;
  if (
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
  const offsets = new Uint32Array(tracks.length + 1);
  const count = tracks.reduce((n, t) => n + t.length, 0),
    totalBytes = count * 212 + tracks.length * 48 + 4;
  if (
    totalBytes > device.workingBytes ||
    count * 20 * 8 > device.bindingBytes ||
    tracks.length * 40 > device.bindingBytes ||
    tracks.length > device.maxKernelElements / 4 ||
    count > device.maxKernelElements
  )
    throw new RangeError('GPU triangulation exceeds working/storage/dispatch budget');
  const raw = new Float64Array(Math.max(1, count * 20));
  let at = 0;
  for (let h = 0; h < tracks.length; h++) {
    offsets[h] = at;
    for (const o of tracks[h]!) {
      const k = validateGpuCamera({ ...o.camera });
      validatePose(o.pose);
      const record = Float64Array.of(
        ...o.pose.R,
        ...o.pose.t,
        k.fx,
        k.fy,
        k.cx,
        k.cy,
        k.k1!,
        k.k2!,
        ...o.pixel,
      );
      if (o.pixel.length !== 2 || !finiteGpuArray(record))
        throw new RangeError('Invalid GPU triangulation observation');
      raw.set(record, at * 20);
      at++;
    }
  }
  offsets[tracks.length] = at;
  return device.enqueue(async () => {
    const before = device.statistics,
      owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
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
        result = wide(tracks.length * 5),
        flags = integer(Math.max(1, count)),
        statuses = integer(tracks.length);
      const control = { maxError, minAngle, maxPairs, iterations },
        total = tracks.length;
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
            device.chunkSize('triangulateTracks', remaining, TRACKS_PER_WORKGROUP),
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
        device.read(result, tracks.length * 5),
        count ? device.read(flags, count) : new Uint32Array(0),
        device.read(statuses, tracks.length),
      ]);
      const status = statusRead.slice();
      const output = Array.from(
        { length: tracks.length },
        (_, h): TrackTriangulation | undefined => {
          if (status[h] !== 0) return undefined;
          return {
            xyz: values.slice(h * 5, h * 5 + 3),
            inliers: Uint8Array.from(masks.subarray(offsets[h]!, offsets[h + 1]!)),
            maxAngleDeg: values[h * 5 + 3]!,
            meanErrorPx: values[h * 5 + 4]!,
          };
        },
      );
      const after = device.statistics;
      successful = true;
      return {
        tracks: output,
        status,
        execution: {
          backend: device.backend,
          report,
          applicationUploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
          applicationReadbackBytes:
            after.applicationReadbackBytes - before.applicationReadbackBytes,
          cpuStages: [],
        },
      };
    } finally {
      for (const b of owned) successful ? device.recycle(b) : device.drop(b);
    }
  });
}
