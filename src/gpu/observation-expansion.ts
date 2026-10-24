import { cameraProjection, unprojectPixel, validateCamera } from '../geometry/camera.ts';
import { projectPoint, validatePose } from '../geometry/pose.ts';
import { buildTracks, validateFeatures } from '../reconstruction/tracks.ts';
import type { PairRotationEvidence } from '../reconstruction/tracking.ts';
import { multiply, transpose } from '../sfm/linear.ts';
import type { ImageFeatures, PairMatches, SparseReconstruction } from '../reconstruction/types.ts';
import type { ShadeDevice } from './device.ts';
import { trackKltGpu } from './klt-device.ts';
import { createKltPairCache } from './klt-pair-cache.ts';
import { KltPointGrid, kltAlive, kltOptionsFor, kltAffineRecords, type KltOptions } from './klt.ts';

/** Frames in playback order, in the same pixel coordinates as their keypoints and camera. */
export interface ObservationExpansionFrame extends ImageFeatures {
  readonly width: number;
  readonly height: number;
  /** RGB-derived grayscale in [0, 1]. Retained original image evidence, not rendered predictions. */
  readonly gray: Float32Array;
  /** Optional caller-supplied valid-region/motion mask: zero excludes a pixel. */
  readonly validMask?: Uint8Array;
  /** Known shot boundary information; this adapter does not detect cuts. */
  readonly shotId?: number | string;
}
export interface ObservationExpansionOptions {
  /** Reuse at most three resident pyramids during this batch. Opt-in; disposal is automatic. */
  readonly pyramidCache?: boolean;
  /** Local patch Jacobian from estimated rotation and K. Experimental, default false. */
  readonly geometryPatchWarp?: boolean;
  /** Search this many neighbours on each side of every seed frame (default 1). */
  readonly radius?: number;
  readonly maxSeedsPerFrame?: number;
  readonly maxNewPerFrame?: number;
  readonly seedSpacingPx?: number;
  readonly joinRadiusPx?: number;
  readonly klt?: Partial<KltOptions>;
  readonly chunkTracks?: number;
  /** Only verified edges may establish ownership. Raw descriptor candidates belong in `pairs`. */
  readonly verifiedPairs?: readonly PairMatches[];
  /** Estimated geometry in ONE connected map, used only to initialize searches. Never source truth. */
  readonly initialModel?: SparseReconstruction;
  /** Calibrated relative hypotheses can guide searches before any finite map is initialized. */
  readonly rotations?: readonly PairRotationEvidence[];
  /** Reject a second local NCC peak this close to the winning score (default .03). */
  readonly uniquenessMargin?: number;
  /** Integer pixel radius for the local second-peak test (default 6). */
  readonly ambiguityRadiusPx?: number;
}
export interface ExpandedObservation {
  readonly viewId: number;
  readonly featureId: number;
  readonly sourceViewId: number;
  readonly sourceFeatureId: number;
  readonly measuredPixel: readonly [number, number];
  readonly ncc: number;
  readonly forwardBackwardErrorPx: number;
  readonly uniquenessMargin: number;
  readonly prediction: 'none' | 'rotation' | 'finite';
  /** Actual local source-to-target patch matrix when affine matching was requested. */
  readonly patchWarp?: readonly number[];
  /** Photometrically measured; the mapper must still verify geometry and depth. */
  readonly geometry: 'candidate';
}
export interface ObservationExpansionResult {
  /** Original feature indices are an unchanged prefix. No descriptor is fabricated for new rows. */
  readonly images: ImageFeatures[];
  /** Includes the supplied candidates. Pass through mapper verification before building tracks. */
  readonly candidatePairs: PairMatches[];
  readonly newObservations: ExpandedObservation[];
  readonly diagnostics: {
    readonly attempts: number;
    readonly measuredEdges: number;
    readonly addedEdges: number;
    readonly newCoordinates: number;
    readonly rejected: Readonly<Record<string, number>>;
    readonly elapsedMs: number;
    readonly trackingMs: number;
    readonly hostValidationMs: number;
    readonly uploadBytes: number;
    readonly readbackBytes: number;
    /** Complete Resident.read snapshots, including affine tails; not physical bus traffic. */
    readonly residentSnapshotBytes: number;
    readonly reads: number;
    readonly cache?: ReturnType<ReturnType<typeof createKltPairCache>['statistics']>;
  };
}

function sample(gray: Float32Array, width: number, x: number, y: number): number {
  const ix = Math.floor(x),
    iy = Math.floor(y),
    a = x - ix,
    b = y - iy;
  return (
    (1 - b) * ((1 - a) * gray[iy * width + ix]! + a * gray[iy * width + ix + 1]!) +
    b * ((1 - a) * gray[(iy + 1) * width + ix]! + a * gray[(iy + 1) * width + ix + 1]!)
  );
}
function inside(
  frame: ObservationExpansionFrame,
  x: number,
  y: number,
  hw: number,
  hh = hw,
): boolean {
  if (!(x >= hw && y >= hh && x + hw < frame.width - 1 && y + hh < frame.height - 1)) return false;
  if (frame.validMask)
    for (let j = Math.floor(y - hh); j <= Math.ceil(y + hh); j++)
      for (let i = Math.floor(x - hw); i <= Math.ceil(x + hw); i++)
        if (!frame.validMask[j * frame.width + i]) return false;
  return true;
}
/** Host quality gate on retained pixels. NCC cannot resolve arbitrary repeated assets. */
function patchNcc(
  a: ObservationExpansionFrame,
  b: ObservationExpansionFrame,
  sx: number,
  sy: number,
  x: number,
  y: number,
  hw: number,
  affine?: ArrayLike<number>,
  offset = 0,
): number {
  const m = affine && Array.from({ length: 4 }, (_, k) => affine[offset + k]!);
  if (
    !inside(a, sx, sy, hw) ||
    !inside(
      b,
      x,
      y,
      m ? hw * (Math.abs(m[0]!) + Math.abs(m[1]!)) : hw,
      m ? hw * (Math.abs(m[2]!) + Math.abs(m[3]!)) : hw,
    )
  )
    return -Infinity;
  let si = 0,
    sj = 0,
    sii = 0,
    sjj = 0,
    sij = 0;
  const n = (2 * hw + 1) ** 2;
  for (let j = -hw; j <= hw; j++)
    for (let i = -hw; i <= hw; i++) {
      const p = sample(a.gray, a.width, sx + i, sy + j),
        q = m
          ? sample(b.gray, b.width, x + m[0]! * i + m[1]! * j, y + m[2]! * i + m[3]! * j)
          : sample(b.gray, b.width, x + i, y + j);
      si += p;
      sj += q;
      sii += p * p;
      sjj += q * q;
      sij += p * q;
    }
  const variance = (sii - (si * si) / n) * (sjj - (sj * sj) / n);
  return variance > 1e-20 ? (sij - (si * sj) / n) / Math.sqrt(variance) : -Infinity;
}

/**
 * Measure missing coordinates from multi-frame seeds. The device runs the existing KLT kernels;
 * host code manages ownership, local ambiguity checks and append-only feature indices. This is
 * a bounded batch refinement, not a streaming session mutation or an inference of visibility.
 */
export async function expandObservationsGpu(
  device: ShadeDevice,
  frames: readonly ObservationExpansionFrame[],
  pairs: readonly PairMatches[],
  options: ObservationExpansionOptions = {},
): Promise<ObservationExpansionResult> {
  const started = performance.now(),
    before = device.statistics;
  validateFeatures(frames, pairs);
  const verified = options.verifiedPairs ?? [];
  validateFeatures(frames, verified);
  const radius = options.radius ?? 1,
    maxSeeds = options.maxSeedsPerFrame ?? 256,
    maxNew = options.maxNewPerFrame ?? 1024,
    spacing = options.seedSpacingPx ?? 8,
    joinRadius = options.joinRadiusPx ?? 0.75,
    uniqueness = options.uniquenessMargin ?? 0.03,
    ambiguityRadius = options.ambiguityRadiusPx ?? 6;
  if (
    ![radius, maxSeeds, maxNew].every((v) => Number.isSafeInteger(v) && v >= 1) ||
    maxSeeds > 8192 ||
    ![spacing, joinRadius].every((v) => Number.isFinite(v) && v > 0) ||
    !Number.isFinite(uniqueness) ||
    uniqueness < 0 ||
    uniqueness > 1 ||
    !Number.isSafeInteger(ambiguityRadius) ||
    ambiguityRadius < 4 ||
    ambiguityRadius > 32
  )
    throw new RangeError('Invalid observation expansion bounds');
  const byId = new Map(frames.map((frame) => [frame.id, frame]));
  const rotationGuides = new Map<
    string,
    { R: Float64Array; first: ImageFeatures['camera']; second: ImageFeatures['camera'] }
  >();
  for (const guide of options.rotations ?? []) {
    if (!byId.has(guide.first) || !byId.has(guide.second) || guide.first === guide.second)
      throw new RangeError('Rotation guide is outside expansion frames');
    validatePose({ R: guide.rotation.R, t: new Float64Array(3) });
    validateCamera(guide.firstCamera);
    validateCamera(guide.secondCamera);
    rotationGuides.set(`${guide.first}:${guide.second}`, {
      R: guide.rotation.R,
      first: guide.firstCamera,
      second: guide.secondCamera,
    });
    rotationGuides.set(`${guide.second}:${guide.first}`, {
      R: transpose(guide.rotation.R, 3),
      first: guide.secondCamera,
      second: guide.firstCamera,
    });
  }
  for (const frame of frames) {
    if (
      !Number.isSafeInteger(frame.width) ||
      !Number.isSafeInteger(frame.height) ||
      frame.width < 2 ||
      frame.height < 2 ||
      frame.gray.length !== frame.width * frame.height ||
      (frame.validMask && frame.validMask.length !== frame.gray.length) ||
      !frame.gray.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)
    )
      throw new RangeError('Frames require finite grayscale in [0, 1] and valid dimensions');
    const klt = kltOptionsFor(frame.width, options.klt);
    if (
      !Number.isInteger(klt.halfWindow) ||
      klt.halfWindow < 1 ||
      klt.halfWindow > 31 ||
      !Number.isInteger(klt.maxIterations) ||
      klt.maxIterations < 1 ||
      klt.maxIterations > 100 ||
      !Number.isInteger(klt.levels) ||
      klt.levels < 1 ||
      klt.levels > 8 ||
      ![klt.epsilonPx, klt.minEigThreshold, klt.fbThresholdPx].every(
        (v) => Number.isFinite(v) && v > 0,
      ) ||
      !Number.isFinite(klt.nccThreshold) ||
      klt.nccThreshold < 0 ||
      klt.nccThreshold > 1 ||
      ![0, 1, 2].includes(klt.photometric)
    )
      throw new RangeError('Invalid observation expansion KLT options');
  }
  const modelViews = new Map(options.initialModel?.views.map((v) => [v.id, v]));
  if (modelViews.size !== (options.initialModel?.views.length ?? 0))
    throw new RangeError('Initial model contains duplicate views');
  for (const view of modelViews.values()) {
    if (!byId.has(view.id)) throw new RangeError('Initial model view is outside expansion frames');
    validatePose(view.pose);
    validateCamera(view.camera);
  }
  const points = new Map<string, Float64Array>(),
    key = (v: number, f: number) => `${v}:${f}`;
  const modelEdges: PairMatches[] = [];
  for (const point of options.initialModel?.landmarks ?? []) {
    const first = point.observations[0];
    for (const observation of point.observations) {
      const frame = byId.get(observation.viewId);
      if (
        !frame ||
        !Number.isSafeInteger(observation.featureId) ||
        observation.featureId < 0 ||
        observation.featureId >= frame.keypoints.length / 2 ||
        point.xyz.length !== 3 ||
        !point.xyz.every(Number.isFinite)
      )
        throw new RangeError('Invalid initial model observation');
      if (points.has(key(observation.viewId, observation.featureId)))
        throw new RangeError('Initial model assigns a feature to multiple landmarks');
      points.set(key(observation.viewId, observation.featureId), point.xyz);
      if (first && first.viewId !== observation.viewId)
        modelEdges.push({
          first: first.viewId,
          second: observation.viewId,
          matches: Uint32Array.of(first.featureId, observation.featureId),
        });
    }
  }
  const coordinates = new Map(frames.map((frame) => [frame.id, Array.from(frame.keypoints)]));
  const owner = new Map<string, number>(),
    members: Map<number, number>[] = [];
  const claim = (v: number, f: number, group: number) => {
    owner.set(key(v, f), group);
    members[group]!.set(v, f);
  };
  for (const track of buildTracks(frames, [...verified, ...modelEdges]).tracks) {
    const group = members.push(new Map()) - 1;
    for (const o of track.observations) claim(o.viewId, o.featureId, group);
  }
  // Existing candidates retain priority within each image pair. Their geometry is not assumed.
  const output = new Map<
    string,
    {
      first: number;
      second: number;
      matches: number[];
      distances: number[];
      a: Map<number, number>;
      b: Map<number, number>;
    }
  >();
  const pairEntry = (first: number, second: number) => {
    const id = `${first}:${second}`;
    let entry = output.get(id);
    if (!entry) {
      entry = { first, second, matches: [], distances: [], a: new Map(), b: new Map() };
      output.set(id, entry);
    }
    return entry;
  };
  for (const pair of pairs)
    for (let k = 0; k < pair.matches.length; k += 2) {
      const reverse = pair.first > pair.second,
        entry = pairEntry(Math.min(pair.first, pair.second), Math.max(pair.first, pair.second)),
        a = pair.matches[k + (reverse ? 1 : 0)]!,
        b = pair.matches[k + (reverse ? 0 : 1)]!;
      entry.matches.push(a, b);
      entry.distances.push(pair.distances?.[k / 2] ?? 0);
      entry.a.set(a, b);
      entry.b.set(b, a);
    }
  const newObservations: ExpandedObservation[] = [],
    rejected: Record<string, number> = {};
  const reject = (reason: string) => {
    rejected[reason] = (rejected[reason] ?? 0) + 1;
  };
  let attempts = 0,
    measuredEdges = 0,
    addedEdges = 0,
    trackingMs = 0,
    hostValidationMs = 0;
  const cache =
    options.pyramidCache === true ? createKltPairCache(device, { capacity: 3 }) : undefined;
  try {
    for (let sourceIndex = 0; sourceIndex < frames.length; sourceIndex++) {
      const source = frames[sourceIndex]!,
        rankedSeeds: number[] = [],
        sourceOptions = kltOptionsFor(source.width, options.klt);
      // Round-robin cells prevents the detector's response ordering from filling one image corner.
      const cells = new Map<string, number[]>();
      for (let f = 0; f < source.keypoints.length / 2; f++) {
        const x = source.keypoints[2 * f]!,
          y = source.keypoints[2 * f + 1]!,
          cell = `${Math.floor(x / 32)}:${Math.floor(y / 32)}`;
        if (!inside(source, x, y, sourceOptions.halfWindow)) continue;
        if (!cells.has(cell)) cells.set(cell, []);
        cells.get(cell)!.push(f);
      }
      for (let rank = 0; ; rank++) {
        let any = false;
        for (const candidates of cells.values()) {
          const f = candidates[rank];
          if (f === undefined) continue;
          any = true;
          rankedSeeds.push(f);
        }
        if (!any) break;
      }
      for (
        let targetIndex = Math.max(0, sourceIndex - radius);
        targetIndex <= Math.min(frames.length - 1, sourceIndex + radius);
        targetIndex++
      ) {
        if (targetIndex === sourceIndex || !rankedSeeds.length) continue;
        const target = frames[targetIndex]!;
        // Do not jump across a known boundary even if the endpoint labels happen to agree.
        if (
          frames
            .slice(Math.min(sourceIndex, targetIndex), Math.max(sourceIndex, targetIndex) + 1)
            .some((f, i, a) => i > 0 && f.shotId !== a[i - 1]!.shotId)
        ) {
          reject('shotBoundary');
          continue;
        }
        if (source.width !== target.width || source.height !== target.height) {
          reject('imageCoordinates');
          continue;
        }
        const seeds: number[] = [],
          occupied = new KltPointGrid(spacing);
        for (const f of rankedSeeds) {
          // Spend the bounded search budget on missing observations, including wholly new tracks.
          const group = owner.get(key(source.id, f));
          if (group !== undefined && members[group]!.has(target.id)) continue;
          const x = source.keypoints[2 * f]!,
            y = source.keypoints[2 * f + 1]!;
          if (!occupied.near(x, y, spacing)) {
            seeds.push(f);
            occupied.add(x, y);
          }
          if (seeds.length >= maxSeeds) break;
        }
        if (!seeds.length) continue;
        const klt = kltOptionsFor(source.width, options.klt),
          xy = new Float32Array(2 * seeds.length),
          guesses = new Float32Array(2 * seeds.length),
          affineWarps = new Float32Array(4 * seeds.length),
          predictions: ExpandedObservation['prediction'][] = [],
          invalidPredictions = new Set<number>(),
          from = modelViews.get(source.id),
          to = modelViews.get(target.id),
          guide = rotationGuides.get(`${source.id}:${target.id}`),
          relativeR = from && to ? multiply(to.pose.R, transpose(from.pose.R, 3), 3) : guide?.R,
          firstCamera = from && to ? from.camera : guide?.first,
          secondCamera = from && to ? to.camera : guide?.second;
        for (let i = 0; i < seeds.length; i++) {
          const f = seeds[i]!,
            x = source.keypoints[2 * f]!,
            y = source.keypoints[2 * f + 1]!;
          xy.set([x, y], 2 * i);
          let prediction: readonly [number, number] | undefined,
            kind: ExpandedObservation['prediction'] = 'none';
          affineWarps.set([1, 0, 0, 1], 4 * i);
          if (options.geometryPatchWarp && relativeR && firstCamera && secondCamera) {
            const project = (xx: number, yy: number) => {
              const ray = unprojectPixel(firstCamera, [xx, yy]);
              return (
                ray &&
                cameraProjection(
                  secondCamera,
                  [0, 1, 2].map(
                    (j) =>
                      relativeR[3 * j]! * ray[0] +
                      relativeR[3 * j + 1]! * ray[1] +
                      relativeR[3 * j + 2]!,
                  ),
                )?.pixel
              );
            };
            const left = project(x - 0.5, y),
              right = project(x + 0.5, y),
              top = project(x, y - 0.5),
              bottom = project(x, y + 0.5);
            if (left && right && top && bottom) {
              const matrix = [
                right[0] - left[0],
                bottom[0] - top[0],
                right[1] - left[1],
                bottom[1] - top[1],
              ];
              try {
                affineWarps.set(kltAffineRecords(matrix, 1).forward, 4 * i);
              } catch (e) {
                if (!(e instanceof RangeError)) throw e;
              }
            }
          }
          const point = points.get(key(source.id, f));
          if (to && point) {
            prediction = projectPoint(to.camera, to.pose, point);
            kind = 'finite';
          } else if (relativeR && firstCamera && secondCamera) {
            const uv = unprojectPixel(firstCamera, [x, y]);
            if (uv) {
              const ray = [0, 1, 2].map(
                (a) =>
                  relativeR[3 * a]! * uv[0] + relativeR[3 * a + 1]! * uv[1] + relativeR[3 * a + 2]!,
              );
              prediction = cameraProjection(secondCamera, ray)?.pixel;
              kind = 'rotation';
            }
          }
          const usable = prediction && inside(target, prediction[0], prediction[1], klt.halfWindow);
          // Invalid projections must not become f32 infinities in the tracking buffers.
          guesses.set(prediction && usable ? prediction : [x, y], 2 * i);
          predictions.push(usable ? kind : 'none');
          if (kind !== 'none' && !usable) invalidPredictions.add(i);
        }
        const t0 = performance.now(),
          hasPredictions = predictions.some((p) => p !== 'none');
        const gpu = {
          chunkTracks: options.chunkTracks,
          readBack: false,
          ...(hasPredictions ? { initialTargets: guesses } : {}),
          ...(options.geometryPatchWarp ? { affineWarps } : {}),
        };
        const tracked = cache
          ? await cache.track(source, target, xy, klt, gpu)
          : await trackKltGpu(
              device,
              source.gray,
              target.gray,
              source.width,
              source.height,
              xy,
              klt,
              gpu,
            );
        trackingMs += performance.now() - t0;
        const hostStarted = performance.now(),
          coords = coordinates.get(target.id)!;
        for (let i = 0; i < seeds.length; i++) {
          attempts++;
          if (invalidPredictions.has(i)) {
            reject('invalidPrediction');
            continue;
          }
          if (!kltAlive(tracked.records[4 * i + 3]!)) {
            reject('tracking');
            continue;
          }
          const f = seeds[i]!,
            sx = xy[2 * i]!,
            sy = xy[2 * i + 1]!,
            x = tracked.records[4 * i]!,
            y = tracked.records[4 * i + 1]!;
          const best = patchNcc(
            source,
            target,
            sx,
            sy,
            x,
            y,
            klt.halfWindow,
            options.geometryPatchWarp ? affineWarps : undefined,
            4 * i,
          );
          let second = -1;
          for (let dy = -ambiguityRadius; dy <= ambiguityRadius; dy += 2)
            for (let dx = -ambiguityRadius; dx <= ambiguityRadius; dx += 2) {
              if (dx * dx + dy * dy < 16) continue;
              second = Math.max(
                second,
                patchNcc(
                  source,
                  target,
                  sx,
                  sy,
                  x + dx,
                  y + dy,
                  klt.halfWindow,
                  options.geometryPatchWarp ? affineWarps : undefined,
                  4 * i,
                ),
              );
            }
          const margin = best - second;
          if (!(best >= klt.nccThreshold && margin >= uniqueness)) {
            reject('ambiguousPatch');
            continue;
          }
          let group = owner.get(key(source.id, f));
          if (group === undefined) {
            group = members.push(new Map()) - 1;
            claim(source.id, f, group);
          }
          let nearest = -1,
            distance = joinRadius * joinRadius;
          for (let j = 0; j < coords.length / 2; j++) {
            const d = (x - coords[2 * j]!) ** 2 + (y - coords[2 * j + 1]!) ** 2;
            if (d <= distance) {
              nearest = j;
              distance = d;
            }
          }
          const existing = members[group]!.get(target.id);
          if (
            (existing !== undefined && existing !== nearest) ||
            (nearest >= 0 &&
              owner.has(key(target.id, nearest)) &&
              owner.get(key(target.id, nearest)) !== group)
          ) {
            reject('ownership');
            continue;
          }
          const reverse = source.id > target.id,
            entry = pairEntry(Math.min(source.id, target.id), Math.max(source.id, target.id)),
            sourceLinks = reverse ? entry.b : entry.a,
            targetLinks = reverse ? entry.a : entry.b;
          if (
            (sourceLinks.has(f) && sourceLinks.get(f) !== nearest) ||
            (nearest >= 0 && targetLinks.has(nearest) && targetLinks.get(nearest) !== f)
          ) {
            reject('pairConflict');
            continue;
          }
          let added = false;
          if (nearest < 0) {
            if (coords.length / 2 - target.keypoints.length / 2 >= maxNew) {
              reject('capacity');
              continue;
            }
            nearest = coords.length / 2;
            coords.push(x, y);
            added = true;
          }
          claim(target.id, nearest, group);
          measuredEdges++;
          if (!sourceLinks.has(f)) {
            entry.matches.push(...(reverse ? [nearest, f] : [f, nearest]));
            // Missing descriptor distances are 0 in the mapper. Preserve original candidates on ties.
            entry.distances.push(1 + tracked.records[4 * i + 2]!);
            sourceLinks.set(f, nearest);
            targetLinks.set(nearest, f);
            addedEdges++;
          }
          if (added)
            newObservations.push({
              viewId: target.id,
              featureId: nearest,
              sourceViewId: source.id,
              sourceFeatureId: f,
              measuredPixel: [x, y],
              ncc: tracked.forward[4 * i + 2]!,
              forwardBackwardErrorPx: tracked.records[4 * i + 2]!,
              uniquenessMargin: margin,
              prediction: predictions[i]!,
              ...(options.geometryPatchWarp
                ? { patchWarp: Array.from(affineWarps.subarray(4 * i, 4 * i + 4)) }
                : {}),
              geometry: 'candidate',
            });
        }
        hostValidationMs += performance.now() - hostStarted;
      }
    }
  } finally {
    await cache?.dispose();
  }
  const after = device.statistics;
  return {
    images: frames.map((frame) => ({
      id: frame.id,
      camera: { ...frame.camera },
      keypoints: Float64Array.from(coordinates.get(frame.id)!),
    })),
    candidatePairs: [...output.values()]
      .filter((p) => p.matches.length)
      .map((p) => ({
        first: p.first,
        second: p.second,
        matches: Uint32Array.from(p.matches),
        distances: Float64Array.from(p.distances),
      })),
    newObservations,
    diagnostics: {
      attempts,
      measuredEdges,
      addedEdges,
      newCoordinates: newObservations.length,
      rejected,
      elapsedMs: performance.now() - started,
      trackingMs,
      hostValidationMs,
      uploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
      readbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
      residentSnapshotBytes: after.residentSnapshotBytes - before.residentSnapshotBytes,
      reads: after.reads - before.reads,
      ...(cache ? { cache: cache.statistics() } : {}),
    },
  };
}
