import { ShadeDevice, type ShadePlanReport } from './device.ts';
import { extractFeaturesGpu, type GpuFeatureOptions, type GpuFeatures } from './features.ts';

export interface ImageRegion {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/** A decoder, tiled image store or procedural source; no full-frame GPU texture is required. */
export interface GpuImageSource {
  readonly width: number;
  readonly height: number;
  readRegion(
    region: ImageRegion,
  ): Promise<Uint8Array | Uint8ClampedArray> | Uint8Array | Uint8ClampedArray;
}

export interface GpuTiledFeatureOptions extends GpuFeatureOptions {
  /** Core edge, aligned to the detector's 24-pixel selection cells. Default: 480. */
  readonly tileEdge?: number;
  readonly onTile?: (event: { completed: number; total: number }) => void;
}

export interface GpuTiledFeatures extends GpuFeatures {
  readonly tiling: {
    readonly sourceWidth: number;
    readonly sourceHeight: number;
    readonly tiles: number;
    readonly halo: number;
    readonly peakRegionBytes: number;
    readonly totalGroups: number;
    readonly groupsTruncated: boolean;
  };
}

/** Lazy source-resolution-independent schedule. Read regions include a 24-pixel halo. */
export function* featureTileRegions(
  width: number,
  height: number,
  edge = 480,
): Generator<{
  core: ImageRegion;
  read: ImageRegion;
}> {
  if (
    !Number.isSafeInteger(width) ||
    width < 48 ||
    !Number.isSafeInteger(height) ||
    height < 48 ||
    !Number.isSafeInteger(edge) ||
    edge < 48 ||
    edge > 480 ||
    edge % 24 !== 0
  )
    throw new RangeError('Invalid feature tile dimensions');
  for (let y = 0; y < height; y += edge) {
    for (let x = 0; x < width; x += edge) {
      const right = Math.min(x + edge, width),
        bottom = Math.min(y + edge, height);
      let left = Math.max(0, x - 24),
        top = Math.max(0, y - 24);
      const readRight = Math.min(width, right + 24),
        readBottom = Math.min(height, bottom + 24);
      if (readRight - left < 48) left = Math.max(0, left - 24);
      if (readBottom - top < 48) top = Math.max(0, top - 24);
      yield {
        core: { x, y, width: right - x, height: bottom - y },
        read: { x: left, y: top, width: readRight - left, height: readBottom - top },
      };
    }
  }
}

type Corner = { x: number; y: number; score: number; descriptor: Uint32Array };
const worse = (a: Corner, b: Corner) =>
  a.score < b.score || (a.score === b.score && (a.y > b.y || (a.y === b.y && a.x > b.x)));

/**
 * Full-resolution FAST/BRIEF through bounded tiles, following TypeNN's device-limit shards.
 * TypeShade executes the existing kernels; the host reads at most one overlapping region
 * and retains a bounded heap of compact features. Halo pixels never become duplicate outputs.
 */
export async function extractFeaturesTiledGpu(
  device: ShadeDevice,
  source: GpuImageSource,
  options: GpuTiledFeatureOptions = {},
): Promise<GpuTiledFeatures> {
  const halo = 24; // Covers FAST-9 and the rotated BRIEF radius (ceil(sqrt(8²+8²)) = 12).
  const edge = options.tileEdge ?? 480;
  const maxFeatures = options.maxFeatures ?? 2000;
  if (
    !Number.isSafeInteger(source.width) ||
    source.width < 48 ||
    !Number.isSafeInteger(source.height) ||
    source.height < 48 ||
    typeof source.readRegion !== 'function' ||
    !Number.isSafeInteger(edge) ||
    edge < 48 ||
    edge > 480 ||
    edge % 24 !== 0 ||
    !Number.isSafeInteger(maxFeatures) ||
    maxFeatures < 1 ||
    maxFeatures > 20000 ||
    (options.onTile !== undefined && typeof options.onTile !== 'function')
  )
    throw new RangeError('Invalid tiled image source or feature budget');
  const maximumRegion =
    Math.min(source.width, edge + 2 * halo) * Math.min(source.height, edge + 2 * halo);
  // Mirrors the feature kernel's working/dispatch bounds before asking a decoder for pixels.
  if (
    maximumRegion * 32 > device.workingBytes ||
    maximumRegion * 4 > device.bindingBytes ||
    maximumRegion > device.maxKernelElements
  )
    throw new RangeError('Tile exceeds TypeShade device limits; reduce tileEdge');
  const nx = Math.ceil(source.width / edge),
    ny = Math.ceil(source.height / edge);
  const total = nx * ny;
  if (!Number.isSafeInteger(total)) throw new RangeError('Tile grid exceeds safe integer bounds');
  const heap: Corner[] = [];
  const retain = (corner: Corner) => {
    if (heap.length < maxFeatures) {
      heap.push(corner);
      let child = heap.length - 1;
      while (child > 0) {
        const parent = (child - 1) >> 1;
        if (!worse(heap[child]!, heap[parent]!)) break;
        [heap[child], heap[parent]] = [heap[parent]!, heap[child]!];
        child = parent;
      }
    } else if (worse(heap[0]!, corner)) {
      heap[0] = corner;
      let parent = 0;
      for (;;) {
        let child = parent * 2 + 1;
        if (child >= heap.length) break;
        if (child + 1 < heap.length && worse(heap[child + 1]!, heap[child]!)) child++;
        if (!worse(heap[child]!, heap[parent]!)) break;
        [heap[child], heap[parent]] = [heap[parent]!, heap[child]!];
        parent = child;
      }
    }
  };
  let snapshotBytes = 0;
  let dispatches = 0,
    submissions = 0,
    uploaded = 0,
    readback = 0,
    peakRegionBytes = 0,
    completed = 0,
    totalGroups = 0;
  const groups: ShadePlanReport['groups'][number][] = [];
  for (const { core, read: region } of featureTileRegions(source.width, source.height, edge)) {
    const { x, y } = core,
      right = x + core.width,
      bottom = y + core.height;
    const left = region.x,
      top = region.y;
    const rgba = await source.readRegion(region);
    if (
      !(rgba instanceof Uint8Array || rgba instanceof Uint8ClampedArray) ||
      rgba.length !== region.width * region.height * 4
    )
      throw new RangeError('Image source returned the wrong RGBA region length');
    peakRegionBytes = Math.max(peakRegionBytes, rgba.byteLength);
    const found = await extractFeaturesGpu(device, rgba, region.width, region.height, {
      threshold: options.threshold,
      maxFeatures: 2000, // Four slots per 24-pixel cell, independent of source dimensions.
    });
    dispatches += found.report.dispatches;
    submissions += found.report.submissions;
    totalGroups += found.report.groups.length;
    // Keep diagnostics bounded as well as pixels; totals always describe the entire run.
    groups.push(...found.report.groups.slice(0, Math.max(0, 128 - groups.length)));
    uploaded += found.execution?.applicationUploadBytes ?? 0;
    readback += found.execution?.applicationReadbackBytes ?? 0;
    snapshotBytes += found.execution?.residentSnapshotBytes ?? 0;
    for (let i = 0; i < found.scores.length; i++) {
      const px = found.keypoints[2 * i]! + left,
        py = found.keypoints[2 * i + 1]! + top;
      if (px < x || px >= right || py < y || py >= bottom) continue;
      const candidate = { x: px, y: py, score: found.scores[i]!, descriptor: new Uint32Array(0) };
      if (heap.length >= maxFeatures && !worse(heap[0]!, candidate)) continue;
      candidate.descriptor = found.descriptors.slice(i * 8, i * 8 + 8);
      retain(candidate);
    }
    options.onTile?.({ completed: ++completed, total });
  }
  heap.sort((a, b) => a.y - b.y || a.x - b.x);
  const keypoints = new Float64Array(heap.length * 2),
    scores = new Float32Array(heap.length);
  const descriptors = new Uint32Array(heap.length * 8);
  heap.forEach((corner, i) => {
    keypoints[2 * i] = corner.x;
    keypoints[2 * i + 1] = corner.y;
    scores[i] = corner.score;
    descriptors.set(corner.descriptor, i * 8);
  });
  return {
    keypoints,
    scores,
    descriptors,
    capacity: maxFeatures,
    report: { tier: device.backend, dispatches, submissions, groups },
    execution: {
      applicationUploadBytes: uploaded,
      applicationReadbackBytes: readback,
      residentSnapshotBytes: snapshotBytes,
    },
    tiling: {
      sourceWidth: source.width,
      sourceHeight: source.height,
      tiles: total,
      halo,
      peakRegionBytes,
      totalGroups,
      groupsTruncated: totalGroups > groups.length,
    },
  };
}
