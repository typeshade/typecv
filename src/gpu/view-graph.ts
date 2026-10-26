import type { ViewGraphBackend } from '../reconstruction/view-graph.ts';
import { matchFloatDescriptorsGpu } from './float-matching.ts';
import { verifyImagePairGpu, verifyImagePairsGpu } from './image-pair.ts';
import type { ShadeDevice } from '../device/gpu/device.ts';
import {
  createViewGraphDescriptorCache,
  type DescriptorLease,
  type ViewGraphDescriptorStatistics,
} from './view-graph-descriptors.ts';

export interface GpuViewGraphBackend extends ViewGraphBackend {
  descriptorCacheStatistics(): ViewGraphDescriptorStatistics;
}

/**
 * The GPU backend of `buildViewGraph` (docs/multi-video.md 3.4, 3.5). A batch uploads the first
 * keyframe once and matches it against each of its candidates. `verifyPair` runs as in the GPU
 * mapper (`verifyImagePairGpu`). `executeMany` verifies the pairs of a batch in shared plans. On the CPU tier of `ShadeDevice` it runs the same kernels in
 * Node, so the CPU-tier test reads the same code as the browser.
 */
export function gpuViewGraphBackend(
  device: ShadeDevice,
  /** Pairs that share the plans and the reads of one verification job (Amendment 8). */
  pairsPerJob = 16,
  /**
   * Models in one batch of a pair search. The result does not depend on it (`pair-batch.test`).
   * Every dispatch of a search checks the arrays it binds, so the cost of a dispatch grows with
   * this number, while a pair of a video usually ends its search in the first batch (Amendment 8).
   */
  modelBatch = 64,
  options: { readonly descriptorCacheBytes?: number } = {},
): GpuViewGraphBackend {
  const cache = createViewGraphDescriptorCache(device, options.descriptorCacheBytes);
  return {
    async match(first, others, options) {
      let source: DescriptorLease | undefined;
      let success = false;
      try {
        if (!first.count) {
          first.floatDescriptors();
          return others.map(() => ({
            matches: new Uint32Array(0),
            distances: new Float64Array(0),
          }));
        }
        source = cache.acquire(first);
        const out = [];
        for (const other of others) {
          if (!other.count) {
            out.push({ matches: new Uint32Array(0), distances: new Float64Array(0) });
            continue;
          }
          const target = cache.acquire(other);
          let matched = false;
          try {
            const result = await matchFloatDescriptorsGpu(
              device,
              {
                descriptors: source.buffer,
                count: first.count,
                dimensions: 128,
                elements: source.elements,
              },
              {
                descriptors: target.buffer,
                count: other.count,
                dimensions: 128,
                elements: target.elements,
              },
              { ratio: options.ratio, mutual: true },
            );
            out.push({ matches: result.matches, distances: result.distances });
            matched = true;
          } finally {
            target.release(matched);
          }
        }
        success = true;
        return out;
      } catch (error) {
        cache.clear(true);
        throw error;
      } finally {
        source?.release(success);
      }
    },
    releaseMatching() {
      cache.clear();
    },
    descriptorCacheStatistics: () => cache.statistics(),
    execute(job) {
      const [matches, first, second, options] = job.args;
      return verifyImagePairGpu(device, matches, first, second, {
        batchSize: modelBatch,
        ...options,
      });
    },
    executeMany(jobs) {
      return verifyImagePairsGpu(
        device,
        jobs.map((job) => {
          const [matches, first, second, options] = job.args;
          return { matches, first, second, options: { batchSize: modelBatch, ...options } };
        }),
        pairsPerJob,
      );
    },
  };
}
