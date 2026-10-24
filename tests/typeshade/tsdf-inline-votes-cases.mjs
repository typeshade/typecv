import { digestArrays, inputChannels, pointDigests } from './tsdf-staging-cases.mjs';

/** Group A leaves strong votes; group B's first camera exits early before weaker cameras vote. */
export function dirtyVoteInput(kind) {
  const pixels = 16 * 16;
  const view = (center, depth, confidence) => ({
    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    center: [center, 0, 0],
    fx: 32,
    fy: 32,
    cx: 7.5,
    cy: 7.5,
    depth: new Float32Array(pixels).fill(depth),
    normal: new Uint32Array(pixels).fill(0x7fff7fff),
    confidence: new Float32Array(pixels).fill(confidence),
    rgba: new Uint32Array(pixels).fill(0xff87a1c4),
  });
  const first = view(0.5, kind === 'zero-depth' ? 0 : 4, kind === 'zero-confidence' ? 0 : 1);
  if (kind === 'wrong-normal') first.normal.fill(0);
  if (kind === 'look-away') first.rotation = [-1, 0, 0, 0, 1, 0, 0, 0, -1];
  return {
    width: 16,
    height: 16,
    views: [
      view(0, 2, 1),
      view(0, 2, 1),
      view(0.25, 4, 1),
      first,
      view(0.5, 4, 0.5),
      view(0.5, 4, 0.75),
    ],
  };
}

export async function inlineVotesCase({ ShadeDevice, fuseTsdfGpu }, backend) {
  const device = await ShadeDevice.open({ backend, poolBytes: 0 });
  const results = [];
  try {
    for (const kind of ['zero-depth', 'zero-confidence', 'wrong-normal', 'look-away']) {
      const input = dirtyVoteInput(kind);
      const before = await digestArrays(inputChannels(input));
      for (const maxBlocks of [undefined, 12]) {
        const runs = [];
        for (const standaloneVoteReset of [true, false]) {
          device.resetKernelStatistics();
          const points = await fuseTsdfGpu(device, input, {
            levels: 1,
            voxelScale: 1,
            truncation: 3,
            minViews: 2,
            visibility: 'view-weighted',
            carveRatio: 1e9,
            ...(maxBlocks === undefined ? {} : { maxBlocks }),
            standaloneVoteReset,
          });
          runs.push({
            count: points.count,
            channels: await pointDigests(points),
            groups: points.tsdf.inlineVoteResetGroups,
            passes: points.tsdf.passes,
            maxFree: Math.max(...points.free),
            maxFreeWeight: Math.max(...points.freeWeights),
            kernels: device.kernelStatistics.rows,
            liveBytes: device.statistics.liveBytes,
          });
        }
        results.push({
          kind,
          boxed: maxBlocks !== undefined,
          runs,
          sourceUnchanged: before === (await digestArrays(inputChannels(input))),
        });
      }
    }
    return { results, liveBytes: device.statistics.liveBytes };
  } finally {
    await device.dispose();
  }
}
