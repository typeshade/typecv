import { mvsScene } from './mvs-cases.mjs';
import { planarInput, planarMvsCase } from './planar-depth-cases.mjs';

export async function observationPathCases(api, backend) {
  const scene = mvsScene({ width: 49, height: 37 }),
    plan = api.planMvsViews(scene),
    device = await api.ShadeDevice.open({ backend, poolBytes: 0 });
  try {
    const images = new Map(
      [...scene.images].map(([id, image]) => {
        const values = new Uint8Array(image.width * image.height);
        // Three independent excluded optical paths, not a segmentation result.
        for (let y = 14; y < 22; y++)
          for (let x = 20; x < 28; x++) values[y * image.width + x] = 1 + (x % 3);
        return [
          id,
          {
            ...image,
            observationMask: {
              width: image.width,
              height: image.height,
              values,
              source: 'ground-truth',
            },
          },
        ];
      }),
    );
    const reference = 2,
      mask = images.get(plan.views[reference].id).observationMask.values;
    let expected;
    for (const streamGroup of [undefined, 1]) {
      const maps = await api.mvsDepthMapsGpu(device, scene, plan, images, {
        photometricIterations: 2,
        geometricIterations: 1,
        readStages: true,
        ...(streamGroup ? { streamGroup, chunkPixels: 65 } : {}),
      });
      let kept = 0;
      for (const view of maps.views)
        for (const stage of [view, view.photometric, view.geometric]) {
          if (!stage) continue;
          for (let i = 0; i < mask.length; i++)
            if (mask[i] && stage.depth[i] !== 0)
              throw new Error('Excluded optical path escaped into a depth map');
        }
      const depths = maps.views[reference].depth;
      for (let i = 0; i < mask.length; i++) if (!mask[i] && depths[i] > 0) kept++;
      if (kept < 100) throw new Error('Mask discarded unrelated finite geometry');
      if (expected && depths.some((z, i) => z !== expected[i]))
        throw new Error('Masked grouped maps differ');
      expected = depths.slice();
    }
    // The gray image is masked after undistortion, keeping the original bytes untouched.
    const before = images.get(plan.views[reference].id).rgba.slice();
    const photo = await api.mvsPhotometricGpu(device, scene, plan, images, reference, {
      photometricIterations: 0,
      readGray: true,
    });
    for (let v = 0; v < plan.packed.length; v++)
      for (let i = 0; i < mask.length; i++)
        if (mask[i] && photo.gray[v * mask.length + i] !== -1)
          throw new Error('Excluded source intensity remains usable');
    if (before.some((x, i) => x !== images.get(plan.views[reference].id).rgba[i]))
      throw new Error('Original RGB changed');

    const { input } = planarInput(),
      excluded = input.views.map(() => new Uint8Array(256));
    for (let y = 5; y < 11; y++) for (let x = 5; x < 8; x++) excluded[0][y * 16 + x] = 1;
    const filled = await api.completePlanarDepths(device, input, { excludedPixels: excluded });
    if (filled.report.filled !== 18)
      throw new Error('Planar fill ignored optical-path policy: ' + filled.report.filled);
    for (let i = 0; i < 256; i++)
      if (excluded[0][i] && (filled.input.views[0].depth[i] || filled.input.views[0].confidence[i]))
        throw new Error('Planar completion restored a masked surface');
    if (device.statistics.liveBytes) throw new Error('Observation-path arrays leaked');
    const integrated = await planarMvsCase(api, backend, true);
    if (integrated.filled !== 18 || integrated.remainingHoles !== 18 || integrated.liveBytes !== 0)
      throw new Error('Dense reconstruction did not forward the optical mask to planar fusion');
    return {
      backend: device.backend,
      kept: expected.filter((z) => z > 0).length,
      filled: filled.report.filled,
      liveBytes: device.statistics.liveBytes,
    };
  } finally {
    await device.dispose();
  }
}
