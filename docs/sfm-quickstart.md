# Structure from motion in TypeCV

SfM belongs to TypeCV's geometry and 3D vision scope. It is currently the most
developed pipeline: feature coordinates and pair matches produce camera poses
and a sparse scene. See the [project architecture](architecture.md) for the
broader computer vision design.

The incremental pipeline verifies geometry, builds conflict-safe tracks, selects
an initial pair, registers cameras with P3P RANSAC, triangulates tracks and runs
local/periodic global BA. Intrinsics are fixed; pinhole and radial k1/k2 cameras
are supported. This is an early reconstruction core. Native features, stronger
minimal solvers, shared calibration optimization and dense MVS remain open.

The package is unpublished. Run `npm ci` and `npm run build` before the local
imports below. All snippets assume that caller-supplied matches/features exist.

## Incremental reconstruction

```ts
import { reconstructIncremental } from './dist/index.js';
const K = { fx: 900, fy: 900, cx: 640, cy: 360 };

const result = reconstructIncremental(
  [
    { id: 0, camera: K, keypoints: firstKeypoints },
    { id: 1, camera: K, keypoints: secondKeypoints },
    { id: 2, camera: K, keypoints: thirdKeypoints },
  ],
  [
    { first: 0, second: 1, matches: firstSecondIndices },
    { first: 1, second: 2, matches: secondThirdIndices },
    { first: 0, second: 2, matches: firstThirdIndices },
  ],
  { seed: 42 },
);
console.log(result.reconstruction.views, result.reconstruction.landmarks);
console.log(result.unregistered, result.diagnostics);
```

`keypoints` are `Float64Array` packed `[x,y,...]` pixels. `matches` are
`Uint32Array` packed feature-index pairs `[i,j,...]`. `K` contains
`{fx,fy,cx,cy,k1?,k2?}`; distortion defaults to zero.
See [contracts, algorithms and limitations](incremental-sfm.md).

## Two-view reconstruction

```ts
import { reconstructTwoView } from './dist/index.js';
const K = { fx: 900, fy: 900, cx: 640, cy: 360 };
const result = reconstructTwoView(matches, K, K, {
  iterations: 1000,
  threshold: 0.0015,
  seed: 42,
});
```

This API requires undistorted pixel correspondences `[x1,y1,x2,y2,...]` in a
`Float64Array`. Each call has its own unit baseline; pairwise clouds cannot be
concatenated into a common-scale reconstruction. See [two-view details](sfm.md).

## TypeShade / WebGPU two-view path

```ts
import { ShadeDevice, reconstructTwoViewGpu } from './dist/gpu.js';

const K = { fx: 900, fy: 900, cx: 640, cy: 360 };
const device = await ShadeDevice.open();
try {
  const result = await reconstructTwoViewGpu(device, matches, K, K, {
    iterations: 1000,
    threshold: 0.0015,
    seed: 42,
  });
  console.log(result.R, result.t, result.execution);
} finally {
  await device.dispose();
}
```

The package subpath is `typecv/gpu` for applications consuming the package.
WebGPU is required by default. Observations remain resident across batched
Essential scoring and GPU reductions. This API is hybrid: small f64 solvers and
pose/point refinement run on CPU. The incremental mapper and BA also currently
run on CPU. See [GPU contracts, TypeNN references and verification](gpu-sfm.md).

## Evidence and next work

The [exploratory two-view video experiment](experiments/pyramid-8134952/README.md)
records seven temporal gaps and three assumed focal lengths. Its scripts use
optional Python/OpenCV feature preparation and TypeCV's CPU geometry; all 21
original configurations and their provenance are preserved.

Current tests cover known-pose recovery, analytical Jacobians, radial inversion,
P3P branches, tracks, local/global BA gauges, compiled scoring, resource lifetimes
and actual WGSL execution. GPU CI uses SwiftShader; hardware performance is
unmeasured.

The [pyramid comparison](experiments/incremental-pyramid/README.md) registers
eight frames with both TypeCV and COLMAP from shared prepared input. TypeCV
recovers 1,729 points versus 1,848 for the reference. This is an assumed-calibration
feasibility case, not proof of comparable general accuracy. The
[SfM roadmap](sfm-roadmap.md) records remaining algorithms and validation gates.
