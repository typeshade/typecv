# TypeCV

TypeScript geometry for **calibrated two-view and incremental multi-view structure
from motion**, with an optional compiled **TypeShade/WebGPU** path. Supply feature
coordinates and pair matches to recover camera poses and a sparse scene. The
synchronous CPU entry has no runtime dependencies.

The incremental pipeline verifies pair geometry, builds conflict-safe tracks,
selects an initial pair, registers cameras with P3P RANSAC, triangulates tracks,
and runs local/periodic global bundle adjustment. Camera parameters are fixed;
pinhole and two-coefficient radial distortion are supported.

This is an early reconstruction core, not a complete replacement for COLMAP.
Image feature extraction uses an optional OpenCV experiment adapter. The GPU
entry batches Essential hypotheses against resident observations, reduces their
scores and selects a winner on the device. Its two-view API is hybrid: small
f64 solvers and pose/point refinement still run on CPU. The incremental mapper
and bundle adjustment currently run on CPU.

## Development

Use Node.js 22.18+ or Node.js 24:

```sh
npm ci
npm run check                 # formatting, strict types, geometry tests, ESM/declarations
npm run example               # original two-view example
npm run example:incremental   # generated multi-view scene through the package export
npx playwright install --only-shell chromium
npm run test:webgpu          # actual WGSL execution using software WebGPU
npm run benchmark:gpu        # transfer/dispatch accounting and reproducible samples
```

The package is not published on npm. `npm run build` produces `dist/`.

## Incremental reconstruction

```ts
import { reconstructIncremental } from './dist/index.js';

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
`Uint32Array` packed feature-index pairs `[i,j,...]`, not pixel coordinates.
`K` contains `{fx,fy,cx,cy,k1?,k2?}`. Distortion coefficients default to zero.
See [API, invariants, algorithms and limitations](docs/incremental-sfm.md).

## Two-view reconstruction

The original API remains available for **undistorted** pixel correspondences:

```ts
import { reconstructTwoView } from './dist/index.js';
const K = { fx: 900, fy: 900, cx: 640, cy: 360 };
const result = reconstructTwoView(matches, K, K, {
  iterations: 1000,
  threshold: 0.0015,
  seed: 42,
});
```

`matches` is `[x1,y1,x2,y2,...]` in a `Float64Array`. Each call has its own unit
baseline; pairwise clouds cannot simply be concatenated. See [two-view options](docs/sfm.md).

## Evidence and next work

- 46 deterministic tests include ground-truth pose/structure recovery, analytical
  Jacobian checks, radial inversion, P3P branches, occlusion/new-track growth,
  disconnected frames, outliers, local/global BA gauges and input immutability.
- Compiled TypeShade CPU differential tests and actual Chromium WebGPU tests
  cover scoring, fusion, transfers, resource lifetimes and hybrid reconstruction.
  GPU CI uses SwiftShader; physical-GPU performance remains unmeasured.
- On a free pyramid aerial clip, a controlled-input comparison registers all
  eight frames with both TypeCV and COLMAP. TypeCV recovers 1,729 points versus
  1,848 for the reference. This is an uncalibrated feasibility result, not proof
  of comparable general accuracy. [Reproduce and inspect all results](docs/experiments/incremental-pyramid/README.md).
- Shared calibration optimization, five-point essential estimation, scalable
  sparse camera solvers, native image features, GPU multi-view optimization and dense MVS remain
  open. [Implementation and validation roadmap](docs/roadmap.md).

## TypeShade / WebGPU two-view path

```ts
import { ShadeDevice, reconstructTwoViewGpu } from 'typecv/gpu';

const device = await ShadeDevice.open(); // Requires WebGPU; does not silently use CPU.
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

The package is unpublished; local applications can import `./dist/gpu.js`.
Inputs must be undistorted pinhole pixels. [`EpipolarSession`](docs/gpu-sfm.md)
exposes resident scoring separately for custom proposal/mapper integration.
See [GPU architecture, TypeNN sources, measured scope and next kernels](docs/gpu-sfm.md).

## License

[Apache-2.0](LICENSE).
