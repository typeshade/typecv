# TypeCV

Dependency-free TypeScript CPU geometry for **calibrated two-view and incremental
multi-view structure from motion**. Supply feature coordinates and pair matches
to recover camera poses and a sparse scene in one common coordinate system.

The incremental pipeline verifies pair geometry, builds conflict-safe tracks,
selects an initial pair, registers cameras with P3P RANSAC, triangulates tracks,
and runs local/periodic global bundle adjustment. Camera parameters are fixed;
pinhole and two-coefficient radial distortion are supported.

This is an early reconstruction core, not a complete replacement for COLMAP.
Image feature extraction uses an optional OpenCV experiment adapter. The
[experimental TypeShade scoring code](experimental/typeshade/README.md) is not
part of the verified CPU implementation and has not been compiled or executed.

## Development

Use Node.js 22.18+ or Node.js 24:

```sh
npm ci
npm run check                 # formatting, strict types, geometry tests, ESM/declarations
npm run example               # original two-view example
npm run example:incremental   # generated multi-view scene through the package export
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
- On a free pyramid aerial clip, a controlled-input comparison registers all
  eight frames with both TypeCV and COLMAP. TypeCV recovers 1,729 points versus
  1,848 for the reference. This is an uncalibrated feasibility result, not proof
  of comparable general accuracy. [Reproduce and inspect all results](docs/experiments/incremental-pyramid/README.md).
- Shared calibration optimization, five-point essential estimation, scalable
  sparse camera solvers, native image features, GPU parity and dense MVS remain
  open. [Implementation and validation roadmap](docs/roadmap.md).

## License

[Apache-2.0](LICENSE).
