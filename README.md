# TypeCV

Computer vision primitives for TypeShade. The first module provides calibrated
**two-view structure from motion (SfM)** from matched pixel coordinates.

The CPU solver is implemented in TypeScript with no runtime dependencies. It
estimates relative camera pose and a sparse point cloud; it does not yet extract
features from images or reconstruct multiple views. TypeShade scoring kernels
are included as an [experimental integration](experimental/typeshade/README.md).

## Development

Use Node.js 22.18+ (or Node.js 24) and npm:

```sh
npm ci
npm run check    # strict type checking, tests, ESM/declaration build
npm run example  # build and reconstruct a generated scene via the package export
```

This repository is not yet published as an npm package. `npm run build` generates
ES modules and declarations in `dist/`. The package root exports
`reconstructTwoView`, `triangulate`, `sampsonSquared` and their public SfM types.

## Two-view reconstruction

```ts
import { reconstructTwoView } from './dist/index.js';

const K = { fx: 900, fy: 900, cx: 640, cy: 360 };
const matches = Float64Array.from([
  // x1, y1, x2, y2 for each correspondence (at least 8 independent matches)
]);
const result = reconstructTwoView(matches, K, K, {
  iterations: 1000,
  threshold: 0.0015,
  seed: 42,
});

console.log(result.R, result.t, result.diagnostics);
for (let i = 0; i < result.validPoints.length; i++) {
  if (result.validPoints[i]) console.log(result.points.subarray(3 * i, 3 * i + 3));
}
```

`R` is row-major and follows `X2 = R * X1 + t`. Translation `t` is a unit direction:
monocular two-view reconstruction cannot determine metric scale. Points are in
camera-1 coordinates, in units of this baseline. Rejected observations have NaN
triplets. The two cameras can have different intrinsics.

### What the solver does

- Pixel-to-camera normalization and Hartley-normalized eight-point estimation.
- Essential-matrix constraint projection and seeded adaptive RANSAC with local refit.
- Four-pose cheirality disambiguation and DLT triangulation.
- Optional nonlinear pose/point refinement, with robust pixel residuals.
- Low-parallax and per-view reprojection filtering, plus quality diagnostics.

See [SfM options, result semantics and limitations](docs/sfm.md) for details.
The camera model assumes **undistorted pinhole observations**. Real images,
noisy planar scenes and TypeShade CPU/GPU parity still need validation; the
current tests use generated correspondences.

## License

[Apache-2.0](LICENSE).
