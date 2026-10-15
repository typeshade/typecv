# Experimental TypeShade scoring

**Superseded:** the compiled, tested implementation is now in
[`src/gpu`](../../src/gpu), exported as `typecv/gpu`.
See [GPU architecture and verification](../../docs/gpu-sfm.md).
These original prototype files are retained only as historical sketches.

`epipolar.shade.ts` contains pixel normalization and Sampson-error kernels.
`epipolar-executor.ts` sketches the `resident` + `fuse` call layer. These files
are retained from the SfM prototype and are **not compiled, typechecked, exported
or executed by the core package**. `reconstructTwoView` currently runs on CPU.

To evaluate this path, use a consuming application's actual TypeShade toolchain
(`typeshade/vite` with `{ fuse: true }`, `typeshade/runtime`, `typeshade/fuse`)
and configure its runtime/device. TypeShade is deliberately not a dependency of
the verified CPU core while the integration is unvalidated. Node cannot execute
the shader source directly.

Before promoting this code into the public API, compile it with the selected
TypeShade version, compare f32 CPU/GPU scores with the f64 CPU reference (including
near-zero Sampson denominators), validate lifetime/error paths, and measure
transfer/dispatch costs. The sketch uses `1e20` where the CPU returns Infinity;
its boundary behavior is not exact parity. For RANSAC acceleration, batch model
scores and keep observations resident rather than uploading for every hypothesis.

The integration patterns were based on TypeNN's runtime execution/device and
matmul kernel sources, and TypeShade's loop-kernel example:

- https://github.com/typeshade/typenn/tree/main/src/runtime
- https://github.com/typeshade/typenn/blob/main/src/kernels/matmul.shade.ts
- https://github.com/typeshade/typeshade/blob/main/examples/loop-kernel.shade.ts
