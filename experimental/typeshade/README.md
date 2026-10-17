# Experimental TypeShade references

## Tested normalized-sampling reference

`normalized-sampling-reference.shade.ts` preserves the original arithmetic for
the physical WebGPU normalized-sampling tests. The tested implementation is the
internal [`normalized-sampling.shade.ts`](../../src/gpu/normalized-sampling.shade.ts)
module. It computes normalization denominators once and divides sampled values
before interpolation, preserving the original f32 operation order. This module
is not a public feature-extraction API.

An experimental resident ALIKED integration uses this primitive to avoid a dense
normalized feature tensor. In a native 640 × 480, twelve-frame TUM excerpt on an
RTX 2080, all twelve feature outputs matched bit for bit, and all five complete
reconstructions matched the control exactly. Reported logical feature-buffer
payload fell from 856,767,364 to 700,713,860 bytes (18.21%); this is not a peak
VRAM measurement. The website and public feature extractor do not automatically
use this experimental learned-feature integration.

| Measured configuration                         | Mean of four warm whole-pipeline samples |
| ---------------------------------------------- | ---------------------------------------: |
| Original source control                        |                             5,392.775 ms |
| Final source, serial pair verification         |                             5,008.375 ms |
| Final source, repeated serial control          |                             5,147.775 ms |
| Final source, four independent pairs per batch |                             4,621.250 ms |

The final batch run's first measured trial took 72.3582 seconds, excluding input
preparation and startup. Same-source serial controls establish observed ratios
of 1.084× and 1.114×; the original-source comparison is 1.167×. These are separate
observations on one small scene. Quality was preserved, and kernel occupancy was
not measured. See the [complete benchmark evidence](../../docs/experiments/gpu/sfm-parallel-geometry-rtx2080.json)
and [pair-batching contract](../../docs/gpu-sfm.md#independent-pair-batching).

## Historical epipolar sketches

**Superseded:** the compiled, tested implementation is now in
[`src/gpu`](../../src/gpu), exported as `typecv/gpu`.
See [GPU architecture and verification](../../docs/gpu-sfm.md).
The two epipolar prototype files below are retained only as historical sketches.

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
