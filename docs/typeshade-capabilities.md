# TypeShade capability review

TypeCV's earlier hybrid design reflected implementation choices, not a requirement
that TypeShade delegate minimal solvers or precise arithmetic to CPU code. The
earlier GPU plan discussed WebGPU's native f64 absence without evaluating
TypeShade's existing f64 emulation. That was an incomplete basis for the design.
It also omitted the compiler's differentiation and shared-IR oracle facilities.

## Review baseline

Reviewed TypeShade `212d26d2fb9fb6dd39b72674e537653fb0f7e888`, which is both
TypeCV's pinned dependency and the main ref observed during this review, and
TypeNN `61259c3733148a8665466163ae1c5c2b920b129c`. TypeCV's starting revision is
`a627acabade9f3597f03885be2584124f2b18779`.

Primary sources below identify delivered behavior. The probe results establish
only the small cases executed here; they do not establish completed GPU SfM.

| Facility                                                                      | Primary source at the reviewed TypeShade revision                                                                                                                                                                                                                                                                                                                                                                                            | Relevance to TypeCV                                                                                      |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Emulated `f64`, vectors, selected matrix operations and flavor/guard handling | [Authoring fp64](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/AUTHORING.md#fp64), [surface §39](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/docs/use-typeshade-surface.md#39-f64-the-emulated-double)                                                                                                                                                       | Evaluate device precision for root finding and geometry; lack of native f64 does not mandate CPU solvers |
| Functions, structs, fixed/runtime arrays and control flow                     | [Authoring guide](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/AUTHORING.md#control-flow)                                                                                                                                                                                                                                                                                                            | Express bounded solver iterations and candidate records in shader source                                 |
| Workgroup memory, barriers, integer atomics and specialization                | [Workgroup reduction](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/examples/workgroup-reduce.shade.ts), [workgroup override](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/examples/workgroup-override.shade.ts), [atomic histogram](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/examples/atomic-histogram.shade.ts) | Cooperative reductions, tiled work, counters and compaction                                              |
| Automatic loop parallelization and reductions                                 | [Authoring kernels/fusion](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/AUTHORING.md#a-loop-over-an-array)                                                                                                                                                                                                                                                                                           | Map/transform/reduce operators need not be written as CPU orchestration with host intermediates          |
| Forward and reverse differentiation, `gradCheck`                              | [Authoring derivatives](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/AUTHORING.md#derivatives-with-grad)                                                                                                                                                                                                                                                                                             | Generate supported projection/residual derivatives and compare with independent formulas                 |
| Shared-IR CPU oracle with f64/f32 evaluation modes                            | [CPU precision modes](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/AUTHORING.md#the-f64-and-f32-precision-modes)                                                                                                                                                                                                                                                                                     | Validate the same authored operator before/alongside device execution, plus independent references       |
| Arbitrary resident host values, including Float64Array and structured buffers | [Resident API](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/src/core/resident.ts), [f64 kernel tests](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/src/compiler/ts/host-kernel.test.ts)                                                                                                                                                                      | TypeCV now pools f32/f64/u32 separately; its previous f32-only owner was a local restriction             |
| Explicit compute entries and a program/frame runtime alongside kernel fusion  | [Host compute](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/examples/hosts/host-compute/host.ts), [runtime interfaces](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/src/runtime/program.ts), [fuse implementation/tests](https://github.com/typeshade/typeshade/blob/212d26d2fb9fb6dd39b72674e537653fb0f7e888/src/fuse.test.ts)                              | Choose explicit dispatch when automatic loop lowering/fusion is not the right execution model            |

The source documentation also covers classes/generics, textures and other shader
facilities. This review does not claim to execute every supported construct.

## Precision: three different meanings

TypeShade's shader `f64` lowers to a pair of f32 words. Its documented precision
is about 48 significand bits over the f32 exponent range, with operation/driver
assumptions described in the authoring guide. It is not IEEE binary64. The float
flavor uses an injected guard texture; the integer flavor changes the lowering
to resist platform fast-math behavior. The program runtime binds the guard from
the manifest.

The CPU oracle's default f64 _evaluation mode_ is separate: it uses JavaScript
doubles even for f32 operations. Its f32 mode rounds f32-typed operations. The
oracle evaluates shader `f64` as doubles rather than reproducing every step of
the GPU pair representation. These are complementary references with different
tolerances, not three interchangeable claims of double precision.

The zero-allocation `resident(length, element)` convenience accepts f32/i32/u32.
That does not restrict `resident(value)` to those types: the probes use
`resident(Float64Array)` and resident arrays of records containing f64 fields.
The runtime packs those values according to reflected layouts. TypeCV's resource
owner now supports f32/f64/u32 and separates their pools by element type and length.

## Authored code and its execution tier

A `.shade.ts` helper called directly from host code can execute synchronously on
CPU. A kernel loop is parallelized when the compiler proves its independence;
an unsupported loop may carry a `TS8070` diagnostic. A `@compute` entry explicitly
defines invocations/workgroups. Therefore, adding the directive to a file alone
does not establish a GPU implementation.

Choose the appropriate form, require the intended tier, inspect compiler
diagnostics and emitted code, then count actual device dispatches. Host setup,
pipeline construction and queue submission remain host responsibilities.

## Executable probes

```sh
npm ci
npx playwright install --only-shell chromium
npm run audit:typeshade
# Separate opt-in validation of the integer lowering:
TYPECV_AUDIT_INTEGER=1 npm run audit:typeshade
```

The script compiles the actual source under
[`experimental/typeshade/audit`](../experimental/typeshade/audit), uses `grad` at
build time, emits manifests with `packModule`, and runs them through the public
program runtime in Chromium. The browser receives no compiler. These probes
are independent of the existing f32 epipolar scorer.

The [recorded software WebGPU result](experiments/gpu/typeshade-capabilities.json)
contains five observed compute dispatches and fourteen queue submissions,
including nine explicit result reads:

- At coordinates near ±1e8 and 2²⁴, GPU pair arithmetic retains a 0.125 increment
  that ordinary f32 arithmetic loses. Input/output storage uses resident f64.
  The ordinary f32 sum is stored before a host comparison with the rounded input;
  this prevents the driver from reassociating its cancellation into the increment.
- Bounded shader loops solve eight explicitly bracketed quartic roots, including
  roots separated by 2⁻¹⁴. A ninth non-bracketing case is rejected. This tests
  precise arithmetic, structured storage, branches and iteration; it is **not**
  P3P, an all-roots solver or a degeneracy-completeness test.
- One workgroup/barrier/atomic kernel runs at specialized sizes 32 and 64 on
  513 values, checks block-tail sums and counts 219 positive values.
- A forward derivative generated from a TypeShade projection function runs on
  GPU and matches the independent analytic derivative at four inputs. Forward
  and reverse `gradCheck` also pass on the CPU oracle (4 and 16 checks). The
  reverse derivative is **not** executed on GPU by this probe, and this does not
  establish differentiation support for every f64 operation or a full BA model.

Negative controls require the compiler to reject a broken program, ordinary f32
to lose the small increment, and the root probe to reject the invalid bracket.
The browser records actual calls and checks uncaptured GPU errors. Its work is
bounded by a 60-second deadline.

Both fp64 flavors are emitted. The recorded default run executes the float
flavor on SwiftShader. An integer-flavor device attempt exceeded the 60-second
deadline during its precision probe in this environment; the trace did not
isolate pipeline creation from execution/readback, and the attempt is not
reported as passing. The opt-in command retains that separate validation path.
This observation does not establish a lack of integer-flavor support on other
adapters. Physical-GPU correctness/performance and platform flavor selection
remain separate checks.

## What the TypeNN review establishes

TypeNN's [execution graph](https://github.com/typeshade/typenn/blob/61259c3733148a8665466163ae1c5c2b920b129c/src/runtime/execute.ts)
keeps operation results resident and submits pending graph nodes in a fuse
scope. Its [device datasets](https://github.com/typeshade/typenn/blob/61259c3733148a8665466163ae1c5c2b920b129c/src/data/device-dataset.ts)
and gather kernels illustrate persistent data, batching and storage-aware
access. Those are reusable execution patterns.

The inspected [matmul kernel](https://github.com/typeshade/typenn/blob/61259c3733148a8665466163ae1c5c2b920b129c/src/kernels/matmul.shade.ts)
uses one output-parallel loop with a serial inner product. The earlier phrase
"TypeNN-style tiled compute" was not supported by that source. TypeShade itself
supplies the workgroup facilities needed for tiled kernels; this review does
not claim that the inspected TypeNN matmul implements them.

## Implementation criteria for TypeCV

1. Author new/reworked numerical operators in TypeShade. Use the shared-IR oracle
   and independent CPU references to validate them; select an explicit GPU path
   for production device operations. Retain the existing CPU API honestly as a
   compatibility/reference implementation while migration is incomplete.
2. P3P now has bounded derivative-based all-real-root isolation, distance refinement,
   rigid-pose construction and branch validation in
   [`p3p.shade.ts`](../src/gpu/p3p.shade.ts). Its explicit compute entry runs one
   problem per invocation and retains four candidate slots per problem. Tests
   compare the compiled shared-IR CPU tier and actual GPU execution with the
   independent recursive CPU solver. [GPU PnP](gpu-geometry.md) now composes
   resident sampling, this P3P entry, cooperative scoring and pose LM; the
   incremental driver consumes it.
3. GPU PnP, direct-SVD triangulation, E/H proposals/refits and relative-pose
   recovery/refinement now use this model. See the [initialization path](gpu-initialization.md).
   Select f32/emulated f64 by measured conditioning and error, then validate the
   necessary branches, degeneracies, coordinate ranges and solver tolerances.
4. Joint GPU BA now has analytic shader Jacobians, sparse blocks, matrix-free
   Schur/PCG, block preconditioning, device LM decisions and gauge/rollback tests.
   Generated derivatives, broader calibration and large-scene scaling remain
   separate validation work; the AD probe is not a claim about the whole optimizer.
5. Keep data resident through algorithm stages. Judge the implementation by its
   actual GPU computation, transfers and task accuracy rather than the presence
   of a directive or a few accelerated scoring calls.
