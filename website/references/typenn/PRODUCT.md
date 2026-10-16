# TypeNN product

## Platform

web

Astro static site with React islands, TailwindCSS v4, and Ant Design v5. English pages use `/`;
Korean pages use `/ko/`. The site is served at `nn.typeshade.dev` only; there is no redirect from any other host.

## Users

TypeScript developers evaluating a compiler-first neural-network framework. They want to run a
small operation, see the code that produced it, understand the Tensor-to-kernel execution path,
and find current API and training guidance without mistaking planned features for shipped work.

## Product purpose

TypeNN is a TypeScript neural-network framework built on TypeShade. TypeNN owns Tensor and Shape
semantics, its deferred Tensor graph, modules, Tensor-graph autograd, optimizers, and execution
scheduling. TypeShade owns its shader language, compiler, generated backend programs, and runtime.

The website helps users move from a quick operation to the execution model, API, model training,
and repository examples. The Playground runs the built TypeNN package in the browser on actual
CPU/WebGPU/WebGL2 paths. It is a small correctness demonstration and does not claim benchmark
quality.

## Positioning

TypeNN makes the relationship between neural-network semantics and GPU compilation explicit.
Users define Tensor and Module code in TypeScript. TypeNN creates and schedules a graph. TypeShade
compiles internal kernels and runs them on the selected device.

The site is pre-release documentation backed by the public TypeNN source repository. The npm
package metadata describes a placeholder package; the site links to source instead of promising
an install command that is not ready.

## Capabilities and constraints

- Tensor method calls construct a deferred dependency graph; `data()` materializes a requested
  result.
- `device()` supports auto selection with WebGPU attempt and CPU fallback, plus explicit CPU,
  WebGPU, and constrained WebGL2 choices.
- The current operations include rank-2 MatMul, limited broadcasting, elementwise operations,
  activations, reductions, softmax, and logits-based softmax cross entropy.
- `Module`, `Sequential`, `Linear`, `Flatten`, `ReLU`, `Dropout`, `CrossEntropyLoss`, `Adam`, and
  `SGD` cover the current example path.
- Every operation is a TypeShade kernel function and every `data()` runs as one `fuse` scope.
  Optimizers (SGD, Adam) update host arrays on every tier. Explicit `Tensor.data()` calls are host reads.
- Tensor graph reverse-mode differentiation supports selected operations. TypeShade
  `shadeGrad`/`shadeGradCheck` are a distinct pure-function differentiation API.
- One active TypeNN Device is supported per JavaScript context due to TypeShade's global call-layer
  configuration.
- On WebGPU, the compiler fuses consecutive elementwise kernel calls of a `data()` into one
  dispatch; WebGL2 and the CPU tier run the calls apart. Device-side Adam, workgroup-tiled
  MatMul, liveness-based buffer reuse, serialization, and WebNN are not implemented.
- The browser demo ships TypeNN runtime JavaScript generated from a recorded source commit. Update
  it with `npm run sync:typenn` after building TypeNN.
- Chromium SwiftShader can validate a software-rendered backend path; those results are not
  physical-GPU measurements. Hardware-specific performance remains unmeasured on this site.

## Information architecture

The default visitor path is Overview → Get started → Execution model or Training → API reference
or Playground → Examples. Shared English and Korean page families provide navigation context,
highlighted TypeScript, source links, and next-step links.

## Brand commitments

- Name TypeNN as the framework and TypeShade as its compiler and runtime foundation.
- Preserve the TypeShade-derived IBM Plex and Ant Design visual language.
- Use plain, factual copy and short sentence-case headings.
- Lead with real code and actual browser execution. Do not fabricate compiler traces or GPU output.

## Evidence

- API example: TypeNN `README.md` and public declarations.
- Runtime behavior: TypeNN `src/runtime`, Tensor, operation, and autograd source.
- Model example: TypeNN `examples/mnist` and the Module training path.
- Browser runtime: `src/vendor/typenn/README.md`, generated from the recorded TypeNN commit.
- Current feature boundary: TypeNN `README.md` and `TODO.md`.
