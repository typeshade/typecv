# BA camera assembly and workgroup specialization

The owner's iPhone record `7b5f61ef` on `f6d8587` lost its WebGPU device while processing the
fifth frame. Before the loss it measured 93 BA operations (84.8 seconds wall time). The
`bundleCameraRows` kernel accounted for 46.0 seconds, 35% of measured GPU time. The last
submitted kernel does not establish the device-loss cause, and this optimization is not a
confirmed fix for Safari's native loss.

## Parallel assembly

The scalar camera-row kernel walks every observation of a camera on one invocation. The
cooperative kernel gives each camera row one workgroup and distributes its observations over
32 or 64 lanes. It reduces six normal-matrix entries, six Schur entries and the RHS using workgroup
memory. It preserves fixed cameras, observation windows, continuation across windows, damping
on the last window and inactive solver state. It adds no resident buffer or numerical readback.

Automatic selection follows the existing `schurReduction` policy: native WebGPU with occupancy
floors, at least 128 observations per camera on average, and a legal dispatch shape. Short
lists and the CPU oracle keep scalar assembly. The existing library-only `schurReduction`
override covers camera assembly as well as the Schur product; the lab adds no setting.

Tests compare raw blocks/RHS before inversion and final BA geometry/costs with scalar assembly.
They include outliers, fixed/local cameras, streamed windows, tails, inactive rows and dispatch
limits. The CPU oracle remains the numerical authority; the reduction order changes and is
checked within stated tolerances.

## TypeShade overrides

The shader declares separate overrides for the scalar row's invocation width and the
cooperative row's reduction width. `scripts/benchmark-ba-workgroups.mjs` uses TypeShade's public
`program.compute(entry, { constants })` to compare actual specialized pipelines, resolved
shapes, cache reuse, scalar CPU results, GPU timestamps and fenced wall time. Compilation and
readback are outside measured samples; variant order rotates between rounds.

This probe supports scalar widths 8/16/32/64/128 and cooperative widths 32/64/128. Each variant
also runs through the production imported-entry call/fuse path, against the same scalar CPU
oracle, with a check of the resolved report shape and a single submission.

Production entry calls now pass `{ constants }` inside their existing device plans, using
TypeShade proposal 0084 ([issue #739](https://github.com/typeshade/typeshade/issues/739),
[accepted proposal #741](https://github.com/typeshade/typeshade/pull/741),
[implementation #743](https://github.com/typeshade/typeshade/pull/743)). The pin is the merged
`d36ac502` revision, including the integration and documentation record in #744. The caller recalculates
scalar dispatch counts for the selected width; cooperative rows still use one group per row.
The owner keeps its fp64 flavor. Selection adds no readback, extra submission or lab setting.

The initial automatic width policy is conservative:

| Path                                | Workload                                                              |       Width |
| ----------------------------------- | --------------------------------------------------------------------- | ----------: |
| CPU or default software scalar      | Any                                                                   |          64 |
| Native scalar with occupancy floors | At most 32 / 64 / 128 camera rows                                     | 8 / 16 / 32 |
| Native scalar with occupancy floors | More than 128 camera rows                                             |          64 |
| Cooperative camera assembly         | At most 256 observations per camera in the current window, on average |          32 |
| Cooperative camera assembly         | Longer observation windows                                            |          64 |

Small scalar systems can occupy several workgroups. A proposed scalar width falls back to 64
if its dispatch count exceeds the device limit. The automatic widths fit WebGPU's minimum
workgroup and shared-memory limits. Width 128 remains in the numerical/measurement probe.
This policy avoids assuming a wider group is always faster; native measurements may refine it.
The Schur-product kernel retains its existing workgroup size.

The probe also exposed [issue #740](https://github.com/typeshade/typeshade/issues/740): fp64
lowering dropped an array's symbolic length, so a scratch declared with default extent 64 remained
64 elements when a 128-lane pipeline was selected. The compiler pin includes
[fix PR #742](https://github.com/typeshade/typeshade/pull/742), and the scratch
now follows the pipeline override. The 128-lane case is part of the numerical gate. The public
TypeShade APIs are used throughout; shader strings and private runtime state are not modified.

Measured on 2026-10-22 at compiler `475d562d`, four cameras (two variable), deterministic
synthetic Jacobian/point blocks, five samples after warmup. Median GPU-pass times in milliseconds:

| Maximum observations/camera | Scalar 8 | Scalar 64 | Cooperative 32 | Cooperative 64 | Cooperative 128 |
| --------------------------: | -------: | --------: | -------------: | -------------: | --------------: |
|                          40 |    0.589 |     0.796 |          1.111 |          3.022 |          10.508 |
|                         131 |    1.643 |     2.610 |          1.514 |          3.118 |          11.978 |
|                        1025 |   11.433 |    20.974 |          6.571 |          7.292 |          14.472 |

Every variant agrees with the scalar CPU oracle within the probe's `2e-10` tolerance. These
software-adapter results show why larger workgroups cannot be assumed faster. They do not
choose a native GPU policy. The default software-adapter solver keeps scalar assembly.

This workspace has a SwiftShader adapter. Its timings characterize that software runner and
do not select an optimal Apple workgroup size. Require a physical adapter with
`TYPECV_HARDWARE_GPU=1`. Full integer BA shader compilation exceeds this software runner's
resources; the isolated integer camera-assembly probe is opt-in with `TYPECV_INTEGER_BA_GPU=1`
and runs automatically with the native-adapter test. That gate compares all eight call/fuse
variants in the integer flavor. Native Apple validation remains pending.

## Adoption validation

On 2026-10-22, the implementation at TypeShade `7a484995` passed the canonical TypeCV build,
type checking and 249 core tests. The final pin `6d418fd1` adds only the upstream proposal's
implementation/validation record; compiler and runtime source are identical.

The software WebGPU gate passed 10 tests, with the native integer-assembly test skipped. It
compares all eight widths through both program pipelines and imported calls, four continuation/
activity modes, raw camera matrices before inversion, final BA costs/geometry, streamed fixed
and shared focal problems, automatic width report shapes, focal search and guided matching.
Every tested imported variant retained one submission and the selected report shape. The CPU
shader checks passed 11 tests covering BA checkpoints, reduction and focal handling.

The full CPU-shader suite and all website browser tests were not rerun for this extension.
Native Apple timing, integer BA and device-loss reproduction remain pending; the small upstream
integer-fp64 entry journey is correctness evidence for the API, not a native BA result.

## Merged compiler integration

The compiler pin advances to `d36ac502` on 2026-10-23, after upstream fixes #742, #743 and the
delivery record #744 merged. This integrates the compiler's object-array layouts and runtime
information alongside the entry override feature. The TypeCV branch also retains main's MVS
reference cache and sparse 320-camera BA checks. Its test commands include both those checks
and the entry-override/focal-search cases.

`ShadeDevice.info` returns the TypeShade owner's frozen build/device information without a GPU
query, including enabled limits/features and whether identity came from the device, its retained
adapter or was unavailable. Occupancy detection uses this snapshot. Opening an externally
supplied device never requests another adapter to choose its fp64 flavor or occupancy policy;
an unavailable identity remains unknown, with the user agent available for fp64 selection.
CPU metadata has no GPU identity, and snapshots remain readable after disposal or device loss.
The lab's preflight adapter query still serves its separate buffer-budget calculation. This
integration adds no automatic transmission or change to the run-record consent flow.

On an available physical GPU, run the numerical gate before measuring:

```sh
npm ci
npm run build
TYPECV_HARDWARE_GPU=1 node --test tests/typeshade/bundle-workgroup-webgpu.test.mjs
TYPECV_HARDWARE_GPU=1 npm run benchmark:ba:workgroups -- --out ba-workgroups-native.json
```

The hardware flag rejects a software adapter. The numerical gate includes the integer-fp64
variants, which are required on Apple hardware. Keep the adapter, timestamp availability and
raw samples in the JSON; compare warmed runs on the same input before changing the automatic
policy. Chromium measurements do not establish Safari behavior. Recheck the original pyramid
video in Safari after deployment, with the default automatic settings, and retain its run record
whether it completes or loses the device. The software checks cannot close that native finding.
