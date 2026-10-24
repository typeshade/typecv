# KLT execution plans and source-cache screening

This stage compares against `1af3a5868d5762b36e193f366685c5e6bd638519`, after
the shared work-elision/read-overlap and compact-output changes. The retained
baseline GPU bundle has SHA256
`75c2b44fb281052449e8e3c77eeffcc3b7d141ae046739ec39145cf4516dcae8`.

- [x] T1: Audit repeated template work and existing numeric/runtime contracts.
- [x] T2: Prototype source reuse and reject candidates without an observed benefit.
- [x] T3: Add opt-in grouped plans and raw-output, default-policy and budget regressions.
- [ ] T4: Measure repeated ordinary RGB workloads and separate GPU profiles.
- [ ] T5: Record the default decision, limits, CI correction and final verification.

## Findings and chosen scope

`src/gpu/klt.shade.ts::kltTrack` already computes the source Hessian and
mean/variance once per pyramid level. It samples fixed source intensities and
gradients again in residual iterations. We tested per-invocation intensity
storage, intensity/gradient storage and a bounded GPU resident source prepass.
All preserved accepted measurements and models in screening, but none established
ordinary photo/video acceleration. The resident candidate also increased active
allocator storage. These candidates are absent from production sources.

Grouping the existing three entries reduced plan calls but initially shrank
128-point work into 64-point chunks: the new rate key measures both directions
and the check, rather than one direction. In the software backend this reduced
the number of simultaneously dispatched workgroups. The opt-in grouped path
starts with two 64-lane workgroups when enough points exist. A single-entry
roundtrip shader and a larger minimum in the legacy path also failed screening
and were removed. The original numerical shader is unchanged.

These are single-repeat screening results; they do not establish universal
regressions, physical register spills or a hardware conclusion. Source-level read
estimates did not predict wall time. The machine exposes 5 affinity CPUs and a
4-core cgroup quota; its WebGPU adapter is Google SwiftShader. Physical GPU
performance remains unmeasured.

## Opt-in execution plans

`KltGpuOptions.batchPasses`, `KltTrackerOptions.batchPasses` and observation
expansion's matching option default to false. Explicit true records the forward,
backward and check entries for the same point range in one chunk plan. The
original separate execution remains the default. There is no new resident buffer,
private patch array, numerical kernel, camera assumption or matching threshold.

The existing TypeShade planner keeps each compute entry as a separate GPU pass.
This is not one fused shader invocation. Actual dispatch counts may stay equal
under fixed chunks, while runtime plan/submission counts fall. Neither counter is
a physical fence or bus-traffic measurement.

Each point owns its output rows, and the affine inverse tail is disjoint from
forward output prefixes. Therefore per-range ordering preserves the original
forward→backward→check dependence. Every started output read still settles before
buffer reuse. Shared image caches keep their existing ownership rules.

Grouped adaptive work uses the separate `kltRoundtrip` rate key: one unit is a
point's two directions and check. Its default minimum is 128 points; legacy
`kltTrack` keeps 64. Explicit `minChunkTracks` and forced `chunkTracks` are honored.
The public logical chunk report remains forward sizes followed by backward sizes.

## Verification and comparison contract

Six focused inputs cover odd image dimensions, fractional positions, pyramid
border clamping, windows1/10/11/31, affine patches, initial targets and all three
photometric modes. They compare all forward/backward/checked values against the
unchanged exhaustive f32 reference and exact same-backend opt-out output. They
also verify real plan counts, unchanged fixed-chunk entry/dispatch counts, omitted
option matching explicit false, exact-original-budget execution and cleanup.

Performance runs use the same production 32 MiB pool, stateless image policy and
geometry/patch/corroboration settings in both modes. `--batch-passes=1` requests
this opt-in schedule; the old compiled baseline has no such option and runs its
original schedule. Actual execution is recorded in the plan/kernel counters.
Three-run comparisons alternate fresh browsers and require exact accepted
measurement/model/cohort equality. Later cases share a device and are not
independently cold. Timestamp profiles stay separate.

The browser totals exclude startup, transport and fixture generation. The photo
case uses RGB-derived cached features/matches; their original cost is excluded.
The uploaded-video window has six frames, a manual crop/background mask and
assumed K. This is not a full gameplay-clip reconstruction or a new accuracy result.

The remote baseline CI's only failing WebGPU case expected an incomplete CPU-stage
list. `gpu/mapper.ts` intentionally reports calibrated rotation hypotheses, using
existing host feature coordinates without extra GPU readback. The test now checks
the exact documented list while retaining its numerical/local-BA assertions.

[Screening data and provenance](experiments/general-video/klt-roundtrip-screening.json)
record candidate/source/bundle hashes and local raw-report hashes. Experimental
workspace commits are explicitly distinguished from committed production source.
Final repeated measurements and validation are added after execution completes.
