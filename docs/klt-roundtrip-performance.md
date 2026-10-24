# KLT execution plans and source-cache screening

This stage compares against `1af3a5868d5762b36e193f366685c5e6bd638519`, after
the shared work-elision/read-overlap and compact-output changes. The retained
baseline GPU bundle has SHA256
`75c2b44fb281052449e8e3c77eeffcc3b7d141ae046739ec39145cf4516dcae8`.

- [x] T1: Audit repeated template work and existing numeric/runtime contracts.
- [x] T2: Prototype source reuse and reject candidates without an observed benefit.
- [x] T3: Add opt-in grouped plans and raw-output, default-policy and budget regressions.
- [x] T4: Measure repeated ordinary RGB workloads and separate GPU profiles.
- [x] T5: Record the default decision, limits, CI correction and final verification.

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
border clamping, half-window radii 1/10/11/31, affine patches, initial targets and all three
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


## Repeated measurements and default decision

The production candidate is commit `5e325aed3f50fbe35d8d193f2d802baae627a1ac`;
its compiled GPU bundle has SHA256
`a235471052986c3a1874840c9384f4c7dd11ff06dedcf8de0576a1166e065cfd`.
The baseline/current modes each have three runs, for 36 case results. Every result
passes the exact accepted-measurement, finite-model and cohort comparison.

| Input | Expansion baseline → grouped median | Expansion time change | Browser pipeline time change |
| --- | --- | --- | --- |
| Rendered translation | 1.362 → 1.214 s | -10.9% | -4.0% |
| Rendered translation, detector omissions | 0.856 → 0.812 s | -5.1% | -4.4% |
| Rendered pure rotation | 0.780 → 0.774 s | -0.8% | -5.4% |
| Rendered low parallax | 0.714 → 0.754 s | +5.6% | -4.6% |
| Graham Hall, 12 cached-feature RGB photos | 6.641 → 6.577 s | -1.0% | +1.2% |
| Uploaded video, six-frame window | 1.707 → 1.769 s | +3.6% | +4.4% |

Negative changes mean less time. These are ratios of mode medians, not a mean
of paired ratios or a statistical/hardware claim. The photo's expansion ranges
overlap (baseline 6.583–6.693 s, grouped 6.453–6.879 s); the video's grouped
expansion range is slower in these three samples (1.679–1.709 s versus
1.739–1.885 s). The full ranges and individual stages remain in the
[repeated report](experiments/general-video/klt-roundtrip-performance.json).

Photo plan calls/native reported submissions fall 352→176; video 132→66.
The same numerical track and check entries still run. Upload/readback counters
and application allocator peaks stay identical: peak active storage is
8,729,600 bytes; final pooled storage is 21,986,176 bytes. Active buffers and
pending reads are already zero before disposal; disposal also releases pooled
storage. Host heap, driver storage and physical bus traffic are not measured.
Fewer submissions did not produce a broad wall-time benefit.

`batchPasses` therefore remains experimental and **false by default**. There is
no automatic input-size heuristic. Explicit opt-in is useful for evaluating this
schedule on a workload/backend; it is not a recommendation for arbitrary videos.
The earlier default work elision and opt-in image residency results are separate
comparisons and cannot be multiplied by this stage's ratios.

## Next work supported by these findings

The cache candidates reduced repeated source reads in source code but did not
reduce measured ordinary-input time. Another larger per-lane patch array is not
a justified next default. A more substantial candidate is to parallelize pixels
inside a patch using multiple lanes, with bounded shared storage and cooperative
reductions, rather than keeping an entire patch in one invocation's private
array. That requires confirming TypeShade's workgroup execution contract and
maintaining the existing CPU path. Different reduction orders must be evaluated
against raw coordinates/statuses and fixed-cohort geometry, not just throughput.
This design is not implemented by the grouped host plans.

Before promoting such a path, run it on an identified physical GPU as well as
SwiftShader, including cold/warm pipeline costs, upload/readback, cleanup and
end-to-end timers. Source-level operation counts and software-only screening
cannot decide a hardware default. Automatic masks/cuts, full-clip reconstruction
and dynamic-object shape are outside this performance stage.

## Separate profiles and final verification

Separate timestamp-enabled photo/video profiles match the repeated accepted
measurements, finite models and cohort evidence. All dispatched groups are
timestamped. Tracking still accounts for 72.0%/71.4% of the grouped path's timed
device work; baseline shares are 71.9%/73.0%. In the isolated photo-first profile,
track dispatches are 178 in both modes and plan calls fall 354→177 (including
warmup); video track dispatches remain 66 and plan calls fall 132→66. Timestamps
change waits, rate adaptation and warmth; these device durations are not mixed
into the headline browser medians.

[Baseline profile](experiments/general-video/klt-roundtrip-profile-baseline.json),
[grouped profile](experiments/general-video/klt-roundtrip-profile-current.json)
and the [validation receipt](experiments/general-video/klt-roundtrip-validation.json)
record controls, source/bundle/report/log hashes and timer boundaries.

Final local checks pass: build, typecheck, format, 45 affected TypeShade CPU tests,
15 affected WebGPU tests, the formerly failing CI local-subsets target, fresh
TypeShade runtime audit and a 101-page site build. The source and tests were
frozen during these runs; only a JSDoc wording change followed numerical tests.
The numerical shader and exhaustive f32 reference are byte-for-byte unchanged
from `1af3a58`. Remote CI is separate and must be checked on the final PR head before
merging. T2 records a completed screening investigation, not adoption of the
rejected patch caches.
