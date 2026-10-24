# Pipeline performance follow-up

Baseline: `88295b4`. This follow-up surveys video decode/preparation, default feature/match/SfM,
Dense MVS and device transfer/lifetime rather than only the opt-in inverse-depth solver.

- [x] W14: Survey the default pipeline and historical real-report stages; freeze baseline runtimes.
- [x] W15: Reduce default descriptor barriers, invalid Dense candidate work and phase-thumbnail preparation.
- [x] W16: Verify outputs, measure paired default-path stages/pipelines and record limits/receipts.

Historical uploaded records (`a4d735ce`, `6c42e7d3`, `1e118cc5`) report pairVerification host elapsed
101.33/51.16/41.35 seconds and 9101/1482/1406 submissions. Guided matching reports 34.35/22.42/31.52
seconds, features 8.88/3.78/4.52, and Dense totals 190.10/140.43/68.63 seconds. These are diagnostic
records from another execution/device, not a new baseline or additive end-to-end timing model.
Host elapsed vs timestamped GPU work suggests scheduling/host/transfer costs also matter.

Independent read-only surveys examined decode/JPEG/worker copies and scene snapshots, batched
pair verification and native BA, and Dense residency. JPEG/worker transport changes and mapper
pair batching affect ownership, session seeds, focal-search reuse and reporting; they remain
separate structural candidates rather than being enabled without behavioral verification.

This round targets three default paths: GPU SIFT Gaussian weights, Dense geometric reference
patch reuse and invalid-window work, and bounded CPU phase-thumbnail preparation. No analysis
resolution, selected features, matching thresholds, RANSAC/BA/MVS iteration budgets or depth gates
are reduced. Baseline GPU public/internal bundle hashes are
`1dfb934adee266fb5fe619ee6ebfc95bb07c66def61414e84c3ec252c7f57a06` /
`3971bbcebf2d3a34d59e4d5c1eaa9a5d52be6626d3eb741d675d792ec5cf2e8a`.

Measurements distinguish native-video extraction, RGB feature/match/native-SfM replay, synthetic
full Dense passes and CPU phase thumbnails. They do not sum overlapping stage timers or claim
worker/UI latency or hardware GPU throughput from software WebGPU. Cold compilation and warm
execution are reported separately. Output comparisons and checks determine which optimizations
are retained; a kernel change that regresses pipeline timing is withheld.


The first proposed SIFT coefficient cache preserved all 11 blur probes and the full video
pipeline output but did not improve warm RGB pipeline latency (26.492s vs 26.187s median;
feature medians 9.376s vs 9.063s). It was removed. The geometric reference-patch cache likewise
preserved full Dense maps/clouds but regressed the complete synthetic pipeline by 7.24%
(27.446s vs 25.594s), so it was removed. Rejected proposal patches/bundles remain private
measurement artifacts; the final default path contains neither cache.

The retained matching change uses 128-channel slabs for GPU descriptors with at least 128
channels and 9536 bytes of workgroup storage; small dimensions/storage, CPU and guided matching
retain their existing kernels. For 128D SIFT, each 8-candidate tile needs two barriers instead of
sixteen. Candidate coverage, channel addition order, top-two/tie rules, strict ratio and invalid
flags are unchanged. A same-device raw differential covers 12 boundary/tie/nonfinite cases in
both directions (2358 output words identical), and the existing high-level matcher passes.

Dense retains only the early exit for unobservable reference windows whose current cost cannot
be improved. Higher/NaN costs preserve the original search. An independent old-body oracle
checks the full state for 32 invalid-window combinations, including fallback cases. No geometric
sample caching remains. Phase preparation retains one shape plan, computes integer cell counts
as axis-population products, and accumulates directly into a newly owned output, preserving luma
addition order and FFT input bits. CPU phase/decision/compiler tests (13), CPU SIFT/matching (7),
GPU raw/high-level matching (2), type checks and core/GPU builds pass before pipeline timing.

Wide rows include one padding element to avoid mapping all rows to the same shared-memory bank on common GPUs. Guided indexing remains unchanged; total workgroup storage is 9536 bytes, below WebGPU minimum limits. This layout also requires output/timing verification; hardware timing remains unmeasured.


## Paired measurements and retained scope

The [pipeline receipt](experiments/general-video/pipeline-performance-results.json) records
source `2f5586a`, baseline `88295b4`, bundle/input/source hashes, all individual samples, cold
calls and output comparisons. Original-video frames at 0/2/4s come from the 3840×2160 H264 source,
identically decoded/scaled to 640×360 for both variants. SIFT capacity stays 4096 and is not
reached: the three frames produce 1878/1800/1820 features. The supplied starting K uses 0.8×width
with shared focal BA, 500 RANSAC iterations and 4/8 BA iterations; automatic focal-search,
keyframe controller, optional KLT/guided/retrieval paths and worker/UI transport are not timed.

Three warm paired runs alternate order; first shared-session calls are recorded separately.
Common compilation can warm the second version, so first-call differences are not evidence of
startup gains. No other build/test/benchmark runs concurrently with the final timing samples.

| Measured interval | Baseline median | Retained median | Reduction |
| --- | ---: | ---: | ---: |
| Original-video RGB SIFT → matching → native SfM | 27.631s | 23.272s | 15.8% |
| Matching within that same interval | 15.616s | 11.595s | 25.7% |
| CPU thumbnail, per frame (9 pairs of 60-frame batches) | 0.922ms | 0.729ms | 20.9% |
| CPU thumbnail + spectrum preparation, per frame | 1.759ms | 1.578ms | 10.3% |
| Complete synthetic Dense, 5 views/160×96/8 photo/2 geo | 26.271s | 26.194s | 0.3% (within variation) |

The video pipeline keeps the same details/descriptor hashes, match hash, complete camera/landmark
model hash, three registered views, 1597 landmarks and 3905 observations in every pair. The Dense
fixture keeps every depth/normal map and point/color/normal byte: 746,944 bytes compared per pair,
6976 points, with equal synthetic surface metrics. Its overall difference is much smaller than
run-to-run variation; **no meaningful overall Dense speedup is established**. The early exit
removes proven redundant invalid-window work, with unchanged search for higher/NaN costs; it
is not presented as a measured broad Dense throughput gain. Lab's 8/4 iteration option remains
unchanged; this synthetic 8/2 workload does not measure the Lab's full configuration.

The source-video native extraction (ffmpeg) is timed separately and is not browser decoding.
The table's nested intervals are not additive and their percentages must not be added. GPU
measurements use Google SwiftShader software WebGPU. They establish subset/fixture behavior,
not whole-video/UI/production hardware throughput or improved reconstruction accuracy. Matching
workgroup storage rises from 2304 to 9536 bytes for the selected wide entry; hardware occupancy
and device-specific timing remain unmeasured. Driver/physical VRAM is not a managed-array counter.

Final checks: 13 CPU phase/gate/compiler tests, seven CPU SIFT/matcher tests, three software-GPU
raw/high-level/guided matcher tests, type/format checks and core/GPU builds pass. Astro reports
zero errors/warnings and four existing hints; the site builds 93 pages. Earlier broad TypeShade,
core and website-unit receipts are separate historical results, not additional reruns here.

The next structural candidates remain batching independent pair verification while retaining
session seeds/focal-search reuse, and separating JPEG/worker/snapshot transport from frame
analysis. They require their own output/ownership and end-to-end measurements. This receipt
makes no speed claim for those paths or for decode, native BA or UI rendering.
