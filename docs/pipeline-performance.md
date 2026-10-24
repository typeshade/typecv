# Pipeline performance follow-up

Baseline: `88295b4`. This follow-up surveys video decode/preparation, default feature/match/SfM,
Dense MVS and device transfer/lifetime rather than only the opt-in inverse-depth solver.

- [x] W14: Survey the default pipeline and historical real-report stages; freeze baseline runtimes.
- [x] W15: Reduce default descriptor barriers, invalid Dense candidate work and phase-thumbnail preparation.
- [ ] W16: Verify outputs, measure paired default-path stages/pipelines and record limits/receipts.

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
