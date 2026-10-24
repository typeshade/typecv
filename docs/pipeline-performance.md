# Pipeline performance follow-up

Baseline: `88295b4`. This follow-up surveys video decode/preparation, default feature/match/SfM,
Dense MVS and device transfer/lifetime rather than only the opt-in inverse-depth solver.

- [x] W14: Survey the default pipeline and historical real-report stages; freeze baseline runtimes.
- [ ] W15: Remove repeated SIFT blur-weight, Dense reference-window and phase-thumbnail preparation work.
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
