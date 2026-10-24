# KLT work elision and overlapping output reads

This is a performance follow-up to `c66871d9f2e2065e53e42223c9669afc030c5e3e`,
within PR224. It changes execution of the shared KLT kernel and its stateless /
resident-pair output reads. Matching thresholds, cameras, seed budgets, patch
warps, ownership, geometric verification and reconstruction are unchanged.

The subsequent [residency and compact-output measurements](observation-residency-performance.md)
compare against this optimized baseline and record separate pool policies and
remaining tradeoffs. The numbers below remain the historical work-elision phase.

- [x] P1: Audit discarded work and readback scheduling against the compiled baseline.
- [x] P2: Elide unused KLT work with exhaustive-reference and chunk-tail parity.
- [x] P3: Overlap independent output reads and verify failure-path buffer lifetimes.
- [x] P4: Repeated baseline/current RGB measurements with identical observations and geometry.
- [x] P5: Record defaults, measured limits and final validation.

## Observed bottleneck and changes

A compiled-baseline Graham Hall probe, with surface patches and original-view
corroboration enabled and resident reuse disabled, takes 22.312s in expansion,
21.999s inside tracking. Its 88 tracking calls make 176 reads in 176 read waves.
This identifies tracking/execution and synchronization as the dominant measured
stage; caching uploads alone did not remove it in the preceding stage.

The old kernel runs all `maxIterations` residual loops even after a level's
displacement and iteration count stop changing. Lanes outside the requested
chunk duplicate a last point's work despite never writing a result. Backward
lanes whose forward seed already ended also compute patches and discard them.

`kltTrack` has no barriers, workgroup storage, derivatives or cross-lane
dependencies. It now returns outside the active range, copies ended input
coordinates/status with zero NCC, and breaks a level after its converging update.
The final NCC, border/structure/end conditions, actual iteration count and full
forward/backward validation are preserved. Other KLT kernels keep their existing
masking logic. This explicitly supersedes the historical fixed-work restriction
in `docs/klt-tracking.md` for this independent-lane kernel, with numerical tests
replacing the former syntactic exit ban. The f32 reference remains exhaustive.

`trackKltResidentPair` starts independent checked/forward/backward and optional
pyramid reads together through the existing `readTogether` helper. The helper
settles every started read before propagating a failure, so `finally` cannot
recycle scratch or image buffers while a sibling read still owns them. Requested
arrays, read counts, byte counts and returned prefixes remain the same. The batch
adapter still omits only its unused backward output read, not backward tracking.

These changes apply automatically to existing callers of the shared KLT kernel;
there is no quality/performance flag that silently weakens matching. Optional
surface/corroboration/pyramid-cache features retain their existing defaults.

## Validation design

The six-case differential fixture compares positions, NCC, complete status words
and iteration counts with the unchanged exhaustive f32 reference. It includes
one point with 80 iterations available, 67 points in 63+4 chunks, affine predicted
initial positions, an iteration-limited nonconverging case, flat patches and
out-of-frame forward seeds, ended backward records and photometric modes 0/1/2.
CPU comparisons are exact; WebGPU comparisons bound float coordinates/NCC and
require complete status equality. Readback tests compare forced-serial and
concurrent arrays, bytes and pyramid snapshots, then inject one failed read and
one held sibling to prove buffer ownership until all reads settle.

The RGB harness can load a retained compiled baseline via `--gpu-bundle` in a
fresh browser. Baseline/current never coexist in one global shader registry.
It records compiled-bundle hashes, original RGB hashes, serialized measurement
and final-model hashes, selected stage times and read waves. Truth labels remain
outside reconstruction as before. Repeated runs alternate execution order and
report medians; timings exclude browser startup/transport/fixture generation,
and video decode/crop preparation stays separately recorded.

Software WebGPU and physical hardware results must remain separate. No hardware
speedup, lower transfer volume or lower peak memory is inferred from fewer
iterations/readback waves alone.

## Repeated RGB results

The performance baseline is the retained compiled GPU bundle from
`c66871d9f2e2065e53e42223c9669afc030c5e3e`; the optimized shared KLT implementation is
`6e9bd56`. This differs from the original SfM base `e4e5046`. The baseline bundle
SHA is `7ff8149d92f6d2ccea5c901812d6ab6e2855af73ca072c250779edfc7156cd25`, and the optimized bundle SHA is
`78edfc223038d16a7db81b2a15b5bd01c1d61313c60b80693ec7ecff34a342fa`. Both are recorded in the performance
report/validation receipt. `sourceHashes` and `implementationCommit` describe the
measurement workspace; with a bundle override they do not describe that old
compiled bundle's source. The receipt explicitly maps its hash to the old commit.

Three runs per mode alternate baseline/current order, each in a fresh browser.
The six inputs share a device in the same order within a run, so the table does
not claim an independently cold start for each case. All quality options are held
constant: rotation/surface patch warps and multiview corroboration are enabled,
with resident pyramid reuse disabled. Speed ratios divide the two mode medians.

| Input                           | Expansion before → after | Expansion speed ratio | Browser pipeline before → after | Pipeline speed ratio |
| ------------------------------- | ------------------------ | --------------------- | ------------------------------- | -------------------- |
| render-translation              | 3.995s → 1.298s          | 3.08×                 | 5.077s → 2.411s                 | 2.11×                |
| render-translation-omission     | 2.604s → 0.860s          | 3.03×                 | 3.214s → 1.522s                 | 2.11×                |
| render-rotation                 | 2.297s → 0.755s          | 3.04×                 | 2.933s → 1.391s                 | 2.11×                |
| render-low-parallax             | 2.295s → 0.719s          | 3.19×                 | 2.960s → 1.391s                 | 2.13×                |
| graham-hall-cached-rgb-features | 22.004s → 6.545s         | 3.36×                 | 27.053s → 11.620s               | 2.33×                |
| user-video-window               | 8.320s → 1.780s          | 4.67×                 | 10.383s → 3.748s                | 2.77×                |

Graham Hall expansion ranges 21.925–22.365s before and 6.508–6.615s after. The clip
window ranges 8.307–8.398s before and 1.675–1.797s after. These are software WebGPU
measurements (`google`, `swiftshader`, `fallback=true`), with no other local GPU
benchmark running. They do not establish speed on physical GPUs or all videos.
Browser pipeline totals include decode/grayscale, available feature/matching
work, initial SfM, expansion and refinement; they exclude browser startup,
transport, fixture generation and separately recorded ffmpeg preparation. Graham
Hall uses cached RGB-derived keypoints/matches, so their original extraction cost
is excluded. The clip remains the same six-frame manually masked/cropped window
with assumed K, not a full-clip or calibrated-camera benchmark.

Every run has **exactly identical JSON-serialized accepted observations, pairs,
cameras/provenance, baseline/refined finite models and cohort evaluation** to the
first baseline. This is not a claim about raw GPU buffer bit identity (JSON also
normalizes negative zero). Graham Hall remains 2457 landmarks /5694 observations;
the clip remains 468 /1176 and retains all 202 original baseline tracks. Synthetic
rotation and low parallax retain zero finite landmarks, pending 2D tracks and
deferred initialization. No geometry/accuracy gain is inferred from running the
same calculation faster.

Read counts, upload/readback bytes and full Resident snapshot bytes are unchanged
in every case. Logical concurrent-read groups halve: 176→88 on Graham Hall and
66→33 on the clip. These groups are not physical submissions, fences or bus
transactions. All runs end with zero live device bytes and pending reads. Peak
active application device storage remains 8,729,088B in this workload; staging,
driver and total host memory are not measured. No transfer-volume or memory
reduction is claimed.

The unused-work elision and overlapping reads are **enabled automatically** in
the shared KLT path. Existing KLT callers benefit without setting a new option;
matching/quality controls and the optional surface/corroboration/cache defaults
remain unchanged. The new batch features are still not integrated into Lab's
streaming/default path; existing callers that already use KLT share its kernel.

Reproduce by retaining `dist/gpu/index.js` from a separate baseline checkout built
with the same dependency pin/toolchain, then building the current checkout:

```sh
npm run build
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-klt-performance.mjs --baseline-bundle=/path/c66871d-gpu.js --repeats=3 --video=/local/clip.mp4 --output=/path/performance.json
```

Omit `--video` for the five public RGB inputs. `--cases` selects supported subsets,
and `--pyramid-cache=0|1` holds cache policy constant for both builds. Compilation
artifacts, original uploaded video and pixels stay local; the checked-in report
contains numeric counters, hashes and cohort summaries.

Final local checks pass: 329 core tests, 32 affected TypeShade CPU tests and twelve
WebGPU tests (including the four-case RGB regression), format/typecheck, full
build/declarations, TypeShade runtime audit and a 97-page website build. The
validation receipt records the baseline/current bundle provenance, source/report/
log hashes and exact comparison assertions. Remote CI is separate.
