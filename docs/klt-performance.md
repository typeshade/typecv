# KLT work elision and overlapping output reads

This is a performance follow-up to `c66871d9f2e2065e53e42223c9669afc030c5e3e`,
within PR224. It changes execution of the shared KLT kernel and its stateless /
resident-pair output reads. Matching thresholds, cameras, seed budgets, patch
warps, ownership, geometric verification and reconstruction are unchanged.

- [x] P1: Audit discarded work and readback scheduling against the compiled baseline.
- [x] P2: Elide unused KLT work with exhaustive-reference and chunk-tail parity.
- [x] P3: Overlap independent output reads and verify failure-path buffer lifetimes.
- [ ] P4: Repeated baseline/current RGB measurements with identical observations and geometry.
- [ ] P5: Record defaults, measured limits and final validation.

## Observed bottleneck and changes

A compiled-baseline Graham Hall probe, with surface patches and original-view
corroboration enabled and resident reuse disabled, takes22.312s in expansion,
21.999s inside tracking. Its88 tracking calls make176 reads in176 read waves.
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
one point with80 iterations available,67 points in63+4 chunks, affine predicted
initial positions, an iteration-limited nonconverging case, flat patches and
out-of-frame forward seeds, ended backward records and photometric modes0/1/2.
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
