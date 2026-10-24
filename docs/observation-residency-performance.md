# Observation residency and compact batch results

This follow-up compares against `946c83138ffc6882ddecc03c86813beb5ed173e1`, after
the shared KLT work-elision/read-overlap improvements. It preserves the matching,
camera, geometry and observation-admission algorithms.

- [x] R1: Audit remaining GPU work, repeated image preparation, pooling and shared-memory ownership.
- [x] R2: Reuse two resident images and compatible buffers without starving other queued jobs.
- [x] R3: Read checked coordinates/status and NCC in one compact snapshot, preserving validation.
- [ ] R4: Compare repeated RGB runs with zero/production pooling and separate timestamp profiles.
- [ ] R5: Record default decisions, measured limits and final verification.

## Findings and design

The prior speedup did not remove the remaining work. A separate timestamp-enabled
Graham Hall baseline profile measures about4.045s in KLT tracking compute passes,
0.648s in downsampling,0.960s in image gradients and0.007s in final checking. These
are software WebGPU compute-pass timings, not the ordinary wall-time benchmark.
Enabling timestamps adds staging mapping/waits and affects chunk scheduling; it
must not be used to claim a production speedup or interpret wall-minus-GPU time
as CPU time. Untimestamped headline comparisons keep this instrumentation off.

The first two-frame prototype nearly halved image uploads while changing expansion
time by roughly9% in one photo probe. Upload reduction alone does not establish
large speedups. The current implementation also reuses compatible evicted buffers
and removes an unnecessary batch-result snapshot.

### Resident image ownership

The batch option is `pyramidCache:'pair'` for explicit two-frame reuse,
`undefined`/`false` for the existing stateless default and `true` for the existing three-frame cap. The
internal cache still supports its previous bounded capacities and budget eviction.
Keeping two images retains the same number of full-image pyramids as one stateless
pair. Original grayscale must remain immutable until the call settles.

A compatible LRU slot changes its logical frame identity and writes the new
padded gray image into its existing allocation. All pyramid levels and gradients
are rebuilt before tracking. An unchanged frame keeps its already computed image
evidence. Point ownership/admission order never changes. Tracked `writeF32` keeps
uploads visible in application counters and rejects a wrong owner, kind or size;
buffer reuse does not mean a new image had zero upload cost.

Default caching must cooperate with other work using the same device. The existing
working-budget check could reject another queued job while an idle cache retained
reclaimable images. A small internal reclaimer hook lets that check release idle
cache storage before rejecting allocation. It protects the active pair through
all awaits, checks actual allocator progress and unregisters at disposal. This is
the concrete reason for extending budget admission rather than merely retaining
buffers longer. It does not change physical-VRAM accounting or evict caller-owned
arrays. Invalid preparation drops dirty buffers; all paths retain explicit cleanup.

The pinned TypeShade runtime validates bindings using `state.view()`. An unmaterialized
`ZeroedState` makes a fresh full-size host array for each validation. Retaining a
gradient across many calls exposed this cost: the first production-pool prototype
regressed Graham Hall expansion from7.001s to8.060s (three-run medians). The final
cache retains a host mirror once per gradient lease. It explicitly counts the initial
zero upload rather than presenting it as free. A short batch may upload more than
the stateless path, and persistent host storage increases. This is a local workaround
using the existing runtime; no dependency or TypeShade source is changed.

### Compact results

The batch adapter needs checked records plus forward NCC, rather than the whole
forward array. `compactOutput:true` makes `kltCheck` copy NCC into a tail after its
four-word checked records. One5N-float snapshot returns checked4N and NCCN host
views; the GPU still executes forward/backward tracking and checking. No predicted
coordinate or synthesized NCC is introduced.

Standalone KLT defaults are unchanged. Explicit compact mode has an empty
`forward` and an `ncc` array; backward and debug-pyramid read options keep their
meaning. Empty batches return empty arrays. Scratch admission includes the extra
NCC word, and failure waits for every started sibling read before recycling.
With affine tails, actual snapshots can fall from48N to20N bytes; logical consumed
prefixes fall from32N to20N. Read calls fall from two to one per batch pair. Logical
read waves were already one per pair after the preceding overlap improvement.

The NCC output offset uses a check-only uniform. The iterative tracker's control
keeps its prior48-byte layout and unchanged function body. Splitting this uniform
did not establish a speed gain: dynamic uniform alignment and the additional check
binding prevent a byte-size argument from proving lower driver cost.

## Default decision

Compact batch output is automatic because it removes an unused snapshot while
preserving every value consumed by admission. A compact-only probe did not establish
a wall-time gain; the transfer/count reduction is the supported claim. Standalone
tracking still returns legacy output unless explicitly asked for compact results.

Two-frame residency remains opt-in. Although ordinary photo/video batches improve,
the initial automatic-cache experiment regressed small omission, rotation and
low-parallax inputs by13–20% in repeated measurements. Restoring the original
tracking uniform did not resolve that regression. No image-size or frame-count
heuristic is inferred from these few cases. The previous shared KLT work-elision and
overlapped-read improvements remain automatic. Surface patches and independent
corroboration retain their prior opt-in defaults. Batch expansion is not newly wired
into Lab streaming.

## Validation plan

Existing KLT, camera/geometry and RGB regressions remain in place. New cases compare
default/off/pair/legacy-three policies, exact accepted observations and original
indices, repeated calls, pool0/32MiB, a two-pyramid budget, unrelated caller buffers,
shared-device pressure, active-pair protection, writes, dirty-plan recovery and
disposal. Compact tests compare every consumed value with legacy output, including
affine/chunk tails, end states, debug pyramids, empty batches and failed reads.

The performance wrapper alternates fresh-browser baseline/current runs. It holds
matching settings and RGB evidence constant and requires exact serialized
accepted-observation/model/cohort equality. It records pool policy, active versus
pooled storage and complete post-disposal cleanup. The production32MiB pool is
measured explicitly; allocation benefits in pool0 must not be generalized to it.
Separate timestamp profiles record supported/requested/granted status, dispatch
counts and timing coverage. Missing timestamps are unavailable evidence, not zero
GPU cost. Physical hardware remains outside the current software environment.
