# Observation residency and compact batch results

This follow-up compares against `946c83138ffc6882ddecc03c86813beb5ed173e1`, after
the shared KLT work-elision/read-overlap improvements. It preserves the matching,
camera, geometry and observation-admission algorithms.

- [x] R1: Audit remaining GPU work, repeated image preparation, pooling and shared-memory ownership.
- [x] R2: Reuse two resident images and compatible buffers without starving other queued jobs.
- [x] R3: Read checked coordinates/status and NCC in one compact snapshot, preserving validation.
- [x] R4: Compare repeated RGB runs with zero/production pooling and separate timestamp profiles.
- [x] R5: Record default decisions, measured limits and final verification.

## Findings and design

The prior speedup did not remove the remaining work. A separate timestamp-enabled
Graham Hall baseline profile measures about4.385s in KLT tracking compute passes,
0.776s in downsampling,0.971s in image gradients and0.007s in final checking. These
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

Resident caching must cooperate with other work using the same device. The existing
working-budget check could reject another queued job while an idle cache retained
reclaimable images. A small internal reclaimer hook lets that check release idle
cache storage before rejecting allocation. It protects the active pair through
all awaits, checks actual allocator progress and unregisters at disposal. This is
the concrete reason for extending budget admission rather than merely retaining
buffers longer. It does not change physical-VRAM accounting or evict caller-owned
arrays. Invalid preparation drops dirty buffers; all paths retain explicit cleanup.

The pinned TypeShade runtime's
[`checkEntryCall`](https://github.com/typeshade/typeshade/blob/d36ac5027356b96e87a9ad900dfd8e6ea4ab55dc/src/core/host-compute.ts#L183)
validates bindings using `state.view()`. Its
[`ZeroedState.view()`](https://github.com/typeshade/typeshade/blob/d36ac5027356b96e87a9ad900dfd8e6ea4ab55dc/src/core/resident.ts#L259)
implementation makes a fresh full-size host array for each validation until a host
value is retained. Retaining a
gradient across many calls exposed this cost: the first production-pool prototype
regressed Graham Hall expansion from7.001s to8.060s (three-run medians). The final
cache retains a host mirror once per gradient lease. It explicitly counts the initial
zero upload rather than presenting it as free. A short batch may upload more than
the stateless path. The cache retains host mirrors; total host-heap and driver peaks
were not measured. This is a local workaround
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

## Final default-path measurements

The final source is `46ba9915367f5f263e495efcf3d68bcd63ff527f` (bundle
`75c2b44fb281052449e8e3c77eeffcc3b7d141ae046739ec39145cf4516dcae8`).
The compiled baseline is `946c83138ffc6882ddecc03c86813beb5ed173e1` (bundle
`78edfc223038d16a7db81b2a15b5bd01c1d61313c60b80693ec7ecff34a342fa`).
Both use the pinned TypeShade runtime. Report source hashes describe the measurement
workspace in both modes; they do not describe the overridden baseline bundle.

The zero-pool run compares the old stateless path with the new default, which also
remains stateless and uses compact results. Values are seconds, medians of three
fresh-browser runs per mode, with alternating order. These modest timing differences
do not establish a wall-time gain. Min/max and individual runs remain in the JSON.
Both comparisons deliberately keep geometry patch warps, surface proposals and
original-view corroboration enabled to exercise the same advanced workload. Those
algorithm options remain opt-in: “default path” here describes the cache/output
policy, not every API option or the Lab streaming pipeline.

| Input                           | Expansion before → after | Browser pipeline before → after |
| ------------------------------- | -----------------------: | ------------------------------: |
| render-translation              |            1.344 → 1.291 |                   2.433 → 2.510 |
| render-translation-omission     |            0.824 → 0.818 |                   1.504 → 1.499 |
| render-rotation                 |            0.780 → 0.762 |                   1.448 → 1.404 |
| render-low-parallax             |            0.730 → 0.724 |                   1.414 → 1.369 |
| graham-hall-cached-rgb-features |            6.493 → 6.640 |                 11.587 → 11.782 |
| user-video-window               |            1.791 → 1.787 |                   3.919 → 3.787 |

All six cases in all six runs retain exact serialized measurement/model/cohort
equality. This is not raw GPU-buffer bit equality or a new geometry-accuracy result.
Rotation and low parallax retain pending depth. Each browser runs cases in fixed
order on one device, so later cases are not independently cold.

Browser totals exclude startup, transport and synthetic fixture generation.
Video ffmpeg preparation is separate. Graham Hall uses its original RGB-derived
cached feature/matcher results; those costs are excluded. The video is a six-frame
window with a manual background mask/crop and assumed K, not a full-clip result.

## Selected resident path

The final production-pool run uses the same32MiB pool in both modes. The baseline
is stateless; current mode explicitly selects two-frame reuse. All other controls
and RGB remain equal. This is a cache/compact comparison within one pool policy,
not a comparison of pool0 against32MiB.

| Input                           | Expansion before → after (s) | Time change | Browser pipeline before → after (s) |
| ------------------------------- | ---------------------------: | ----------: | ----------------------------------: |
| render-translation              |                1.300 → 1.207 |       -7.2% |                       2.427 → 2.390 |
| render-translation-omission     |                0.808 → 0.912 |      +12.8% |                       1.483 → 1.565 |
| render-rotation                 |                0.756 → 0.941 |      +24.4% |                       1.406 → 1.577 |
| render-low-parallax             |                0.732 → 0.801 |       +9.5% |                       1.360 → 1.437 |
| graham-hall-cached-rgb-features |                6.689 → 5.640 |      -15.7% |                     11.629 → 10.778 |
| user-video-window               |                1.725 → 1.546 |      -10.4% |                       3.730 → 3.510 |

All six cases in all six runs have exact measurement/model/cohort parity. Small
omission/rotation/low-parallax cases regress9.5–24.4%, supporting the decision to
leave residency opt-in. The ordinary photo and clip improvements are15.7% and10.4%
in expansion; browser totals improve7.3% and5.9%. These are three-run medians on
SwiftShader, not a hardware or general-input guarantee. Per-run ranges are in the
JSON reports. The explicit resident path is:

```ts
await expandObservationsGpu(device, frames, pairs, { pyramidCache: 'pair' });
```

For Graham Hall, expansion upload counters fall255,936,000→138,259,200B (46.0%),
and the clip falls60,689,920→38,657,920B (36.3%). These include the initial gradient
mirror zero uploads. Graham Hall builds176→91 image pyramids using two logical
pyramid leases and178 retained buffer reuses; the clip builds66→38 with72 reuses.
Logical leases/reuses are not counts of native GPUBuffer creation.

With affine tails, Graham Hall result snapshots fall145,920→60,800B and clip
snapshots61,152→25,480B (58.3%). Logical consumed prefixes fall37.5%; read calls
halve176→88 and66→33. Read waves remain88 and33. These are application/Resident
counters, not physical bus bytes or fence counts.

Active application device peak is8,729,088→8,729,600B (+512), while the production
pool live peak, including pooled buffers, is21,981,472→21,986,176B (+4,704). Two full-image pyramid slots remain
the same image bound; the compact NCC tail adds one word per point. This does not
establish lower peak GPU or host memory. The two Graham Hall gradient host mirrors
hold5,811,200B. All measured modes release active buffers and pending reads; disposal
also clears pooled storage.

Reproduce with the baseline compiled from946c831 and the pinned dependency:

```sh
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-klt-performance.mjs \
  --baseline-bundle=/local/residency-946c831-gpu.js \
  --baseline-pyramid-cache=0 --pyramid-cache=2 --pool-bytes=33554432 \
  --repeats=3 --video=/local/input.mp4 --output=/local/pair-results.json
```

For the default-path comparison, use `--pyramid-cache=auto --pool-bytes=0`.
The original uploaded clip is not committed. Reports record RGB/frame hashes,
calibration, manual crop/mask, software adapter, source/bundle hashes, case order,
all stage times and allocator counters. Timestamp profiles remain separate.

## Verification receipt

The final source passes44 affected TypeShade CPU tests and14 WebGPU tests, including
four RGB reconstruction cases. Core329-test evidence comes from the earlier phase;
core sources did not change during this follow-up. Full formatting, type checking,
GPU/core/declaration build, TypeShade runtime audit and99-page site build pass. The receipt records
exact source/report/log hashes and separates local results from remote CI.

New regression entry points are `klt-compact.test.mjs`,
`klt-compact-webgpu.test.mjs`, `klt-cache-pressure.test.mjs`,
`observation-residency.test.mjs` and `observation-residency-webgpu.test.mjs` under
`tests/typeshade`. They cover consumed values, retained mirrors, two-image budgets,
shared-device pressure, dirty preparation, held reads and complete cleanup.

- [Default policy, pool0](experiments/general-video/observation-residency-pool0.json)
- [Explicit pair policy, production pool](experiments/general-video/observation-residency-pool33554432.json)
- [Separate baseline profile](experiments/general-video/observation-residency-profile-baseline.json)
- [Separate current profile](experiments/general-video/observation-residency-profile-current.json)
- [Validation and provenance](experiments/general-video/observation-residency-validation.json)

## Next measured targets

Final separate timestamp profiles cover every reported expansion group and keep
pool0 in both modes. They compare the stateless baseline with explicit pair reuse.
Seconds below sum compute-pass timestamps, not ordinary wall time:

| Input       | Downsample before → after | Gradients before → after | Tracking before → after | Check before → after |
| ----------- | ------------------------: | -----------------------: | ----------------------: | -------------------: |
| Graham Hall |             0.776 → 0.362 |            0.971 → 0.466 |           4.385 → 4.196 |        0.007 → 0.006 |
| Clip window |             0.208 → 0.088 |            0.244 → 0.124 |           0.962 → 0.998 |        0.002 → 0.002 |

The reduced pyramid work explains the main measured device improvement. The
tracking formula has not changed in this phase; these single instrumented runs do
not establish a tracking-kernel speedup. Tracking still occupies83.4%/82.3% of
current timed device expansion work. Missing timestamps in ordinary runs mean
unavailable GPU timing, not zero cost; wall-minus-GPU is not a CPU-time estimate.

- KLT patch tracking remains the largest device cost. Precomputing reusable source
  patch/gradient terms is a candidate; it needs numerical, iteration/status and
  memory-admission comparisons before a speed claim.
- TypeShade binding validation could use resident type/shape metadata without
  materializing a zero array. This needs an upstream runtime change and binding
  regressions; it could remove the local host-mirror/initial-upload workaround.
- Whole-pipeline work remains outside expansion, including initial geometry and
  accepted-map refinement. Profile these before choosing a CPU/TypeShade change.
- Lab integration requires a bounded original-image evidence archive and stable
  frame/pixel identities. The batch API does not by itself improve Lab streaming.
- Validate on physical GPUs and more RGB clips before promoting residency to a
  default. Software WebGPU timings do not establish hardware acceleration.
