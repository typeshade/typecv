# Large-run recovery and host memory

Status: 2026-10-25. Baseline: `dd086b4`. Only aggregate observations are recorded here;
original reports, images, depth maps and point clouds remain outside the repository.

## Tasks

- [x] R1 — Diagnose the reported failures and provide a local, read-only existing-tab depth backup helper.
- [x] R2 — Isolate unreadable attachments, split large exports, and bound upload preparation memory.
- [x] R3 — Snapshot and send every component, with explicit partial outcomes and a new-record retry.
- [x] R4 — Remove redundant TSDF sorting and packed-RGBA allocations without changing geometry.
- [x] R5 — Provide validated streaming reassembly of multipart result files.
- [x] R6 — Complete focused regressions, build/type checks and document measured limits.
- [x] R7 — Feed diagnostic compression from owned streams without an intermediate raw Blob.
- [x] R8 — Send saved depth backups directly to the existing collector without chat-sized attachments.
- [x] R9 — Retire privately owned planar source maps after their last reader, with identical completed maps.
- [x] R10 — Remove per-view TSDF staging caches and reuse complete allocation discoveries across boxes.
- [x] R11 — Reduce pair-verification QR copies and verify unchanged models and masks.
- [x] R12 — Preserve per-component host-memory and work measurements in future run records.
- [ ] R13 — Retire owned SfM feature and match data at stage boundaries, including failure paths.
- [ ] R14 — Share camera and point storage when evaluating observations instead of expanding every row.
- [ ] R15 — Skip unused fixed-camera pose derivatives in ordinary and shared-focal bundle adjustment.
- [ ] R16 — Reuse bounded immutable descriptor residency across view-graph matching pairs.
- [ ] R17 — Remove separate visibility-vote reset passes without changing group evidence.
- [ ] R18 — Bound submitted view uploads on GPU paths without completion fences.
- [ ] R19 — Validate cross-stage lifetimes, work counts and representative combined workloads.
- [ ] R20 — Avoid redundant converged PCG dispatches using existing solver checkpoints.

R9–R11 address the expanded memory and runtime request. Comparisons retain the same input
resolution, source selection, iteration counts and geometric thresholds. Each task requires
focused output comparisons and a measured receipt before completion. Synthetic host-array
budgets and software-GPU timings are reported separately from actual browser RSS and a full
15-video rerun.

Future records retain planar map backing-byte peaks, TSDF staging bytes, discovery work,
uploads and synchronous host wall sections separately for each component. Missing measurements
remain absent; multi-component summaries do not sum memory peaks. Adaptive chunk-rate timing
and GPU timestamps are different: every GPU group in the supplied Chimney record has a timestamp.
The runtime can map timestamp results even for plans that skip adaptive rate calibration,
introducing completion waits. GPU time still cannot simply be subtracted from total Fusion
time to label the rest CPU time: queue waits, uploads and host scheduling also contribute.
The new wall sections exclude awaited GPU execution and are not a partition of total time.
The propagation and compatibility regressions passed (67 tests); Astro reported no errors or
warnings, with four existing hints.

## Preserve an existing completed tab

Keep the completed tab open without reloading or starting another run. Chrome Memory Saver
can discard inactive tabs: add the site to its always-active exceptions while recovering.
Keeping a tab open is temporary preservation, not a durable backup.

First save **Run record and sharing → View the record → Download JSON**, then select and
export each component's reconstruction JSON, sparse PLY, Dense PLY and mesh when present.
These exports regenerate their bytes from the retained model, independently of the depth
archive Blobs used by automatic sharing.

To back up the retained raw depth archives without reloading:

1. Review and copy the complete [tab recovery helper](../scripts/typecv-tab-rescue.js).
2. In the completed tab, open Chrome DevTools **Sources → Snippets → New snippet**.
   Paste the helper and run it with **Ctrl+Enter** (Cmd+Enter on macOS).
3. In the added **TypeCV 깊이맵 백업** panel, click **폴더 선택 후 전체 백업** and choose a
   folder with enough free disk space. Files are streamed into a new subfolder sequentially.
4. Keep the resulting `typecv-depth-recovery.json`. It distinguishes successful byte copies,
   unreadable files, missing component manifests, and incomplete original captures.

The helper reads strictly matched retained archive references through React internals, which
are version dependent. It stops on an unsupported or ambiguous layout. It does not reload,
reconstruct, send network requests, clear application references, or overwrite source files.
Folder access requires the user's click and browser permission. Individual browser downloads
are a fallback and cannot be marked verified merely because a download was requested.

`allRetainedFilesSaved` and `sourceArchivesComplete` are separate: copying everything still in
memory does not prove that every expected raw map was captured. Unknown capture scope remains
incomplete. If the original capture is complete, verify both flags and separately keep the
run record and every component's primary exports before treating the tab as expendable.
If the capture already reports missing maps or a capture error, backup cannot recreate those
maps: `sourceArchivesComplete` will remain false. Successful saving of every retained file,
alongside primary exports, then preserves all the information still available in the tab,
while the incomplete original capture remains documented. If Chrome cannot read a Blob's
backing bytes, the helper records that failure; final point clouds cannot recreate all
intermediate maps.

## Confirmed report findings

All three supplied runs used site `e4e5046`, before the current merged implementation.
The Dense angular threshold discussed below is still present at `dd086b4`.

| Run | Registered views | Sparse landmarks | Dense points |
| --- | ---: | ---: | ---: |
| `44bb5d97` | 22 | 7,113 | 564,730 |
| `106d9185` | 41 | 11,897 | 1,439,206 |
| Chimney Rock, 15 videos | 333 | 69,245 | 3,079,603 |

For `44bb5d97`, a manually selected distant-road region loses all its depth hypotheses in
geometric filtering. Three existing sparse landmarks there have maximum visible-camera
triangulation angles of approximately 2.37°, 1.94° and 2.06°. They cannot meet the existing
Dense filter's two-source minimum angle of 3°, even if every camera were considered. This
is a structural rejection of nonzero parallax, not a demonstration that the hypotheses are
correct. Sky pixels around the thin pole also survive geometric filtering; the inspected
sky regions receive no additions from planar completion.

For `106d9185` near 20.237 s, the manually selected distant wall retains 4.61% finite filtered
depth. Replaying the recorded hypotheses and source maps finds sufficient field of view and
mostly sufficient angle, but substantially fewer sources passing the one-pixel round-trip
check. The estimated focal ratio also lies at its search floor, about 0.300006. These results
call for better calibrated hypotheses and source reliability, not bypassing consistency.
Reflective pond pixels already retain finite geometric-filtered depths (44.75% in the
inspected region); planar completion adds none there. Their acceptance does not identify the
true water-surface depth. ROIs are manual and are not ground-truth surface coverage.

The 15-video run registers every retained keyframe. Its components contain 240 views from
11 catalog group-B videos and 93 views from four group-A videos. `notConnected` is empty:
no video is isolated, but the groups have no accepted bridge. The record lacks pair identities
and rejection reasons, so it cannot distinguish missed retrieval candidates from insufficient
verifiable overlap. Matching group names does not justify a forced geometric connection.

## Memory and transmission evidence

The 333 registered inputs are 1280×720. During component 1's planar completion, the existing
code can retain original depth/normal/confidence maps, replacement maps, packed RGBA and
original RGBA for pending components: approximately 7.42 GB decimal (6.91 GiB) of pixel arrays
before archives, point extraction, sparse structures, browser or driver memory. The record's
70,042,312-byte GPU depth working set does not measure these host arrays or Chrome RSS.
An observed 11 GB footprint is therefore plausible from the allocation design; it is not a
required target or a measured attribution of every byte. SSD free-space loss alone cannot
distinguish OS paging from Chrome's own Blob backing files or caches.

The first component has 2,433,283 Dense points. Its normal-bearing PLY payload alone is
62.66 MiB (27 bytes per point), exceeding the existing collector's 50 MiB per-file limit.
The analysis workspace separately limits each uploaded attachment transfer to 32 MiB.
These are different limits. Lossless ordered parts are needed for large primary exports.

At the baseline, one unreadable depth Blob aborts `planFiles()` before the small record JSON
is posted; even oversized files are fully read and hashed before exclusion. Primary exports
also include only the selected component, while depth archives include every component.
The fixes must preserve explicit per-file failures and must not invent an unknown digest.

## Implemented changes and verification

Sharing snapshots all components and regenerates serialized exports one at a time. A failed
producer or Blob read becomes an explicit skipped entry without an invented digest. Large
exports use at most 16 MiB parts with original name, byte offset, total size and a real SHA-256
per part. The existing 50 MiB/file, 4 GiB/run, 1024-file and 256 KiB/record limits remain
explicit. An oversized attachment manifest falls back to the small record with a preparation
failure summary. Partial sends show failures and permit an explicit new-record retry.

Reassemble downloaded parts before opening them as an ordinary result file:

```sh
node scripts/reassemble-result-files.mjs --record run.json --parts downloaded --out recovered
```

The utility streams 256 KiB buffers, verifies each part's size and SHA, checks contiguous
coverage, refuses overwritten outputs and removes failed temporary output. An optional
`--file typecv-dense-c000.ply` selects one original without accepting incomplete data.

MVS now borrows an aligned, little-endian read-only RGBA word view while retaining the
original input. Other layouts use an owning fallback. Component 1 avoids about 885 MB of
packed-RGBA copies. This does not eliminate the much larger original/replacement depth-map
overlap or establish a final browser RSS budget.

Fresh, exclusively owned TSDF output arrays are sorted in place with typed permutation
indices; the public immutable sorter is unchanged. A synthetic 2,433,283-point sort benchmark
used three alternating fresh-process pairs. Median sort time was 2.853 → 2.002 s and median
process maximum RSS was 362 → 206 MiB, with identical complete-output SHA-256 in all six
runs. These measurements cover sorting only, not GPU Fusion, source maps, diagnostics,
driver memory or the user's whole-browser peak. [Aggregate receipt](experiments/general-video/large-run-memory-sort.json)
and `scripts/measure-tsdf-owned-sort.mjs` reproduce the comparison.

The private MVS planar path now schedules references by their remaining source readers and
retires original maps after their last possible reader. Every verification still reads frozen
original evidence, and completed maps and per-view counts retain their original indices. The
public immutable helper and caller-owned map bytes remain unchanged. Fifteen differential
cases passed on CPU and WebGPU, including reordered graphs, shared backing buffers, missing
confidence, cancellation and a case that detects accidental propagation of newly filled holes.

On the recorded 240-view source graph, generated 256×256 maps needed 13 extra original
versions at peak instead of 240. Scaling the unique retained map backing bytes to 1280×720
gives 5,308,416,000 → 2,797,977,600 bytes, a 47.3% reduction for that map term. Three fresh
process pairs had identical output hashes and median maximum RSS of 425,948 → 331,544 KiB.
This is a generated-map lifetime simulation, not a replay of plane fitting or Chrome RSS.
It excludes RGB, external caller references, GPU memory, readback scratch and archives.
[Receipt](experiments/general-video/planar-host-memory.json) and
`scripts/measure-planar-host-memory.mjs` describe and reproduce the scoped comparison.

TSDF now uploads four borrowed channels instead of caching an interleaved copy of every view.
With confidence maps supplied, the 240-view run's former 3,538,944,000-byte host staging term
is eliminated; missing confidence needs only one pixel-sized default channel. The GPU view
storage remains 16 bytes per pixel and the kernels use at most seven storage bindings. A
successful complete global allocation discovery also supplies each box's exact key subset,
including the original interpolation halo, without repeating all input ray walks. Integration
and visibility checks still process every contributing view with unchanged thresholds.

An isolated, warmed, alternating five-pair software-WebGPU benchmark retained all seven
output channels bit for bit. Median Fusion time was 3,646.7 → 2,954.5 ms (19.0% lower);
discovery pixel work fell 91.7%, uploads 40.4% and dispatches 1,113 → 849. The fixture has
24 views at 128×96 and 11 boxes. The receipt also preserves an earlier noisy three-pair window
whose median was 4.0% slower. These measurements do not establish a hardware-GPU speedup or
the actual 15-video runtime. Planar lifetime savings and TSDF staging savings occur in different
phases and must not be added as simultaneous memory reductions.
[Receipt](experiments/general-video/tsdf-pass-cache.json) and
`scripts/measure-tsdf-pass-cache.mjs` reproduce the comparison.

Large pair verification uses an in-place QR factor and 512-point refit windows while preserving
row, pivot and homography design-row order. This path requires at least 1,024 input points,
float WebGPU, unbounded submissions and a requested time budget of at least 250 ms. Smaller
pairs and the CPU, integer, bounded or shorter-budget paths keep the original kernel and
128-point windows. Boundary, fallback, raw factor, model/mask and resource-release checks passed.

The 2,049-point fixture deterministically reduced dispatches from 254 to 170 with identical
terminal results, hypotheses and submissions. Whole-function wall gains varied between repeats;
no dependable percentage is claimed. The receipt retains both timing windows and identifies the
final source and runtime bundle. [Receipt](experiments/general-video/pair-row-inplace.json) and
`scripts/benchmark-pair-rows.mjs` reproduce the comparison.

Diagnostic compression now pulls the already-owned raw chunks directly into gzip instead
of constructing a temporary raw Blob. Tests forbid Blob construction and verify lossless
roundtrip and honest partial capture after a later error. This removes a concrete disk-backed
Blob dependency; it is not proof that every browser storage or out-of-memory failure is fixed.

Focused CPU/website regressions: 109 passed, including unknown hashes, multipart corruption,
bounded regeneration, all-component snapshots, backup failures and archive compression.
The planar→TSDF integration passed on CPU and software WebGPU with unchanged surface and
leak assertions. Core/GPU builds and core type checking passed. Astro reported zero errors,
zero warnings and four existing hints; 103 pages built. Hardware GPU speed and the actual
333-view run's post-fix RSS have not been measured.

TSDF validation also passed five channel/table unit cases, CPU/WebGPU staging and failure
cleanup cases, seven existing WebGPU regressions, three feedback cases and the CPU reference
and quarter-box comparisons. The workgroup-cap regression now uses a smaller fixture that
still spans pixel and voxel chunks; 1,054 chunks obeyed the 256-element cap with all seven
output channels unchanged. Two additional unchanged full-scene CPU MVS stress cases were
stopped after four active CPU minutes; they are not counted as passing. Focused MVS→planar→TSDF
integration passed separately on CPU and WebGPU.

## Received recovery files

All 50 readable saved files (48 gzip archives and two source manifests) were received,
alongside the recovery manifest. Every received size matches its transfer metadata and all
48 gzip archives passed streaming CRC/ISIZE and tar payload-size checks. The saved depth
files total 1,261,490,561 bytes; the recovery manifest adds 12,255 bytes.

Component 1 expected 960 stage records and recorded 246 (240 photometric, three geometric
unfiltered and three geometric filtered), with 714 missing during capture. Component 2
expected 372 and recorded 15 photometric maps, with 357 missing during capture. Both source
manifests record `TypeError: Failed to fetch` despite the MVS computation completing.

The readable physical files contain 235 complete component-1 photometric maps and five
component-2 photometric maps, plus 238 complete RGBA inputs. No complete geometric-filtered
or planar map remains readable in this backup. The 21 recorded maps with missing physical
segments are separate from the 1,071 stage records omitted during original capture. Missing
last archives are `c000-0047` (28,228,414 bytes) and `c001-0001` (24,832,083 bytes); unflushed
chunks can also be discarded when capture fails. This backup cannot establish final per-pixel
surface coverage from positive photometric hypotheses.

## Send a saved backup without the original result tab

`scripts/typecv-send-backup.js` adds a separate consent-based sender to an existing or fresh
TypeCV Lab tab. Run it as a reviewed DevTools Snippet. In an empty Lab tab choose the saved
small `typecv-run-record.json`, then choose the actual backup subfolder containing
`typecv-depth-recovery.json`. This does not run reconstruction. Confirm the displayed
destination, selected saved files, byte count and missing files; only the explicit consent
checkbox and Send button begin requests.

It uses the already-deployed collector's legacy healthy-file contract and existing POST/PUT
routes, hashes and uploads one file at a time, and never routes data through chat. It preserves
the original recovery manifest and capture omissions as record metadata. It supports receipt
JSON, failed-file retry under the same ID and a manual new ID for remaining files after the
server's one-hour upload window. The original backup is retained. This legacy helper sends
saved depth archives only, not oversized primary point clouds or reconstruction exports;
those use the new multipart application sender. No live upload was performed by automated
tests, and this run's files were already received through individual chat attachments.
