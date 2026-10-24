# Large-run recovery and host memory

Status: 2026-10-25. Baseline: `dd086b4`. Only aggregate observations are recorded here;
original reports, images, depth maps and point clouds remain outside the repository.

## Tasks

- [x] R1 — Diagnose the reported failures and provide a local, read-only existing-tab depth backup helper.
- [x] R2 — Isolate unreadable attachments, split large exports, and bound upload preparation memory.
- [x] R3 — Snapshot and send every component, with explicit partial outcomes and a new-record retry.
- [x] R4 — Remove redundant TSDF sorting and packed-RGBA allocations without changing geometry.
- [ ] R5 — Provide validated streaming reassembly of multipart result files.
- [ ] R6 — Complete focused regressions, build/type checks and document measured limits.
- [ ] R7 — Feed diagnostic compression from owned streams without an intermediate raw Blob.

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
