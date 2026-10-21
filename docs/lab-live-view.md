# The lab's live view: what changes, what works, and when the worker last reported

Status: accepted, 2026-10-21, by the owner's go-ahead in the session. Step 1 is implemented in
the pull request that adds this document. Steps 2 and 3 are not started.

Every path and symbol named here exists on the branch of that pull request. A name marked
_(new)_ does not exist yet.

## 1. Problem

The owner's report of 2026-10-21: during a run the progress line moves, but the 3D view stays
still, and a visitor cannot tell what the run is doing. The code on `main` at `8d06a38` shows
three causes.

1. The 3D view changes only when a camera registers. The SfM worker posts a map
   (`snapshot`) from `onReconstruction`, at most every `snapshotIntervalMs` (500 ms). Feature
   extraction, matching, bundle adjustment, the final refinement and the first kernel compiles
   post no map.
2. A new map replaces the old one at once. Nothing tells the new points, the dropped points and
   the cameras that bundle adjustment moved apart from the rest (change blindness).
3. The viewer renders on demand (`SfmViewer.tsx`, "one frame, not a loop"). With no new map it
   draws nothing new, and the progress events (one every 100 ms or more) change only the status
   text.

A multi-video run shows the same thing for longer: no map exists before the reconstruction
stage, so the 3D view is empty for the frame, candidate, matching and verification stages.

## 2. Rules

**Rule L.1.** The live view shows only what the worker reported. It adds no pose, point or
correspondence the run did not produce. A tween between two reported places is drawing, not
data, and it ends at the reported place. (`website/PRODUCT.md`: missing poses are never
interpolated.)

**Rule L.2.** One thing moves at a time where possible, briefly, and the rest of the scene stays
as it was. A visitor reads a change, not a stream.

**Rule L.3.** The live view takes little from the run. A pulse alone redraws at most every
`PULSE_FRAME_MS` (50 ms). A change animates for at most `GHOST_S` (1.2 s) after a map.

**Rule L.4.** With `prefers-reduced-motion: reduce`, nothing slides and nothing pulses. New
points still glow, since that is a change of color, not of place.

**Rule L.5.** A result that is opened, replayed or browsed after a run is drawn as it is. The
live view is on only while a run is busy (`live` of `SfmViewer`).

## 3. Step 1: the change, the pulse, the frame and the badge (implemented)

All of step 1 is in the page and the viewer. The workers do not change.

### 3.1 What a new map changed

`landmarkChanges` (`website/src/lib/live-view.ts`) compares the new map's landmarks with the
previous map's by `trackId`. Each landmark is `Added`, `Moved` (farther than 10⁻⁴ of the map's
extent), `Kept` or `Skipped` (a coordinate is not finite). The tracks the new map dropped come
back as `removed`. `addedViews` does the same for views by `id`.

`SfmViewer` keeps the previous map's points, frusta and camera centers, and gives each drawn
vertex two attributes: `aFrom` (its place in the previous map) and `aBorn` (when it appeared,
or long ago). A shader patch (`liveMaterial`) on the sparse points and the camera lines:

- moves each vertex from `aFrom` to its place over `TWEEN_S` (0.45 s) after a map with a move;
- lights a vertex born less than `GLOW_S` (1.6 s) ago, and draws a new point larger;
- brightens the pulsing cameras (3.2).

The dropped points are drawn in red and fade out over `GHOST_S` (1.2 s).

The viewer reports the counts of the last map on its element: `data-added`, `data-moved`,
`data-removed` and `data-new-views`, all 0 while `live` is off.

### 3.2 The pulse

The page keeps the stage that the worker reported last and the time it came (`activity`).
`pulseTarget` maps a progress detail to the cameras its work touches:

| Stage of the progress detail                                     | Target                  |
| ---------------------------------------------------------------- | ----------------------- |
| `bundle`, `final`                                                | every camera of the map |
| `matching`, `verifyPair`, `pnp`, `triangulation`, `observations` | the newest camera       |
| any other stage, such as `features`                              | none                    |

A multi-video run pulses the newest camera during its `reconstruction` stage and nothing
before it. The pulse runs while the last progress event is younger than `PULSE_HOLD_MS`
(1.5 s), so it stops when the work stops. The viewer reports it as `data-pulse`.

### 3.3 The frame being analysed

While a run is busy, `FrameInset` draws the frame the worker analyses in the lower left corner
of the 3D view, with its detections and candidate matches as they arrive (`paintOverlay`, the
drawing of the frame card). The map panel then moves with every frame, even when no camera is
added.

### 3.4 The freshness badge

`LiveBadge` says how long ago the worker last posted any message, and redraws every second. Past
`QUIET_MS` (5 s) it says how long the GPU has worked without a message. A long bundle adjustment
is work, and the badge says so rather than looking stalled.

### 3.5 Verification

- `website/live-view.test.mjs`: the change classes, the dropped tracks, the added views, the
  pulse targets and the badge text.
- Owed: a browser run of the lab on the eight-frame sample that reads `data-added` and
  `data-pulse` during the run. Not run in the pull request's container; see its record.

## 4. Steps 2 and 3 (not started)

**Step 2.** The multi-video stages before the reconstruction get a picture: the keyframes of
each video as a row of thumbnails, and the graph of the verified pairs as it grows. The data is
in the `graph` and `multiStage` messages that the page receives now.

**Step 3.** The SfM worker posts the cameras after each iteration of bundle adjustment _(new
message)_, so the map moves while it is refined, not only after. This changes the worker and
its message protocol, and the cost of each post must be measured against Rule L.3.
