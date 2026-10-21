# The lab's live view: what changes, what works, and when the worker last reported

Status: accepted, 2026-10-21, by the owner's go-ahead in the session. Step 1 (3.1 to 3.7) is implemented in
the pull request that adds this document. Steps 2 to 4 are not started.

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

- moves each vertex from `aFrom` to its place over `TWEEN_S` (0.65 s) after a map with a move;
- lights a vertex born less than `GLOW_S` (1.6 s) ago, and grows a new point from nothing to its
  size over `TWEEN_S`;
- brightens the pulsing cameras (3.2).

The owner's rule of 2026-10-21 for every point of the live view: an added point grows from small
to its size, a dropped point shrinks to nothing, and an updated point moves to its new place.

The dropped points are drawn in red and shrink to nothing over `GHOST_S` (1.2 s), fading in the
last third (the owner's request of 2026-10-21: a point does not just vanish).

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

While a run is busy, `FrameInset` draws the frame the worker analyses in the upper right corner
of the 3D view, with its detections and candidate matches as they arrive (`paintOverlay`, the
drawing of the frame card). The map panel then moves with every frame, even when no camera is
added.

### 3.4 The freshness badge

`LiveBadge` says how long ago the worker last posted any message, and redraws every second. Past
`QUIET_MS` (5 s) it says how long the GPU has worked without a message. A long bundle adjustment
is work, and the badge says so rather than looking stalled.

### 3.5 The 2D–3D links of a camera, in the 3D view

The owner's request of 2026-10-21: show in three.js which 3D points the 2D features of a frame
are tied to. Each landmark of a map holds its observations (`viewId`, `featureId`, `pixel`), and
an observation is a 2D–3D link that the mapper accepted: a PnP inlier, a continued track or a
triangulated point. `viewLinks` (`website/src/lib/live-view.ts`) collects the links of one view.

For the selected camera, `SfmViewer` draws, on the plane of its photo:

- every feature the worker detected in the frame, as a small gray point, when the page holds
  them (`features`; a single-video run keeps them for each analysed frame);
- each linked feature as a point in the depth color of the frame card (red near, blue far).

A linked mark keeps its previous place by track, and slides to its new place when a map moves
its camera. No line joins a mark to its 3D point: the owner judged the fans of lines hard to
read (2026-10-21).

During a single-video run the selected camera is the frame being analysed, so the links of a
frame appear the moment its camera registers. A frame without a pose has no plane in the map
and shows no links (Rule L.1). The links that PnP is weighing before it accepts a pose are not
in a map; showing them needs a mapper event and is step 4.

The Display menu's **2D–3D links** switch turns the marks off. The viewer reports the count
drawn as `data-links`.

### 3.6 Points grow in place

The owner first asked for new points to fly from the camera's image plane to their places, with a
short trail (2026-10-21), and then withdrew it the same day. A new point grows from small to its
size in its place (3.1). The flights, their trails and `flightOrigin` are removed.

### 3.7 One height for the run's two cards

The owner's reports of 2026-10-21: the frame card and the 3D card differed in height, and a run
did not fit one monitor. On `main` the 3D view was `calc(100vh - 150px)` tall, so the 3D card
alone outgrew the screen, and the frame card ended at another height. A first fix gave both
cards the viewport's height; the owner then found the 3D card too tall and chose a 16:9 view.

From 1100 px wide (`website/src/styles/global.css`, "One height for the run's two cards"):

- the 3D view is 16:9 at its card's width (500 px tall at 1440 × 900 and wider, since the page
  is at most 1440 px wide), and the 3D card is as tall as its content (811 px at 1440 × 900);
- under the view, the map's state and the camera motion share one row (`sfm-map-notes`), and the
  statistics sit in one row of eight;
- the frame card takes no height of its own (`contain: size`): it fills the 3D card's row and
  scrolls inside itself.

The wide layout, full screen and the narrower layouts keep their rules.

### 3.8 Verification

- `website/live-view.test.mjs`: the change classes, the dropped tracks, the added views, the
  pulse targets and the badge text.
- Owed: a browser run of the lab on the eight-frame sample that reads `data-added` and
  `data-pulse` during the run. Not run in the pull request's container; see its record.

## 4. Steps 2 to 4 (not started)

**Step 2.** The multi-video stages before the reconstruction get a picture: the keyframes of
each video as a row of thumbnails, and the graph of the verified pairs as it grows. The data is
in the `graph` and `multiStage` messages that the page receives now.

**Step 3.** The SfM worker posts the cameras after each iteration of bundle adjustment _(new
message)_, so the map moves while it is refined, not only after. This changes the worker and
its message protocol, and the cost of each post must be measured against Rule L.3.

**Step 4.** The mapper reports the 2D–3D correspondences that PnP weighs and its inlier mask
_(new event of `src/reconstruction/mapper.ts` and `scene-mapper.ts`)_, so the 3D view shows them
before a camera registers, and shows them for a frame that fails to register. This changes the
library's API and follows `docs/design-rules.md`.
