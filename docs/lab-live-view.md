# The lab's live view: what changes, what works, and when the worker last reported

Status: accepted, 2026-10-21, by the owner's go-ahead in the session. Step 1 is implemented;
preparation feedback and step 2's multi-video stage view are implemented as of 2026-10-23.
Steps 3 and 4 remain proposals.

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
live view is on only while a run is busy (`live` of `SfmViewer`). The viewing camera can ease
between selected registered views while browsing: position, quaternion rotation and field of
view interpolate over 300 ms, with a new selection starting from the current displayed pose.
This is a viewing transition; it does not interpolate missing reconstruction poses or alter
exports. Manual navigation, disabling follow, fitting, presets and placement changes cancel an
unfinished move. Reduced motion makes the viewing move immediate.

## 3. Step 1: the change, the pulse, the frame and the badge (implemented)

Sections 3.1 to 3.8 are in the page and the viewer. Section 3.9 adds library callbacks and worker messages for the dense fusion.

### 3.1 What a new map changed

`landmarkChanges` (`website/src/lib/live-view.ts`) compares the new map's landmarks with the
previous map's by `trackId`. Each landmark is `Added`, `Moved` (farther than 10⁻⁴ of the map's
extent), `Kept` or `Skipped` (a coordinate is not finite). The tracks the new map dropped come
back as `removed`. `addedViews` does the same for views by `id`.

`SfmViewer` keeps the previous map's points, frusta and camera centers, and gives each drawn
vertex two attributes: `aFrom` (its place in the previous map) and `aBorn` (when it appeared,
or long ago). A shader patch (`liveMaterial`) on the sparse points and the camera lines:

- moves each vertex from `aFrom` to its place over `TWEEN_S` (0.8 s) after a map with a move;
- lights a vertex born less than `GLOW_S` (1.6 s) ago, and grows a new point from nothing to its
  size over `TWEEN_S`;
- brightens the pulsing cameras (3.2).

The owner's rule of 2026-10-21 for every point of the live view: an added point grows from small
to its size, a dropped point shrinks to nothing, and an updated point moves to its new place.

The dropped points are drawn in red and shrink to nothing over `GHOST_S` (1.2 s), fading in the
last third (the owner's request of 2026-10-21: a point does not just vanish).

Map deliveries can be only 500 ms apart, shorter than the 800 ms growth and movement. The drawing
state in `website/src/lib/live-vertices.ts` therefore keeps `aBorn`, `aFrom` and `aStarted` by
identity. An unchanged target keeps its unfinished transition; a changed target starts from the
position currently drawn, rather than the previous target. Sparse tracks, camera vertices,
linked features and preview blocks have independent clocks. Deletions sample the current
position and size, and overlapping deletion batches keep their original shrink clocks.

Fused dense chunks also carry birth times (one extra float per drawn point). Appending a chunk,
growing its buffers or rebuilding its budgeted subset preserves earlier chunks' birth times.
Sparse points shrink while the first fused points grow. The final map completes its transition
even when React receives the result and the end of the run in the same render. Saved results
still draw immediately; reduced motion disables movement and size changes, retaining the glow.

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

The video-first layout of 2026-10-22 (§6) keeps a 16:9 3D view and equal-height cards from
1100 px wide. The video has a larger share than the original narrow sidebar, and the cards grow
with their contents. Run settings live above the cards; frame statistics and camera-motion
details can be expanded. The frame card has no internal scrolling region. Wide 3D, full screen
and narrower layouts keep their respective rules.

### 3.8 Verification

- `website/live-view.test.mjs`: the change classes, the dropped tracks, the added views, the
  pulse targets and the badge text.
- `website/live-vertices.test.mjs`: overlapping births, interrupted moves, reordered points,
  deletion clocks, buffer reuse and reduced motion.
- `website/live-view-browser.test.mjs`: the real React/three.js viewer in Chromium, including
  shader compilation, dense arrivals, depth preview updates and the final map, with and without
  reduced motion. Run with `node --test website/live-view-browser.test.mjs`; use
  `TYPESHADE_CHROMIUM` to select an installed Chromium.
- Owed: a browser run of the lab on the eight-frame sample that reads `data-added` and
  `data-pulse` during the run. Not run in the pull request's container; see its record.

### 3.9 Dense fusion feedback (2026-10-22)

The owner's running version was `d8a4782`. Its default TSDF fusion reported nothing between
the last geometric depth preview and the complete sorted point cloud. A long fusion looked idle.

`fuseTsdfGpu` now reports allocation, assignment, integration, free-space checking, surface
counting, extraction, readback and sorting. `reconstructDenseGpu.onFusionProgress` translates the
packed view index to the original scene view id. The worker posts `denseActivity`, throttling
repeated chunk updates to 200 ms while retaining phase changes and operation completion. The
page draws the phase, region, camera, elapsed time and current-operation progress over the 3D
view. The camera being used pulses; volume operations pulse the cameras together. Cleanup and
meshing post their phase before they start and show an indeterminate bar and elapsed time.

The dense panel stays mounted from the first dense phase until the run ends, including waits
between multi-video components. It names and explains the current operation, highlights it in
the depth/fusion/free-space/surface/readback/finish strip, and advances its local phase clock
even when no worker event arrives. The strip marks only the current step: regions and readbacks
can repeat. Depth-reference progress is also forwarded to this panel. Changing cameras in the
same phase and region preserves the phase clock.

`TsdfProgress.phaseDone` and `phaseTotal` aggregate submitted work across cameras in the current
region; free-space totals also include each camera group's vote reset. The displayed bar uses
these counts instead of resetting for every camera. Surface counting and extraction retain their
chunk progress. Readback, sorting, cleanup and meshing have no measurable intermediate fraction,
so their bar remains indeterminate while the explanation, extracted point count and elapsed time
remain visible. Extracted box previews remain in the map during final processing.

The percentages describe submitted units of the current operation, not a time estimate for the
whole run. Untimed plans can still be in flight when submission returns. Readback is shown as
its own operation; no extra numerical readback or completion fence is introduced for feedback.

The top run track occupies the first row of the status area and remains the same element through
preparation, analysis, final refinement, dense processing and completion. Unknown-duration phases
use a moving segment that never leaves the track; reduced motion holds that segment still.
Changing a stage updates its label and measurement instead of removing and recreating the bar.

Each extracted TSDF box also reaches `onFusionPreview`, as independent arrays in world units.
`denseFusionPoints` draws those temporary points while sorting and later boxes finish. They
never enter the export or final point counts. The final `densePoints` replaces them, and
completion, cancellation or failure clears temporary activity. The same messages are routed by
component for multi-video runs. The normal `onView` events retain their previous contract.

Verification: `tsdf-feedback.test.mjs` and the feedback cases of `tsdf-webgpu.test.mjs` prove
identical final arrays with callbacks enabled and disabled, unchanged submission counts after
warm-up, independently transferable previews, cancellation cleanup, original camera ids and
world coordinates. `dense-run.test.mjs` checks routing and throttling. The real Lab browser
test, `dense-activity-browser.test.mjs`, delivers worker events through the normal page handler,
checks desktop/mobile feedback, temporary-to-final replacement and a final-only PLY export.

### 3.10 GPU preparation feedback (2026-10-23)

Before frames are analysed, the Lab shows the current preparation phase and elapsed time:
the 3D view, worker startup, analysis runtime loading, adapter lookup, device connection and
session setup. Multi-video runs also show the matching vocabulary loading. The two worker
paths post `preparing` with the request id before entering each preparation operation; the
page reports viewer and worker startup locally. A reused runtime or device can pass its phase
immediately. These are preparation boundaries, not shader-compilation counts or a time estimate.
Kernel compilation on first use continues to be reported by the existing analysis progress.

The indicator has no percentage while the duration is unknown. It updates elapsed time even
without a worker message and clears on readiness, cancellation or error. A multi-video stage
ticker cannot cover the preparation
display with an unstarted frame-stage message. The same compact step list wraps on mobile,
uses the current theme and stops its spinner with reduced motion.

Verification: `website/gpu-preparation-browser.test.mjs` exercises both real worker paths with
a controlled numerical runtime, checks that preparation precedes runtime loading and device
opening, and drives the real Lab through phases, elapsed time, errors, Stop, stale events and
readiness in English/desktop and Korean/mobile/dark/reduced-motion configurations.

## 4. Multi-video stages and remaining steps

**Step 2 (implemented 2026-10-23).** The 3D reconstruction viewport shows input videos as
selectable thumbnail nodes during preparation and the frame, candidate, matching, verification
and component stages. The active input, recent accepted keyframes, counts and stage time update
from actual messages. The graph positions are schematic UI layout; they create no camera pose
or point. Candidate, descriptor-matched and geometry-accepted connections have separate styles.
Only accepted pairs add verified links and inlier counts. Selecting an input shows its recent
keyframes. Disconnected inputs remain visible and component membership arrives before mapping.

The final `graph` message previously arrived only after mapping. `ViewGraphOptions.onActivity`
now reports independent host-side candidate/work/result snapshots from `buildViewGraph`,
including expansion rounds. It adds no matching, verification, submission or readback. Worker
accounting aggregates frame pairs by video pair and posts cumulative `graphActivity` snapshots
at most every 200 ms within a stage; stage changes and the component result flush all completed
counts. Display accounting retains at most 256 video links. The UI draws at most 16 inputs and
keeps at most four recent thumbnail URLs per input and 64 in total, with an explicit subset note
when links or inputs are limited. These display limits do not limit the reconstruction.
End-of-video thinning removes dropped thumbnails from the preview.

The same renderer stays mounted under the stage view. Once a real partial map has views, the
graph contracts to a stage badge, leaving the actual cameras and landmarks visible. An icon
opens the graph for inspection and closes it back to the map. Dense MVS retains its existing
depth/fusion feedback. Readiness, cancellation and failure end their overlays, and another run
clears prior graph/thumbnail state. Reduced motion disables both the spinner and edge flow;
the stage view also retains the existing no-response warning.

Verification: the graph differential test compares candidates, accepted edges, component
membership and statistics with callbacks absent/present, including mutation of callback
payloads. Worker integration checks that a complete connection graph arrives before the first
map and that throttled counts match the final inlier matrix. Bounded display accounting and
thumbnails have unit coverage. The real Lab browser check covers phase styles, selection,
recent frames, mobile/dark/reduced motion and graph-to-map navigation.

**Step 3.** The SfM worker posts the cameras after each iteration of bundle adjustment _(new
message)_, so the map moves while it is refined, not only after. This changes the worker and
its message protocol, and the cost of each post must be measured against Rule L.3.

**Step 4.** The mapper reports the 2D–3D correspondences that PnP weighs and its inlier mask
_(new event of `src/reconstruction/mapper.ts` and `scene-mapper.ts`)_, so the 3D view shows them
before a camera registers, and shows them for a frame that fails to register. This changes the
library's API and follows `docs/design-rules.md`.

## 5. Browsing a completed video run

The selected timeline frame drives the selected camera immediately. Its image loads separately;
while it is loading, the previous image and its marks are hidden. A late image from an earlier
selection cannot replace the current one. Uploaded and catalog source videos both re-decode the
selected timestamp; prepared frames keep their original pixels.

Native video seeking and playback select the nearest analysed frame without pausing playback or
seeking it back to the sampled time. Manual frame selection and trajectory replay seek the source
video to that frame. Manual navigation stops trajectory replay. The **Follow selected camera**
switch beside the timeline also moves the 3D viewpoint; frames without a registered pose clear the
camera selection and photo plane.

For multiple videos, each reconstructed component has a video selector and a timeline of that
video's registered cameras. Camera picking selects the corresponding video, time, image and
thumbnail. Switching components starts at its first video and frame. Source loading events cannot
reset the selected frame to the beginning of the new video. Motion arrows and readouts only use
adjacent registered cameras from the same video. Camera picking uses the transformed positions
shown on screen, including the map's axis conversion and placement.

`website/frame-navigation-browser.test.mjs` exercises the real Lab, video decoder and WebGL viewer
with deterministic SfM worker results: URL sources, uploads, native seeking and playback, missing
poses, follow mode, multiple videos and components, camera picking and delayed image delivery.

## 6. Video-first workspace layout

The input, run settings and main run action share one bar above the media. Settings open in a
popover, close with Escape or when a run starts, and keep the resolution, focal controls, dense
options, per-video input rows and benchmark together. Run status and progress sit below that bar.
The multi-video stage list can be expanded there. Record consent, notes and sharing are in a
separate disclosure below the workspace; changing their position does not change consent.

At desktop widths, the video and the larger 3D pane sit side by side with their media aligned.
The video is wide enough to inspect and neither pane has an internal scrolling region. On
smaller screens, the video comes first and the map follows. Wide 3D and full-screen views keep
their existing behavior. Frame statistics and camera-motion details are expandable, while the
actual frame-stage feedback and primary scene counts remain visible.

A source video uses its own playback controls and seek bar. It has no duplicate frame slider or
Previous / Next / Replay buttons. Focusing the video or frame panel enables Left / Right to step
through analysed frames; text fields, sliders and 3D navigation keep their own keys. Inputs with
no original video retain the frame slider, Previous / Next and frame replay. Empty panes explain
what will appear there, and image-loading feedback overlays the media without shifting controls.
