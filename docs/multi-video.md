# Reconstructing one scene from several videos: design

Status: accepted (2026-10-20, under the owner's approval of the work in the conversation).

This document is written for an engineer or an agent who has not read the conversation that led
to it. It names files, symbols, defaults, gates and the numbers that accept each step. A name
marked _(new)_ does not exist yet. Every other path and symbol named here exists on `main` at
`ebc39b6`. The owner approved the work on 2026-10-20 and asked how a randomly ordered set of
videos is handled. This record is the answer. It changes no code.

Labels used below. **Fact**: read in the code or in a record on `main`. **Inference**: reasoned,
not measured. **Decision**: a default this record chooses. **Open**: a question a step answers.

## 1. Problem and goal

### 1.1 What the video path does today

The lab (`/lab/`) takes one video. The page decodes it at 4 frames per second
(`VIDEO_SAMPLE_FPS`, `website/src/lib/video-sampling.ts`). The worker
(`website/src/workers/sfm.worker.ts`, `processFrame`) then handles each sampled frame:

1. The phase gate (`PhaseGate`, `website/src/lib/keyframe-gate.ts`) keeps or skips the frame
   before any SIFT work. It compares the frame with the last keyframe of the same video.
2. A keyframe gets GPU SIFT features (`extractSiftGpuResident`).
3. The keyframe is matched with the earlier keyframes that the session still holds
   (`matchFloatDescriptorsGpu`, ratio 0.8, mutual). The last keyframe is matched first.
4. The pairs go to one `IncrementalSfmSession` (`src/gpu/mapper.ts`) through `session.add`.

Fact: the worker keeps at most a few keyframes resident (`registerFrame` keeps the reference
frames and the last three). The session keeps a window of 32 views (`windowSize` 32). It drops
older views and the edges of dropped views.

### 1.2 Why this fails for several videos

Fact: a new frame is matched only with recent frames. Those frames are the frames of the same
video, because the worker holds one job with one clock. Three consequences follow.

1. Frames of video 2 are never matched with frames of video 1. No cross-video edge exists, so the
   mapper cannot join the two videos in one map.
2. The result depends on the order. The first frame of the first video is the first view. The
   session evicts old views as new ones arrive. The seed pair and the order of registration
   follow the arrival order.
3. An unrelated video cannot be told from a related one. Its frames build their own map or fail
   to initialize, and the page shows the failure as an error of the run.

The photo path is the opposite. A photo is matched with every active earlier photo, whatever the
order. That is exhaustive matching, N(N−1)/2 pairs, and it holds only for small N (the window of
32 bounds it). It does not scale to hundreds of keyframes from tens of videos.

### 1.3 Goal

Given n videos of one object (an orbit, a vertical ascent, a lateral pass, in any order and
possibly mixed with unrelated videos), produce one sparse map for every set of videos that see
the same scene. Report every video that joins no map as "not connected".

The goal has four parts:

- **G1. Order independence.** The result depends on the content of the videos. The order inside a
  video counts. The order of the videos does not.
- **G2. Scale.** Matching cost grows with N·k, not with N². N is the number of keyframes. k is
  about 10.
- **G3. Separation.** Unrelated videos become separate components. They do not break the map of
  the related videos.
- **G4. Honest failure.** When two videos see the same object from views too far apart, the
  lab says that they are not connected. It does not guess.

Non-goals: dense MVS over several components (the dense path keeps its present limits and runs
on one component at a time), a learned retrieval network, loop closure inside one video as a new
feature (the global descriptor gives it for free, section 3.3), rolling-shutter modeling,
and video synchronization (the videos have no shared clock).

## 2. Input

### 2.1 Videos and samples

The lab's upload control accepts several files (section 7.1). A lab sample set (section 7.2) adds
several remote samples at once. Each input is one **video** with:

- **`videoId`**: the content hash of the file (2.4). It is the name of the video in the run record.
  Two uploads with the same hash are one video. The lab shows one "duplicate" tag and drops the
  second.
- **a label**: the file name or the sample id. Labels never decide anything.
- **a camera**: the intrinsics of this video (2.3).
- **its sampled frames**: decoded at `VIDEO_SAMPLE_FPS` (4), as today.

### 2.2 Per-video sampling and keyframe gate

Decision: each video has its own `PhaseGate`. The gate sees only the frames of its own video, in
time order. A keyframe decision therefore does not depend on the other videos or on their order.

Fact: the phase gate is deterministic for a given decoded frame sequence. So the keyframe set of a
video is the same in any input order of the videos. The shuffle gate (7.1) tests this.

The gate keeps its present options (`PHASE_GATE_OPTIONS`: `minShift` 0.04, `minGapSeconds` 0.2,
`maxGapSeconds` 1). Open: a slow video can yield many near-duplicate keyframes. A cap
`maxKeyframesPerVideo` (default 60, Decision) bounds N. The cap keeps the first and last frame and
thins the others evenly in time. The cap is an option, and the run record names it.

### 2.3 Per-video intrinsics

Different videos can come from different cameras. Fact: the lab today takes one assumed focal
ratio for the whole run (`DEFAULT_VIDEO_FOCAL`, `focalRatio`) and zero distortion. The worker builds
each frame's camera from it (`frameCamera`).

Decision: the focal ratio is a property of the video, not of the run.

1. The default of each video is the run's focal ratio. The sample catalog may give a calibrated
   camera for a sample (`frame.camera`), as it does today.
2. The lab shows the focal ratio of each video as one editable field in the video's row.
3. A pair is verified with the two cameras of its two videos (`verifyPair` takes both cameras).
   Pairs inside a video and across videos use the same call.
4. The final bundle adjustment refines one focal scale for each video (a camera group), not one
   for the whole run. Fact: `refineSharedFocal` refines one scale that all views share
   (`src/reconstruction/bundle-adjustment.ts`). A per-group form is new work (step 5).

Inference: the Pexels clips of the test set (section 9) come from a drone, possibly from one
camera. If the cameras are equal, the per-video focal groups add parameters and no accuracy. Step 7
measures both the per-video and the shared form on the test set.

### 2.4 Content identity

Decision: every identifier that decides anything comes from content. Input order, file names and
the arrival order of decoded frames decide nothing.

| Identifier | Definition                                                                                                                                      |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `videoId`  | SHA-256 of the file's size (8 bytes), its first 1 MiB and its last 1 MiB, cut to 16 hex digits. Cheap for files of hundreds of megabytes.       |
| `frameKey` | SHA-256 of the analysis frame's RGBA bytes (the pixels the worker receives), cut to 16 hex digits. Equal frames in two videos get equal keys.   |
| `viewId`   | The rank of `(frameKey, videoId, time)` in the sorted list of all keyframes of the run, from 0. The sort is by the three fields, in that order. |

The mapper's integer ids (`ImageFeatures.id`, `PairMatches.first` and `second`) are the `viewId`s.
Fact: the mapper breaks ties by id. `verifyPairs` sorts pairs by id and seeds each pair with
`seed + pi`, where `pi` is the index in the sorted list. The next view is chosen by the number of
2D–3D correspondences, then by the lower id (`incrementalSteps`). With canonical ids, every one of
those ties breaks by content.

Two changes in the mapper follow (step 2):

1. The RANSAC seed of a pair comes from the two ids (a fixed mix of `seed`, `first` and `second`),
   not from the pair's index in the list. The index changes when a pair is added or dropped.
2. A test shuffles the ids of a synthetic input and expects the same registered set and the same
   cameras, after the shuffle is undone (7.1).

Fact: `frameKey` depends on the decoder. Two browsers can decode a frame to bytes that differ in
the last bit. So `viewId` is canonical inside one run on one decoder, not across devices. The shuffle
gate runs on one device and one build. Open: whether to hash a 32 × 32 gray thumbnail with 4-bit
quantization instead, so that two decoders agree. Step 2 measures how often two decodes of the
same video give different keys.

## 3. Order independence: the view graph

### 3.1 Overview

The view graph is a graph whose nodes are keyframes. An edge joins two keyframes that a geometric
check has accepted. The weight of an edge is its number of inliers. The graph is built from
content only, in four stages:

1. **Features.** SIFT features of every keyframe (as today), kept for the matching stage (3.5).
2. **Global descriptor.** One compact vector for each keyframe (3.2).
3. **Candidates.** For each keyframe, the k most similar keyframes (3.3), and its neighbours in
   time inside its own video.
4. **Verification.** SIFT matching and geometric verification on the candidate pairs only (3.4).

The cost is O(N·k) matchings. Exhaustive matching is O(N²). For N = 375 and k = 10 that is about
4,900 pairs against 70,125 (section 10).

### 3.2 The global descriptor

**Decision: a bag of visual words over RootSIFT, with a fixed vocabulary of 512 words and tf-idf
weights.** The steps for one keyframe:

1. Take the RootSIFT descriptors of the keyframe. Fact: the GPU SIFT already produces RootSIFT with
   `keys.rootSift` (`src/gpu/sift.shade.ts`).
2. Assign each descriptor to the nearest of 512 visual words (L2, ties to the lower word index).
   This is a new kernel, `globalAssign` _(new)_. It takes descriptors and the vocabulary, and writes
   one word index for each descriptor.
3. Count the words in a histogram of 512 bins. Take the square root of each count (a damping,
   so that repeated structure does not dominate), then multiply each bin by its idf weight.
4. L2-normalize. The result is the global descriptor, 512 floats.
5. The similarity of two keyframes is the dot product of their descriptors (cosine).

The vocabulary is a file of 512 × 128 floats (256 KiB) trained offline. It is trained by k-means
(Lloyd, fixed iteration count, a fixed seed) on the RootSIFT descriptors of the lab's samples. A
version number names it (`rootsift-512-v1`). The idf weights come from the keyframes of the run:
`idf(w) = log((N + 1) / (df(w) + 1)) + 1`, where `df(w)` is the number of keyframes that contain
word `w`. The weights depend on the set of keyframes and not on their order.

Why a fixed vocabulary and not one trained on each run: the descriptor of a keyframe then depends
on that keyframe alone (plus the idf scale). It does not depend on the other videos or on the
order in which they arrive. A run can also add a video to a finished run without retraining.

Why a bag of words and not the alternatives:

| Candidate                                    | Dimension | For                                                                                     | Against                                                                                                         |
| -------------------------------------------- | --------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Bag of words, 512 words, tf-idf (**chosen**) | 512       | Invariant to the viewpoint change that SIFT survives. Small. Standard in SfM retrieval. | Needs a vocabulary file. Quantization loses detail, but only candidates need it, not the geometry.              |
| VLAD with 32 words (aggregated residuals)    | 4,096     | Keeps more detail per word.                                                             | Eight times larger. Needs PCA or intra-normalization to work well. Same vocabulary problem.                     |
| Downsampled image (for example 16 × 16 gray) | 256       | No vocabulary. Trivial kernel.                                                          | Not invariant to rotation, scale or a viewpoint change. Two videos of one object from two paths look different. |
| Global learned descriptor (NetVLAD and kin)  | 4,096     | Strong on place recognition.                                                            | A neural network and its weights. Out of scope (section 1.3).                                                   |

Decision method: measure, then confirm. Step 3 measures all three non-learned candidates on the
test set (section 9) and on the lab's photo samples. The measure is recall at k = 10 of the **true
pairs**. A true pair is a pair of keyframes whose exhaustive SIFT matching and verification gives
at least `minEdgeInliers` inliers (3.4). The bag of words stays the default when its recall at 10
is at least 0.9 of the recall of the best candidate. Otherwise the record is amended with the
measured winner. The downsampled image stays in the code as the baseline of the measurement only.

Inference, not measured: the image descriptor fails across paths. An orbit and a vertical ascent
show the same object under different in-plane rotation and scale, and a thumbnail of each is a
different picture.

Inference: the cost of `globalAssign` is the number of descriptors times 512 times 128
multiply-adds. For 375 keyframes of 4,096 features that is 1.5 million descriptors and about
100 GFLOP. That is below a second on a physical GPU and tens of seconds on SwiftShader.

Parity: a CPU reference of the whole descriptor (`globalDescriptorReference` _(new)_, in
`src/gpu/`, plain TypeScript) is the oracle of the kernel. Section 7.4 states the gate.

### 3.3 Candidate selection

For each keyframe `i` the candidate set is the union of:

- **Temporal neighbours.** The previous 3 keyframes and the next 3 keyframes of the same video.
  This keeps each video connected inside itself, as today's path does.
- **Retrieved neighbours.** The `k` = 10 keyframes with the highest cosine similarity to `i`,
  excluding keyframes of the same video within 10 keyframes in time. The ones further away in the
  same video stay: they are loop-closure candidates, for an orbit that returns to its start.

Then the pairs are made unique: (i, j) and (j, i) are one pair. A pair of two keyframes with the
same `frameKey` is dropped (a duplicate).

Order independence: the similarity of `i` and `j` does not depend on the order, and the sort
of equal similarities breaks ties by lower `viewId` (canonical, 2.4). So the candidate set is the
same in every input order.

Two guards against a missed link:

1. **Per-video reserve.** For each pair of videos (u, v), the best 3 candidate pairs between them
   are always kept, even when they rank below 10 for both keyframes. This makes sure that a video
   whose frames all rank other videos first is still tried against each other video. It adds at
   most 3·n(n−1)/2 pairs. For n = 15 that is 315 pairs, 6 % of the budget (section 10).
2. **Expansion.** After the first verification round, for each verified cross-video edge (i, j),
   the best neighbours of j are tried against i, and the best neighbours of i against j. This is
   query expansion on the verified graph. It runs once. It is off by default and step 4 measures
   whether it adds edges on the test set.

### 3.4 Matching and verification

For each candidate pair:

1. Match the SIFT descriptors with `matchFloatDescriptorsGpu` (ratio 0.8, mutual, as today).
2. If the pair has fewer than `minMatches` = 30 matches, drop it. This cheap test removes most
   false candidates before the costly verification.
3. Verify the matches with `verifyImagePairGpu` (the `verifyPair` job of
   `src/reconstruction/mapper.ts`), with the two cameras of the two videos. It runs the RANSAC for
   the essential matrix and for a homography, and it triangulates the inliers.
4. Accept the pair when it has at least `minEdgeInliers` = 25 inliers (Decision, a default to be
   tuned in step 4; COLMAP's matching accepts 15 and the lab's mapper needs 12 initial points).
5. Keep the result: the inlier matches, the relative pose, the inlier count, the homography
   inlier count and the median triangulation angle. The mapper reuses it (4.2).

The weight of an accepted edge is its inlier count. An edge is **cross-video** when its two
keyframes belong to different videos.

Fact: `verifyPairs` in the mapper runs the verification again for every input pair. Without a
change, a pair would be verified twice. Step 5 adds a `preVerified` input to `incrementalSteps`:
a list of `PairVerification` results that `verifyPairs` takes in place of a new run.

### 3.5 Keeping the features

The matching of 4,900 pairs needs the descriptors of all keyframes, not of five. Fact: the worker
keeps GPU handles for a few keyframes only.

Decision: the view-graph stage keeps the keypoints and the descriptors of every keyframe on the
host, and uploads them in batches.

- The descriptors are stored as 8-bit values, as COLMAP does (`round(512 · v)`, clamped to 255,
  where `v` is the RootSIFT value in [0, 1]). A keyframe of 4,096 features takes 512 KiB. For
  N = 375 that is 192 MiB on the host. Float storage would take 768 MiB.
- The pairs are ordered by the first keyframe. A batch uploads one keyframe and matches it against
  all its candidates, so that one upload serves about 13 matchings.
- Open: 8-bit descriptors give matches that differ a little from the float matches of the single
  video path. Step 4 measures the change in the number of inliers on the test set. The result
  decides whether the stage stores 8-bit or 16-bit floats.

### 3.6 What stays the same

The single-video path keeps working as it does. A run with one video is the present run: the
streaming session, the window of 32, and no view graph. The multi-video job is a second code path,
chosen when the run has two or more videos (6.2).

## 4. Components, initialization and growth

### 4.1 Connected components

The components are the connected components of the view graph (union-find over the accepted
edges). Each keyframe belongs to one component. Each video's keyframes can belong to several
components (a video that sees two unrelated scenes).

Rules:

1. A component with fewer than `minComponentViews` = 5 keyframes is not reconstructed. Its videos
   get the tag "not connected" (6.3).
2. A video is **connected** when at least `minVideoShare` = 50 % of its keyframes sit in one
   component that holds at least two videos. Otherwise it is **not connected**.
3. A video alone in its component is reconstructed on its own. The lab shows it as its own map with
   the tag "alone". It is not an error.
4. Components are ordered by their number of keyframes, then by the lowest `viewId`.

An unrelated video (for example the water-tower sample next to a rock formation) forms its own
component. Its edges to the other videos fall under `minEdgeInliers`, or it has none. The distractor
gate (7.2) tests this.

### 4.2 The default: incremental growth over the whole component

Decision: the default is one incremental reconstruction per component, over all its keyframes and
all its verified edges, as COLMAP does. It reuses `incrementalSteps` in
`src/reconstruction/mapper.ts`. The steps and the changes are:

**Initial pair.** Fact: `verifyPairs` scores the seed candidates by
`positiveDepthCount × min(10, medianParallaxDeg)` and drops pairs that are homography-dominated
(homography inliers at least 85 % of the invertible matches and 90 % of the inliers). The best score
wins, then the lower ids. Decision: keep this rule, applied to **all edges of the component**,
inside and across videos. The seed is therefore the pair with the most triangulated points, the
widest angle (up to 10°) and a non-planar geometry. The list of seed candidates is sorted by score
then by canonical ids, so the seed does not depend on the input order.

Inference: a seed from inside one video is likely, because neighbours in time have the most
matches. The score still prefers a wider angle. A cross-video pair can win if it has a wide angle
and many inliers, and then the map starts across two paths. Both give a valid map.

**Next view.** Fact: the next view is the unregistered view with the most 2D–3D correspondences,
with ties to the lower id (`incrementalSteps`). That is COLMAP's rule in the simple form (COLMAP
also weighs how the visible points spread over the image). Decision: keep it, with the canonical
id as the tie-break. No rule uses the input order or the video order.

**Registration and refinement.** Unchanged. PnP with RANSAC registers the view
(`minPnPInliers` 12). Triangulation, the local BA over the last 6 views, and the global BA when the
map grows by 10 % (`globalBundleDue`, PR #113) follow. When no view registers, the recovery pass
runs once at each map size and the failed views get one more try (`recoveredAt`). The final global
BA closes the run.

**Why this is the default.**

1. One gauge. Scale, position and rotation are set once, by the seed. No scale between maps is
   estimated, and a wrong estimate cannot enter.
2. The tracks cross videos. `buildTracks` joins the matches of all verified edges into tracks, so
   a landmark can be seen by views of several videos. The BA constrains the videos together from
   the start. A merge of separate maps by Sim(3) leaves this to the final BA.
3. It reuses code that is tested and measured (the PR #113 schedule, the recovery pass).
4. The next-view rule is by content, so the order does not enter.

**What changes in the mapper (step 5):**

1. `preVerified` (3.4).
2. Seeds of the RANSAC from the ids (2.4).
3. The streaming session's window of 32 is not used. The multi-video job calls
   `reconstructIncremental` or `incrementalSteps` once for each component, with all its views.
   Open: the cost of the global BA over hundreds of views. The schedule of PR #113 limits the
   number of global passes to about log(N)/log(1.1) growth steps.
4. A per-video focal group in the BA (2.3).

### 4.3 The alternative: per-video submaps merged by Sim(3)

The alternative builds one map for each video, then joins the maps.

1. For each video, reconstruct a **submap** from its own keyframes and its internal edges. This is
   the present video path.
2. Build the **submap graph**. Its nodes are submaps. The weight of an edge between two submaps is
   the sum of the inliers of the cross-video edges between them.
3. Take the **maximum spanning tree** of the submap graph (Kruskal; ties by the lower pair of
   `videoId`s). Each tree edge is a link to merge.
4. For each link, estimate a **Sim(3)** transform between the two submaps (scale, rotation,
   translation). Two ways give it. (a) Register the keyframes of submap B into submap A with PnP
   against A's landmarks, and fit the Sim(3) to the camera centers (Umeyama). The existing
   `alignPoses` (`website/src/lib/pose-alignment.ts`) is a similarity alignment of the same kind,
   used today for the reference comparison. (b) Link landmarks across the cross-video matches and fit
   the Sim(3) to the 3D–3D pairs with RANSAC.
5. Apply the transforms along the tree, from the root, which is the submap with the most views.
6. Run a **global BA** over all views and landmarks, with the cross-video tracks.

**Scale** is the main hazard. Each submap has its own arbitrary scale, so the Sim(3) scale is a
free parameter. A wrong scale from few correspondences bends the merged map, and the BA may not
repair it. A chain of merges along the tree also compounds the errors.

**Decision on which one is the default: the incremental growth (4.2).** The reasons are the
one gauge, the tracks across videos from the start, and the reuse of tested code (the list in 4.2).

**When the submap merge runs instead:**

1. **Stalled growth.** The incremental run of a component ends with fewer than 70 % of the
   component's keyframes registered, and the view graph says that more videos connect. The cause is
   usually a thin cross-video link: too few shared landmarks for PnP. The run then reconstructs
   each video as a submap, merges along the tree and runs the global BA. The record names the
   fallback (`growth: 'submaps'`). This is step 8.
2. **A very large component.** A component with more than 600 keyframes (Decision, to be measured in
   step 7) would make the incremental global BA too slow. The submaps run in parallel in time, and
   only the merge and the final BA see the whole map.
3. **A new video added to a finished map.** The new video's submap is merged to the finished map
   with one Sim(3), without a run of the whole component.

The incremental path stays the reference in all three cases: a result of the submap path is
compared with the incremental result on the test set, and the record reports the difference.

### 4.4 The tie-break rule, stated once

Wherever a choice can tie, the tie breaks by the canonical `viewId` (2.4) or by the lower pair of
`videoId`s. No tie breaks by the arrival order, the position in the upload list, the file name or
a `Map` iteration order of unsorted input. A test lists the places and fails if a changed order
changes the output (7.1).

## 5. Failure modes

Each row names the failure, what the lab shows, and the remedy. The viewpoint figures come from the
SIFT literature and are background, not measurements of this project.

| Failure                         | Cause                                                                                          | What the lab shows                                                                                                                                 | Remedy                                                                                                                                                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Viewpoint gap**               | SIFT matches fail above a viewpoint change of about 30 to 45° between two videos (background). | Two components, each video tagged "not connected" to the other. The tooltip gives the best cross-video inlier count that fell under the threshold. | Add a **bridging video**: a path that passes through the intermediate angles, such as a vertical ascent between two orbit rings. The set then joins in one component.                                      |
| **Pure-rotation clip**          | A pan or tilt from one position has no baseline. Pairs are homography-dominated.               | The video carries the tag "rotation". Its views register only if they see landmarks of other videos.                                               | `verifyPairs` already refuses a homography-dominated seed. The clip registers by PnP against the landmarks of other videos. Its own new points are not triangulated.                                       |
| **Scale difference**            | One video shoots the object from far and another from close (a factor above about 2 to 3).     | Few cross-video edges, or none.                                                                                                                    | A video at an intermediate distance bridges. The incremental path needs no scale estimate.                                                                                                                 |
| **Lighting and exposure**       | Different time of day, shadows, auto exposure.                                                 | Fewer inliers across videos. The weak edges fall under the threshold.                                                                              | RootSIFT tolerates moderate change. The lab's CLAHE option (`clahe`) may help. Not measured.                                                                                                               |
| **Distractor with false links** | Repeated structure (sand, rock texture) produces a few false matches between unrelated videos. | An unrelated video joins a component.                                                                                                              | The geometric check (E and H RANSAC) with `minEdgeInliers` 25 removes most. The distractor gate (7.2) measures it. Open: a check that a component's cross-video edges agree with one rigid motion, step 7. |
| **Different cameras**           | Per-video intrinsics are wrong.                                                                | High reprojection error on one video.                                                                                                              | A per-video focal field and a per-video focal refinement (2.3).                                                                                                                                            |
| **Duplicate video**             | The same file uploaded twice.                                                                  | Tag "duplicate". One copy is used.                                                                                                                 | The `videoId` hash (2.4).                                                                                                                                                                                  |
| **Too many keyframes**          | Many long videos.                                                                              | The run stops at `maxKeyframes` of the run (Decision: 800) with a message.                                                                         | The cap per video (2.2). The user removes a video.                                                                                                                                                         |

## 6. Lab and API

The owner dislikes text-heavy UI. The lab shows state in tags, small marks and tooltips. It adds no
paragraph of explanation to the page.

### 6.1 Upload

1. The video file input takes several files (`multiple`). Fact: the control now accepts one file at
   a time (`website/src/components/SfmLab.tsx`, `accept="video/*,application/json,.json"`). The
   JSON input already accepts several files.
2. Each file becomes a row: a thumbnail of its first keyframe, its label, its focal field, its
   keyframe count, and a tag (6.3). A row has a remove control.
3. One "Run" starts the multi-video job when two or more videos are present. With one video the
   present run starts.
4. The page decodes the videos one after another (one decode worker, bounded memory). It sends the
   frames of one video in time order. The order of the videos is the list order. It does not matter
   for the result (7.1).

### 6.2 The worker

A new job type, `multiVideo` _(new)_, in `website/src/workers/sfm.worker.ts`. Its messages:

- `video`: the start of one video (`videoId`, label, camera). The worker makes a `PhaseGate` for it.
- `frame`: as today, with a `videoId`. The worker runs the gate, extracts SIFT for a keyframe, keeps
  the keypoints and descriptors on the host (3.5) and computes the global descriptor.
- `finish`: the worker assigns `viewId`s (2.4), builds the view graph (section 3), finds the
  components (4.1), reconstructs each component (4.2) and publishes the models.

The worker posts a `graph` message when the view graph is ready. It posts a `model` message for each
component, as `publish` does today. The messages carry the data of section 6.3, not text.

### 6.3 Samples, tags and the viewer

- **Sample set card.** The sample catalog (`website/public/sample/index.json`) gets an entry of kind
  `video-set` _(new)_ with a `members` list of sample ids. The card shows a stack of thumbnails and
  the count of videos, for example "11 videos". One click loads every member into the upload list.
  The first sets are `chimney-rock-a` (4 videos) and `chimney-rock-b` (11 videos) and
  `chimney-rock` (all 15), in section 9. A card for a set with an unrelated member, `chimney-rock-distractor`,
  is the manual form of the distractor gate.
- **Tags on a video row.** `component 1`, `component 2`, ..., `alone`, `not connected`, `rotation`,
  `duplicate`. A tag has a tooltip with the numbers (views registered, cross-video inliers, the
  best rejected edge). Each component has a color. The color also marks the cameras in the viewer.
- **Viewer.** The viewer shows one component at a time, selected by its tag. The cameras are colored
  by video. No explanatory text is added.
- **View-graph matrix** (optional, step 6). A small grid of n × n cells. A cell's shade is the
  number of cross-video inliers between two videos. It shows at a glance which videos connect.

### 6.4 The run record

Fact: the record is `typecv-run-record-v5` (`website/src/lib/run-record.ts`,
`docs/lab-run-records.md`). Earlier additions (`options.keyframeGate`, `options.keyframes`) are
optional fields and the format number stayed. This design does the same: new fields are optional and
the format number stays. The validator accepts a record without them.

| Field                    | Content                                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------------------- |
| `input.videos`           | A list: `videoId`, label, duration, keyframes, focal ratio, component number, tags.                 |
| `options.multiVideo`     | `descriptor` (`bow-512-v1`), `k`, `minMatches`, `minEdgeInliers`, `maxKeyframesPerVideo`, `growth`. |
| `result.components`      | A list: keyframes, registered views, landmarks, videos, mean reprojection error.                    |
| `result.crossVideoEdges` | The count of accepted cross-video edges, and the sum of their inliers.                              |
| `result.viewGraph`       | Nodes, candidate pairs, matched pairs, verified edges, and the same count for exhaustive matching.  |
| `result.notConnected`    | The `videoId`s that join no component of two or more videos.                                        |
| `timing.stages`          | New operations: `globalDescriptor`, `retrieval`, `pairMatching`, `pairVerification`, `components`.  |

The record does not hold the file names (a privacy rule of the collector: only the hash and a label
the visitor typed). Open: whether the label is sent. Default: not sent for uploads, sent for lab samples.

## 7. Verification plan

The house rules apply: a gate shows that it can fail before it is trusted to pass.

### 7.1 The shuffle gate (order independence)

Input: a set of at least 4 videos (a lab sample set). Run it in three input orders: the list order,
the reverse order, and one fixed random permutation. Compare the three results.

The gate passes when all of these hold:

1. **Same keyframes.** Each video has the same set of `frameKey`s in the three runs.
2. **Same registered view set.** The set of `frameKey`s of registered views is equal.
3. **Same components.** The partition of the `videoId`s (and of the keyframes) is equal.
4. **Same cameras, up to similarity.** After a similarity alignment of the runs' camera centers on
   their common views (`alignPoses`), the median center error divided by the trajectory radius is at
   most 0.005 and the largest at most 0.02. The rotation error is at most 0.5° median.
   Inference: on one device and one build the values are near zero, because all ids, seeds and ties
   are canonical. The tolerance is for the order of floating-point reductions on a GPU.

Three layers of test:

- **CPU tier (CI).** A synthetic scene with 3 camera paths (orbit, ascent, lateral) renders its
  observations. The test shuffles the order of the videos and of the ids and calls
  `incrementalSteps` and the view-graph function. It checks items 2 to 4. It runs in `npm test`.
- **Lab, SwiftShader (CI, section 10 gives the reduced set).** Real frames of a sample set.
- **Prove the instrument.** A debug switch (`?canonical=0` _(new)_) assigns `viewId`s by arrival
  order and turns the canonical seeds off. The gate must **fail** with it on a set where the order
  changes the seed pair. A gate that cannot fail proves nothing.

### 7.2 The distractor gate

Input: the `chimney-rock-b` set plus the `water-tower` lab sample (an unrelated video).

Passes when:

1. The water-tower video is in its own component, tagged "alone" or "not connected".
2. No accepted edge joins a water-tower keyframe to a keyframe of another video.
3. The other videos' components are the same as in the run without the distractor (same
   partition of their `videoId`s).
4. The record states the margin: the largest inlier count among the water-tower candidate pairs
   that were verified and rejected. A margin under half of `minEdgeInliers` is the target. A larger
   margin is reported and discussed. It is not hidden.

### 7.3 The connection gate (the test set)

Input: the sets of section 9: group A (4 videos), group B (11 videos), and A and B together (15).

This gate **records numbers first**. The owner confirmed by eye that all 15 videos show one object,
but nobody has measured whether SIFT can connect them. Step 7 runs the sets, records the numbers,
and this record is amended with the thresholds before the gate becomes a CI check.

| Recorded                              | Meaning                                                             |
| ------------------------------------- | ------------------------------------------------------------------- |
| Cross-video edges, per pair of videos | The matrix of 6.3. Which videos connect, and with how many inliers. |
| Registered views, per video           | How many keyframes of each video are in the map.                    |
| Components                            | Whether one map forms, for A, for B, and for A and B together.      |
| Mean reprojection error               | Of each component.                                                  |
| Time                                  | Of each stage (6.4), on SwiftShader and on a physical GPU.          |
| Growth                                | `incremental`, or `submaps` after a stall (4.3).                    |

Provisional pass rule (to be confirmed or changed by the recorded numbers): in group B at least one
component holds at least 8 of its 11 videos, with a mean reprojection error under 1 px. Group A
alone forms one component of its 4 videos or the record explains why not. The A and B sets together
either form one component or the record names the missing bridge (5).

### 7.4 CPU-reference and WebGPU parity of the global descriptor

The global descriptor adds one kernel, `globalAssign`, and a host step. The tests follow the house
form of the KLT and MVS work:

| Gate                                 | Number                                                                                                                                    | Where                 |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| Reference against a naive definition | the histogram of a hand-made descriptor set equals the hand count                                                                         | `npm test`            |
| Kernel against reference, CPU tier   | the word indices are equal (the argmin ties break to the lower index); the descriptors equal to 1e-6                                      | `npm run test:shade`  |
| Kernel against reference, WebGPU     | at least 99.9 % of the word indices equal (a floating-point near-tie can flip a word); the cosine of each descriptor pair at least 0.9999 | `npm run test:webgpu` |
| Chunk invariance                     | the output does not depend on the chunk size of the time budget                                                                           | `npm run test:shade`  |
| Retrieval recall                     | recall at 10 of the true pairs, on the test set and on the photo samples (3.2)                                                            | lab, recorded         |

### 7.5 Gates summary

| Gate              | Pass condition                                               | Where                          |
| ----------------- | ------------------------------------------------------------ | ------------------------------ |
| Shuffle           | 7.1, items 1 to 4; and it fails with `?canonical=0`          | CPU tier (CI); lab reduced set |
| Distractor        | 7.2, items 1 to 3                                            | lab                            |
| Connection        | 7.3, numbers recorded; thresholds set by amendment           | lab, run records               |
| Descriptor parity | 7.4                                                          | `test:shade`, `test:webgpu`    |
| Single-video path | the present single-video runs give the same result as before | existing lab gates             |
| Physical devices  | the 15-video set completes without a device loss             | lab, run records               |

## 8. Steps

Each step is one pull request. Each commit names this document.

| Step | Pull request content                                                                                                                                                                                                                                               | What it proves                                                                                           |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| 1    | **Publish the test set.** 13 new entries in `website/samples/remote-videos.json` with `set` and `group` fields. `scripts/lab-remote-samples.mjs catalog` and `publish`. The `video-set` catalog entries. A test of the manifest.                                   | All 15 clips have frames in the bucket. The manifest, the catalog and the sets agree (section 9).        |
| 2    | **Canonical identity.** `videoId`, `frameKey`, `viewId` (2.4). Seeds from ids in `verifyPairs`. The CPU shuffle test of `incrementalSteps`. The decoder-agreement measurement.                                                                                     | Shuffled ids give the same map. How often two decodes of a frame differ.                                 |
| 3    | **Global descriptor.** `globalDescriptorReference`, the kernel `globalAssign`, the vocabulary `rootsift-512-v1`, the idf. The recall measurement of the three candidates. A record of the choice (amendment).                                                      | Parity of 7.4. The default of 3.2 holds, or the record is amended with the winner.                       |
| 4    | **The view graph.** `buildViewGraph` _(new)_: candidates (3.3), host descriptor store (3.5), matching and verification of candidates (3.4), components (4.1). A CPU-tier test on a synthetic 3-path scene and on the photo samples. Recorded: expansion on or off. | N·k pairs instead of N². Components are right on synthetic data. The 8-bit against float inlier change.  |
| 5    | **The mapper on a component.** `preVerified` input, per-component call of `incrementalSteps`, per-video focal groups in the BA, the content tie-breaks of 4.4 under a test.                                                                                        | A component reconstructs from verified edges with no second verification. The shuffle gate (CPU) passes. |
| 6    | **Lab and worker.** Multi-file upload, the `multiVideo` job, sample set card, tags, per-component viewer, the record fields (6.4), the optional graph matrix. `?canonical=0`.                                                                                      | The lab runs a set end to end. The record validates. The shuffle gate fails with `?canonical=0`.         |
| 7    | **The gates on the test set.** The shuffle gate on the lab (reduced set on SwiftShader). The distractor gate. The connection gate numbers on groups A, B and A+B, then an amendment with the thresholds. Physical-GPU runs.                                        | 7.1 to 7.3 with numbers. The costs of section 10 against measured values.                                |
| 8    | **Submap fallback.** Per-video submaps, the maximum spanning tree, the Sim(3) of 4.3, the global BA, the stall rule, the comparison with the incremental result.                                                                                                   | The stalled set joins, and the result is near the incremental one where both run.                        |

Steps 1 and 2 need no lab change. Steps 3 and 4 can run in parallel after step 2. Step 8 can wait
for the numbers of step 7.

## 9. The test set

The owner confirmed by eye that all clips show the same object: Chimney Rock, Colorado. The
creator is EJ Merl (Pexels). The license is the Pexels license (`https://www.pexels.com/license/`),
checked 2026-10-20 in `website/samples/remote-videos.json`.

| Group | Pexels video ids                                        | Count | Already a lab sample          |
| ----- | ------------------------------------------------------- | ----- | ----------------------------- |
| A     | 32291999, 32292000, 32292001, 32292002                  | 4     | 32292001 is `desert-monument` |
| B     | 32332978, 32332979, ..., 32332988 (every id in between) | 11    | 32332986 is `colorado-rock`   |

The unrelated video for the distractor gate is `water-tower` (Pexels 33341651, also by EJ Merl).
Fact: it is already a lab sample.

**How the samples are published (fact).** `website/samples/remote-videos.json` lists each clip with
`id`, `title`, `creator`, `page`, `file` (a Pexels download URL), `duration` and a bilingual
`description`. `scripts/lab-remote-samples.mjs catalog` writes the entries into the sample catalog
(`catalogEntry`, `withRemoteEntries`). `publish` downloads one clip at a time with curl, checks the
duration with ffprobe against the manifest (it fails when they differ by 0.5 s or more), cuts the
frames at `videoSampleTimes(duration)` with ffmpeg, scales them to 640 × 360 and uploads them to the
R2 bucket `typecv-samples` under `sample/<id>/`. It needs `CLOUDFLARE_API_TOKEN`. The site's Worker
serves `/sample/<id>/...`.

**What step 1 does:**

1. For each of the 13 clips not yet in the manifest, read the Pexels page for the file URL
   (the numeric file id in the URL is not the video id and cannot be derived from it) and the
   duration. Prefer a file of about 1920 × 1080 or 3840 × 2160. Add an entry. Use the ids
   `chimney-rock-<pexels id>`, with the title "Chimney Rock video" plus the group and the number.
   Keep the two existing ids (`desert-monument`, `colorado-rock`) and add `set` and `group` to them.
2. Add the optional fields `set` (`"chimney-rock"`) and `group` (`"A"` or `"B"`) to every entry
   of the set. The manifest keeps its format `typecv-remote-videos-v1`. `readManifest` checks
   only the format string, so the fields are optional and old readers ignore them.
3. Add the `video-set` entries `chimney-rock-a`, `chimney-rock-b`, `chimney-rock` and
   `chimney-rock-distractor` (B plus `water-tower`) to the catalog (section 6.3).
4. Run `catalog` and `publish --only <ids>`. The publish needs the owner's token. Open: who runs it.
   The existing remote samples were published from CI (the script's comment says "(CI)").
5. Record the date, the commit and the durations in the pull request.

Fact not yet checked: the durations and the frame times of the 13 new clips. The sampling at 2 frames
per second of the sample frames (`videoSampleTimes`) is not the 4 frames per second of an uploaded
video. Section 2.2 and 10 use the uploaded-video rate. The sample frames are the lab's way to run a
video without a decoder.

## 10. Cost estimate (inference)

This section is an inference. No multi-video run exists yet. Step 7 replaces each number with a
measured one.

**Keyframes per video.** Fact: the video record `20261020121308655-a6fb6f3b` (RTX 2080) made 27
keyframes from 566 decoded frames of one 4K clip (`docs/klt-tracking.md`, section 1). The clips of
the test set last about 12 to 30 s (the two known ones: 29.7 s and 19.5 s). At 4 frames per second
that is 48 to 120 sampled frames. Assume 25 keyframes for a video (range 15 to 40). For 15 videos,
N = 375 (range 225 to 600).

**Candidate pairs.** With k = 10 and 3 temporal neighbours on each side (13 candidates per keyframe,
of which some are the same pair counted from both ends):

| Quantity                            | Value                            |
| ----------------------------------- | -------------------------------- |
| Directed candidates, N·13           | 4,875                            |
| Unique pairs (between 50 and 100 %) | about 2,500 to 4,875             |
| Reserve (3.3, guard 1)              | up to 315                        |
| Exhaustive pairs, N(N−1)/2          | 70,125                           |
| Saving                              | about 14 to 28 times fewer pairs |

**Cost per stage.** Fact: the 8-view pyramid benchmark has 28 pairs
(`docs/lab-run-records.md`, "SwiftShader against the RTX 2080"). Its warm pass is the only table
with both devices.

| Stage             | SwiftShader, 8 views | RTX 2080, 8 views | Per unit, SwiftShader | Per unit, RTX 2080   |
| ----------------- | -------------------- | ----------------- | --------------------- | -------------------- |
| Features          | 58.8 s               | 0.53 s            | 7.4 s per keyframe    | 0.066 s per keyframe |
| Matching          | 159.4 s              | 0.25 s            | 5.7 s per pair        | 0.009 s per pair     |
| Pair verification | 11.3 s               | 8.08 s            | 0.40 s per pair       | 0.29 s per pair      |

Scaled to N = 375 keyframes and 4,875 pairs (the upper end), at 640 px and 4,096 features:

| Stage                     | SwiftShader            | RTX 2080               | Note                                                   |
| ------------------------- | ---------------------- | ---------------------- | ------------------------------------------------------ |
| Features, 375 keyframes   | about 2,760 s (46 min) | about 25 s             | Only keyframes pay (the phase gate runs first).        |
| Global descriptor         | tens of seconds        | under 1 s              | 100 GFLOP (3.2).                                       |
| Matching, 4,875 pairs     | about 27,800 s (7.7 h) | about 44 s             | Arithmetic-bound on SwiftShader.                       |
| Verification, 4,875 pairs | about 1,950 s (33 min) | about 1,410 s (24 min) | Round-trip-bound: the RTX gains only 1.4 times.        |
| Mapper (BA, PnP)          | not estimated          | not estimated          | Depends on the size of the component. Step 7 measures. |

Three readings of the table (all inference):

1. **Verification dominates on a physical GPU.** The matching is cheap there. The verification
   costs about 0.29 s a pair, because of many small submissions. The `minMatches` filter of 3.4
   removes most false candidates before the verification. If it removes half of the pairs, the
   verification time halves. A batch verification of many pairs in one submission would cut it
   more. That is a later optimization, and the record of the resident pair verification
   (`docs/experiments/gpu/resident-pair-verification-software.json`) is the place to start.
2. **The full set does not run on SwiftShader.** Matching alone would take hours. The CI gates use
   a **reduced set**: 5 videos (two from each group and the distractor), `maxKeyframesPerVideo` 8
   (N = 40), 1,024 features. Matching cost scales with the square of the feature count, so 1,024
   features cost about 1/16 of 4,096: about 0.36 s a pair. With 40 × 13 = 520 candidate pairs,
   matching takes about 3 minutes and verification about 3.5 minutes. The reduced set tests the
   order, the separation and the plumbing. It does not test the connection of the full set.
3. **The connection gate needs a physical GPU or a long run.** The owner's machine (RTX class)
   runs the full set in the order of 30 minutes by this estimate.

**Memory.** Host descriptors: 192 MiB for N = 375 in 8-bit form (3.5). GPU: one keyframe's
descriptors and its candidates' descriptors in a batch, under the working budget of the device
(64 MiB at 640 px).

## 11. Risks and open questions

**SIFT may not connect the paths.** The owner confirmed the object by eye. The paths can still be
too far apart for SIFT (section 5). The test set may give two components. That is a result of the
gate and a correct answer, not a defect. Step 7 records it. If groups A and B do not join, the
record says which bridge is missing. Open.

**The vocabulary is trained on the lab's samples.** Samples are mostly outdoor, textured scenes.
A scene with other statistics (indoor, a face) can retrieve worse. The idf weights adapt to the
set, and the recall measurement of step 3 uses the photo samples too. Open: whether a vocabulary
trained on the run's own descriptors (a deterministic k-means in a canonical order) is better.
Step 3 compares the two.

**Retrieval misses a true pair.** A true pair outside the top 10 is lost. The guards of 3.3 (the
per-video reserve and the expansion) reduce this. The recall at 10 is measured in step 3. A value
under 0.9 of the exhaustive result raises k. Open: the best k.

**False edges between unrelated videos.** Repeated textures can give an edge that passes the
check. The distractor gate measures the margin on one case. Open: a consistency check on the
cross-video edges of a component (they must agree on one rigid motion between the two videos).

**Verification time on a physical GPU.** About 24 minutes for the full set (section 10). This is
the dominant cost. Open: batch verification, and a lower RANSAC iteration count for pairs that
already have a high inlier ratio.

**The global BA on a large component.** Hundreds of views make the global BA slow. The schedule
of PR #113 helps. Open: the cap of 600 keyframes in 4.3 is a guess. Step 7 measures the time of the
final BA against N.

**Per-video focal refinement.** One focal for each video adds parameters and can drift when a video
has little parallax. Open: use the shared form unless the cameras differ. Step 7 compares.

**8-bit descriptors.** A change in the matches against the float path (3.5). Step 4 measures it.

**Decoder dependence of the hash.** `frameKey` can differ between two decodes of one video
(2.4). The shuffle gate runs on one decoder, so it is not affected. Open: a hash of a coarse
thumbnail, if step 2 shows a difference.

**Memory on a phone.** 192 MiB of host descriptors plus decoded frames. The per-video cap and
`maxKeyframes` (5) bound it. Open: the limit on a phone browser. Not measured.

**Dense MVS.** The dense path has a limit on the number of views (`docs/dense-mvs.md`, section 4.2).
It runs on one component at a time and selects its views as it does now. This record changes no
dense code. Open: which component a click on "Dense" uses. Default: the selected one.

**Privacy of the record.** The record holds hashes, not names (6.4). The label of a sample is public.
Open: whether a typed label is sent. Default: no.

**Publishing the samples.** The `publish` command needs the owner's Cloudflare token (section 9).
Open: whether the CI job that published the earlier samples can run it again.

## 12. Record of this document

Written on 2026-10-20 from `main` at `ebc39b6`. Files read: `website/src/lib/sfm-video.ts`,
`website/src/lib/keyframe-gate.ts`, `website/src/lib/video-sampling.ts`,
`website/src/lib/pose-alignment.ts`, `website/src/workers/sfm.worker.ts`,
`src/reconstruction/mapper.ts`, `src/gpu/mapper.ts`, `scripts/lab-remote-samples.mjs`,
`website/samples/remote-videos.json`, `docs/lab-run-records.md`, `docs/klt-tracking.md` and
`docs/dense-mvs.md` (as the model of form). No code was changed, and no check other than the
formatting check of this file was run.

Not checked: the Pexels pages of the 13 new clips (their durations and file URLs), the
`SampleEntry` type in `website/src/lib/sample-catalog.ts` (the `video-set` kind needs a type change),
and the validator in `website/src/lib/run-record.ts` (the new optional fields need its update).
Step 1 and step 6 check them.

Decisions on the open questions of the draft, made by the coordinator by default on 2026-10-20:

1. Step 1 publishes the test set with `scripts/lab-remote-samples.mjs publish` from the session
   container, as `desert-monument` was published. If the container has no Cloudflare credential,
   step 1 stops and reports it to the owner.
2. The `video-set` catalog kind is accepted.
3. The typed label of an uploaded video is not sent in the run record.
4. The provisional connection-gate rule is accepted until step 7 records numbers: at least 8 of
   the 11 group B videos in one component, with a mean reprojection error under 1 px.

### Amendment 1: step 2, decoder agreement (measured, 2026-10-20)

Step 2 implements the identity of 2.4: `videoIdOf`, `frameKeyOf`, `assignViewIds` and `pairSeed` in
`src/reconstruction/identity.ts`. `verifyPairs` seeds each pair with `pairSeed(seed, first,
second)`. The CPU tier of the shuffle gate (7.1) is `tests/multi-video-identity.test.ts`.

**Measured.** How often two decodes of one video give different `frameKey`s.

- Command: `ffmpeg -i <clip>.mp4 -t 10 -vf scale=1280:-2 -c:v libvpx-vp9 -an <clip>.webm`, for three
  clips of the test set (section 9), then
  `node scripts/measure-decoder-agreement.mjs a.webm b.webm c.webm`.
- Configuration: `main` at `beb8f1e` plus this change, Node 22.22.0, Playwright 1.56.1, Chromium
  141.0.7390.37 (software VP9 decode, headless, no GPU), mediabunny 1.61.6, 4 frames per second,
  analysis edge 640, 2 decodes on each path of the lab (the page and the decode worker).
- Result, 2026-10-20: 41 frames per clip, 123 frames in all, 41 different keys in each clip.
  Repeat on the page path: 0 of 123 keys differ. Repeat on the worker path: 0 of 123. Page path
  against worker path: 0 of 123.

**What this does not show.** The Chromium build of the container has no H.264 decoder, so the
clips were transcoded to VP9 and no H.264 decode was measured. Hardware decoders, other browsers
and other devices were not measured. The result holds for one machine and one browser build.

**Disposition of the open question in 2.4.** On one decoder the full-RGBA hash is stable, so it
stays the definition of `frameKey`. The question whether two devices agree stays open. The
thumbnail hash is the remedy if a later measurement on two devices shows a difference.

**Found, not changed.** `runIncrementalGpu` (`src/gpu/mapper.ts`) replaces the seed of a pair in a
streaming session by `seed + session.verification.size`, a count of the arrival order. Step 2 does
not change it: the record names `verifyPairs` only, and the multi-video job does not use the
session (4.2, item 3). Step 5 or step 6 decides. The `seed + order.length` of the PnP request in
`incrementalSteps` and `seed + cameraIds.length` in `scene-mapper.ts` count registered views. The
count follows the content-driven registration order, so these stay.

### Amendment 2: step 3, the global descriptor (measured, 2026-10-21)

Step 3 implements the descriptor of 3.2 and the parity gate of 7.4. The code: `globalDescriptorReference`
and its parts in `src/gpu/global-descriptor.ts`, the kernel `globalAssign` in
`src/gpu/global-descriptor.shade.ts` with its host call `assignWordsGpu` and `globalDescriptorGpu` in
`src/gpu/global-descriptor-gpu.ts`, and the vocabulary `rootsift-512-v1` in `src/gpu/vocabulary/`. The
candidates VLAD and the thumbnail, and the k-means, are in `src/gpu/global-descriptor-candidates.ts`.
The scripts are `scripts/global-descriptor-data.mjs` and `scripts/global-descriptor.mjs`.

**The vocabulary.** Command: `node scripts/global-descriptor.mjs train <work-dir>`. It runs Lloyd's
k-means (25 iterations, seed 1, the initial centers are 512 distinct training points) on the RootSIFT
descriptors of the lab's samples: 103,991 descriptors from 150 frames (at most 1,000 for each frame) of the
video samples `pyramid`, `pyramid-dense`, `canyon`, `bridge-rocks`, `desert-arch`, `water-tower` and
`devils-slide` (every fourth frame) and the photo samples `south-building`, `south-building-24`,
`gerrard-hall`, `person-hall`, `graham-hall` and `tum-fr3-structure-texture-far`. The file has 512 × 128 f32
values (256 KiB). `rootsift-512-v1.json` holds its SHA-256 and the training facts. Decision: the 15 videos of
the test set (section 9) are not in the training data, so that the recall on them is not a measure of the
training data. This is a narrower reading of "the lab's samples" in 3.2.

**The recall protocol.** Fact: 3.2 defines a true pair as a pair that the exhaustive SIFT matching and the
verification accept with at least `minEdgeInliers` (25) inliers. The measurement does this:

1. Features: RootSIFT of the CPU tier (`extractSiftGpuResident`, at most 4,096 features) on the sample
   frames of `https://cv.typeshade.dev/sample/<id>/frame-NN.jpg` (640 px).
2. Frames: the test set keeps every third sample frame (188 frames of 15 videos, about 12.5 for each). The
   photo corpus holds all 88 photos of the six photo samples. Each photo counts as its own video.
3. Candidates of a pair: for the test set, all pairs of two different videos and the pairs of one video
   that are more than 10 kept frames apart (16,514 pairs). For the photos, all pairs inside one sample (660).
4. Matching: nearest and second nearest by L2, ratio 0.8, mutual, on the host (the rule of
   `matchFloatDescriptorsGpu`). Fewer than 30 matches drops the pair. `verifyImagePair` (CPU) then verifies.
   The camera of a video frame has a focal length of 0.8 × the width and no distortion (the lab default). The
   camera of a photo comes from the `sample.json` of the sample.
5. True pairs: 626 in the test set (they join 21 of the 105 pairs of videos) and 189 in the photos.
6. Retrieval: the cosine of the descriptors of 3.2. A query ranks all eligible frames (3.3: the frames of its
   own video within 10 kept frames are not eligible). The tf-idf weights come from the frames of the corpus.
   A true pair is found at k when it is in the top k of either end (`either`, the union of 3.3 makes pairs
   unique). The table also gives the share of the directed pairs (`directed`).
7. Candidates: the bag of words as in 3.2; VLAD with 32 words (a k-means on the training descriptors of
   the vocabulary, same seed and iterations), intra-normalized, 4,096 values; and the 16 × 16 gray
   thumbnail (area scaling by ffmpeg, mean removed, unit norm), 256 values. Two diagnostics: the bag of words
   with a vocabulary trained on the corpus's own descriptors (the open question of section 11), and, for the
   photos, with a vocabulary trained without the photo samples.

Configuration: `main` at `92639f9` plus this change, Node 22.22.0, the TypeShade pin of `package-lock.json`,
CPU tier SIFT. The commands: `global-descriptor-data.mjs download|extract`, then `global-descriptor.mjs train`,
`truth` and `recall`. The pair counts above come from `truth`.

**Result.** Recall at k = 10 of the true pairs (`either`; `directed` in brackets).

| Candidate                                              | Dim   | Test set (626 pairs) | Photos (189 pairs) |
| ------------------------------------------------------ | ----- | -------------------- | ------------------ |
| Bag of words 512, tf-idf, `rootsift-512-v1`            | 512   | 0.692 (0.534)        | 0.947 (0.886)      |
| VLAD 32 words, intra-normalized                        | 4,096 | **0.781** (0.680)    | **0.974** (0.931)  |
| Downsampled image 16 × 16 gray (baseline)              | 256   | 0.559 (0.479)        | 0.963 (0.905)      |
| Diagnostic: bag of words, own-descriptor words         | 512   | 0.768 (0.625)        | 0.958 (0.897)      |
| Diagnostic: bag of words, no photo samples in training | 512   | not applicable       | 0.847 (0.741)      |

At k = 20 the bag of words gives 0.853 on the test set and 0.989 on the photos, VLAD 0.898 and 1.000, the
thumbnail 0.677 and 0.995.

**Decision rule of 3.2, applied.** The bag of words stays the default when its recall at 10 is at least 0.9 of
the best candidate. On the photos the ratio is 0.947 / 0.974 = 0.97 (met). On the test set it is
0.692 / 0.781 = 0.885 (not met, and 0.78 for `directed`). By the rule, the measured winner is VLAD with 32
words. Disposition: **open**. The code of this change delivers the bag of words as 3.2 declares, because
the interface that step 4 consumes depends on the choice. The coordinator decides whether to switch the
default to VLAD (the kernel `globalAssign` serves it unchanged with a vocabulary of 32 words; the residual sums
are a host step over the descriptors that 3.5 keeps), or to keep the bag of words and raise `k`. Two facts bear
on the decision. A vocabulary trained on the run's own descriptors reaches 0.768, which is 0.98 of VLAD (the
open question of section 11, answered for this set). And at k = 20 the bag of words reaches 0.95 of VLAD
(0.853 / 0.898).

**Not shown.** The test set has 15 videos of one object, so the measurement has 21 connected pairs of videos
and few independent samples. The ground truth uses a host matcher and the CPU verifier, a thinned frame set
(about 12 frames for each video, not the 25 of section 10), the sample frames at 2 frames per second, and a
guessed focal length. The photo recall is in-sample for the photo samples (the 0.847 row is the held-out
estimate). Time of the stages and results on a physical GPU were not measured.

**Parity (7.4).** Reference against a naive definition: `tests/global-descriptor.test.ts` (`npm test`). CPU tier:
`tests/typeshade/global-descriptor.test.mjs`: the word indices equal the reference exactly (random descriptors,
descriptors near a word and descriptors on a word), the descriptors equal to 1e-6, and the words are the same
for chunk sizes of 1 to 1,000 descriptors and for two time budgets (`npm run test:shade`). WebGPU on SwiftShader:
`tests/typeshade/global-descriptor-webgpu.test.mjs`: at least 99.9 % of the word indices equal and the cosine of
each descriptor pair at least 0.9999, for host descriptors and for the resident descriptors of a SIFT run
(`npm run test:webgpu`). The numbers of the runs are on the pull request.
