# Hardware observations

The SfM lab at `/lab/` and the triangulation probe at `/lab/probe/` run TypeCV's GPU pipeline in
the visitor's browser. With the visitor's consent, each run sends one record to the site's
collector (`docs/lab-run-records.md`). The **Export lab run records** workflow
(`.github/workflows/lab-records.yml`) reads the records back ("Reading the records" in that
document).

This document is the digest of those records by device. It covers the owner's two devices,
2026-10-19 to 2026-10-20. [Lab run records](lab-run-records.md) is the chronological log, with
every observation paragraph in the order it was made. This document gathers each device's
observations in one place, with what each one changed in the code.

Each observation has four parts:

- **Observed.** Facts copied from the records. A record id is the collector's id, the received
  time in UTC and a hash, for example `20261020012028865-b2ca931e`. A probe record is named by its
  hash alone.
- **Inference.** What the records suggest and do not prove. Each inference is labelled.
- **Remedy.** The change in this repository and its pull request, or the branch that is not
  merged yet.
- **Open.** What the records do not settle yet.

## Configurations

| Site commit | Pull request | Compiler pin | What the site added                                                     |
| ----------- | ------------ | ------------ | ----------------------------------------------------------------------- |
| `486df3b`   | #42          | `212d26d2`   | The lab of 2026-10-19 morning, before both flavors of `f64`             |
| `7eef646`   | #54          | `41a140a`    | The time budget in the compiler's device (change 0074)                  |
| `329043c`   | #59          | `d70b539`    | Run record v4 with the chunk rates. `triangulateTracks` in one dispatch |
| `a3041af`   | #60          | `d70b539`    | `triangulateTracks` as chunks of tracks                                 |
| `52d39bb`   | #64          | `d70b539`    | The probe page `/lab/probe/`                                            |
| `cd97b62`   | #65          | `d70b539`    | The probe sends its result as a probe record                            |

The site at `7eef646` also holds pull requests #49, #50 and #53. The site at `52d39bb` also holds
pull request #63. The pin `d70b539` implements the compiler's changes 0073, 0074 and 0075.

## Device A: NVIDIA Blackwell (observed on a GeForce RTX 5060 Ti), Windows, Chrome 155, D3D12

The records identify the GPU by its architecture only. The WebGPU adapter reports vendor `nvidia`,
architecture `blackwell`, and an empty device and description. The model, a GeForce RTX 5060 Ti,
is the owner's statement of 2026-10-20. The owner also stated that every Windows run of
2026-10-19 and 2026-10-20 was on this computer. The other facts of the computer: Windows 10/11,
Chrome 155 with Dawn on the D3D12 backend, 16 hardware threads and 32 GB of memory.

The flavor of the emulated double on this device is `'float'`. `recommendFp64Flavor` picks it
(`ShadeDevice.open`, `fp64Flavor: 'auto'`), and only `?fp64=integer` forces the other one.

### A1. A workgroup whose idle lanes return before the loops hangs the device

`triangulateTracks` (`src/gpu/triangulation.shade.ts`, `@compute([16])`) solves one track per
lane in emulated `f64`. Each lane runs up to `maxPairHypotheses + 1` hypotheses, each with a
Jacobi SVD of up to 160 sweeps. A lane past its chunk returned at once (`h >= range.end`). The
probe ran the lab job (`tests/fixtures/south-building-pair-triangulation.json`, 1672 tracks) on
the site at `cd97b62`.

Observed:

| Probe record           | Tracks                      | Workgroups                                     | Result                                               |
| ---------------------- | --------------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| `e0c75517`, `41d3ba90` | chunks of 16, 64, 256, 1024 | full                                           | completed, the longest in 5 ms                       |
| `e0c75517`, `41d3ba90` | 1360..1671 (312)            | 19 full, 1 with 8 active lanes and 8 idle ones | `DXGI_ERROR_DEVICE_HUNG` (`0x887A0006`)              |
| `02ed0bfc`             | 1360..1663 (304)            | 19 full                                        | completed in the three variants, 303 of 303 accepted |
| `14349a11`             | 1664..1671 (8)              | 1 with 8 idle lanes                            | `DXGI_ERROR_DEVICE_HUNG`                             |
| `e37f29b0`             | 1656..1671 (16)             | 1 full, holding the 8 tracks of `14349a11`     | completed in the three variants, 16 of 16 accepted   |

In `02ed0bfc` and `e37f29b0` the statuses were those of the CPU tier.

Observed, the earlier losses in `triangulateTracks` on this device. Each was a single dispatch
over all the tracks of a call or a chunk whose track count was not a multiple of 16:

| Record                       | Site      | Dispatch                       | Lost in             |
| ---------------------------- | --------- | ------------------------------ | ------------------- |
| `20261019095412032-e5e0f937` | `486df3b` | one dispatch over 1672 tracks  | `triangulation #2`  |
| `20261020012028865-b2ca931e` | `329043c` | one dispatch over all tracks   | `triangulation #11` |
| `20261020020739347-d7d25f91` | `a3041af` | chunks, lost after 8 completed | `triangulation #11` |
| `20261020021036188-ba3c638d` | `a3041af` | chunks, lost in the third      | `triangulation #11` |
| `20261020021122316-8f3753de` | `a3041af` | chunks, lost in the third      | `triangulation #2`  |
| `20261020022404777-4881d87d` | `a3041af` | chunks, lost after 8 completed | `triangulation #11` |
| `20261020022502483-57d0b9cd` | `a3041af` | chunks, `?bounded=1`           | `triangulation #11` |

Observed elsewhere: the CPU tier, SwiftShader (Vulkan) and the owner's iPhone (Metal) run the
same kernel with partial workgroups and lose no device. On this device the pair-model kernels
(`scorePairModels` and the others of `src/gpu/pair-models.shade.ts`) have the same guard. In every record of the
second table they verified all the pairs without a loss, 10 pairs in each standard benchmark.

Inference: the data of the tracks is not the cause, because the same 8 tracks complete inside a
full workgroup. The driver or the shader compiler mishandles a workgroup in which some lanes leave
the entry before the loops. Not every early return hangs, because the pair-model kernels do not.
The loop structure of the kernel matters, and which property matters is not known.

Inference, weak: the hang belongs to the Blackwell architecture under this driver, not to the
compiler's HLSL. The same kernel ran on an NVIDIA GeForce RTX 2080 on 2026-10-17 and lost no
device (`docs/experiments/gpu/sfm-cg-chunks-rtx2080.json`, `adapterArchitecture: 'turing'`).
The compiler's change 0073 record states that the kernel was unchanged from that run to
`e5e0f937`. That run dispatched the kernel once over all the tracks of a call, so its last
workgroup was almost certainly partial. A cause in Tint's HLSL or in DXC would have shown on
Turing too. So the hang is expected on every Blackwell GPU under the same driver. The inference
is weak because the experiment file does not record the browser or the backend of the 2080 run.

Remedy:

- Pull request #60 ran `triangulateTracks` as chunks of tracks under the time budget. It did not
  remove the hang (the records on `a3041af` above).
- Pull requests #64 and #65 added the probe page and its probe record, which found the
  workgroup.
- Branch `fix/triangulation-full-workgroups` (not merged): no lane returns for being past its
  chunk. An idle lane computes the chunk's last track with the same control flow and writes
  nothing. On the CPU tier and on SwiftShader the results match those of the kernel at
  `cd97b62` (`docs/gpu-sfm.md`, "Triangulation", on that branch).
- typeshade/typeshade#703 reports the hang to the compiler.

Open:

- The lab has not run the remedy on this device. That run decides whether the clamp-and-mask
  form completes where the early return hangs.
- The component at fault (Tint's HLSL, DXC or the NVIDIA driver) is not known.

### A2. The D3D12 compile of the emulated-`f64` kernels is long

Observed, under the `'float'` flavor with a cold pipeline cache:

| Record                       | Pair verification, 10 pairs | First chunk of `triangulateTracks` |
| ---------------------------- | --------------------------- | ---------------------------------- |
| `20261020020739347-d7d25f91` | 10.9 s                      | 8763 ms                            |
| `20261020022404777-4881d87d` | 16.0 s                      | 9095 ms                            |

Observed, a second run in the same browser: pair verification took 1265 ms
(`20261020012028865-b2ca931e`) and 1171 ms (`20261020021036188-ba3c638d`). In `ba3c638d` the
longest chunk of `triangulateTracks` took 31 ms.

Inference: the time is the D3D12 compile of the lowered `f64` kernels. The browser caches the
pipelines, so the second run pays no compile. The first timed chunk of a key measured the compile
as work, and the rate it taught the time budget was far too slow.

Remedy: pull request #63 makes the pipelines of a chunk key in an untimed warm-up plan of zero
workgroups, before the first timed chunk (`docs/gpu-sfm.md`, "The time budget").

Open: no record on this device has run with pull request #63.

### A3. The `'integer'` flavor did not reach the first dispatch of pair verification

Observed: `20261020022307924-efd20d5f` (`?fp64=integer`, 2 frames, site `a3041af`). The features
and the match ran in 0.5 s. `verifyPair #1` then ran 57 s, and no pair-model kernel finished. The
device was lost at `finishPairModels` with Dawn's "A valid external Instance reference no longer
exists", not a DXGI reason.

Inference, not measured: the GPU process ended under the browser's watchdog while the
integer-flavor kernels compiled.

Remedy: none is needed. `recommendFp64Flavor` picks the `'float'` flavor on this device (pull
request #47, change 0073), and that flavor works here.

Open: the cause of the loss is not measured.

### A4. After `DXGI_ERROR_DEVICE_HUNG` the page cannot open a new device

Observed: `requestDevice` on a new adapter fails with `D3D12 create command queue failed with
DXGI_ERROR_DEVICE_REMOVED (0x887A0005)` (probe records `e0c75517`, `41d3ba90`, `14349a11`). Only
a page reload ends it.

Remedy: pull request #66 (open) lets the probe reload the page after a device loss and go on.

Open: whether a reload gives a working adapter on this device is not tested. Pull request #66 has
not run there.

### A5. Pair verification passes under the time budget

Observed, on the site at `329043c` with the `'float'` flavor (`20261020012028865-b2ca931e`): the
10 pairs verified in 1265 ms, 330 submissions and 2166 dispatches. The key `scorePairModels` ran
90 timed chunks, the longest 4.1 ms, at 0.77 ms per unit, with a budget of 250 ms.

Observed, before the time budget (`20261019095412032-e5e0f937`, site `486df3b`): the one pair of
that run verified in 176 ms (3 submissions), and the device was lost in the triangulation after it
(A1).

Inference: the fences of 90 timed chunks cost more than the work of chunks of 4.1 ms
(`docs/gpu-sfm.md`, "The time budget").

Remedy: pull requests #49, #50, #53 and #54 run pair model generation as short dispatches under
the time budget, with no Jacobi sweep. Pull request #57 dispatches the batch loop indirectly
(change 0075). Pull request #63 stops timing a key after 8 chunks.

Open: no record on this device has run with pull request #63.

### A6. Bounded submissions do not prevent the hang of A1

Observed: `20261020022502483-57d0b9cd` (`?bounded=1`) ran 2166 submissions for 2166 dispatches in
pair verification. The device was then lost in `triangulateTracks`, as in the unbounded runs.

Remedy: none. Bounded submissions stay a diagnostic (`docs/lab-run-records.md`, "A lost
device").

## Device B: iPhone, Apple GPU

An iPhone with iOS 18.7, Safari 27.0.1, an Apple GPU (adapter vendor `apple`) on Metal and 4
hardware threads. The model of the iPhone is not recorded.

The flavor of the emulated double on this device is `'integer'`. `recommendFp64Flavor` picks it on
an Apple GPU (B1).

### B1. The `'float'` flavor collapses under Apple's compiler

Observed, on the site at `486df3b` (the compiler's change 0073 record):

| Record                       | What happened                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------------- |
| `20261019100719260-789ba764` | 10 pairs verified in 35.3 s (310 submissions). The device was lost in the fifth verification.     |
| `20261019100902196-f82a27f5` | The pyramid benchmark verified 22 pairs in 59.2 s (682 submissions), then no stable initial pair. |

On SwiftShader the same pairs verify in under 0.4 s each, and the sample registers eight of eight
views.

Inference (change 0073's record): Metal's shader compiler reassociates the error terms of the
`'float'` flavor and collapses it to `f32` precision. A RANSAC whose scores lose their precision
neither stops early nor finds the right model.

Remedy: pull request #47 moves the pin to the compiler's change 0073 and picks the flavor at
`ShadeDevice.open`: `'integer'` on an Apple GPU, `'float'` elsewhere. Under the `'integer'`
flavor the iPhone registered the pyramid sample's eight views (`20261019140521190-07c4d3e5`,
`20261020020913972-527f815b`).

### B2. One long dispatch loses the device, and Safari gives no reason

Observed:

- Before the time budget, on 2026-10-19, every unbounded run lost the device during pair
  verification, under either flavor.
- `20261019140521190-07c4d3e5`, the first bounded run under the `'integer'` flavor, registered
  the eight views (691 landmarks, 0.400 px, 116 s).
- `20261019140528615-057b3152`, a bounded run, lost the device in one long dispatch
  (`sweepPairModelSvd`).
- `20261019225221735-85ed0ae6` (site `7eef646`, unbounded, 22:52 UTC) passed pair verification
  for all 22 pairs and registered the eight views (640 landmarks, 1979 observations) in 40.2 s.
  It then lost the device in `bundle #25`, the final bundle adjustment.

Inference, from the code and from the kernels that `85ed0ae6` names first in that bundle: the loss
moved to the one long dispatch left on that path, `bundlePersistentCg` of
`src/gpu/joint-focal.shade.ts`. That kernel ran the whole PCG loop of an LM iteration in one
workgroup.

Remedy:

- Pull requests #49, #50, #53 and #54: pair model generation as short dispatches under the time
  budget. The fixed-cost null-vector solve removes the Jacobi SVD of `sweepPairModelSvd`.
- Pull request #56: the fixed-focal BA runs its PCG loop as chunk steps.
- Pull request #58: the shared-focal BA and the Schur cache run their PCG loops as chunk steps
  (`docs/gpu-sfm.md`, "The BA's PCG loop").

Open: none from the records. The loss of `85ed0ae6` did not recur (B3).

### B3. Under the time budget the whole pipeline completes

Observed, on the site at `a3041af`, the `'integer'` flavor, unbounded submissions, the standard
benchmark:

| Measure                 | Cold, `20261020020913972-527f815b` | Warm, `20261020020957239-1d174b95` |
| ----------------------- | ---------------------------------- | ---------------------------------- |
| Total                   | 57.8 s                             | 41.9 s                             |
| Pair verification       | 24.4 s (22 pairs)                  | 21.7 s                             |
| Bundle adjustment       | 24.1 s                             | 15.1 s                             |
| Triangulation           | 4.7 s                              | 1.2 s                              |
| Registered views        | 8 (640 landmarks, 1979 obs.)       | 8                                  |
| Mean reprojection error | 0.520 px                           | 0.520 px                           |

Observed, the chunk keys of the cold pass:

| Key                 | Timed chunks | Longest | Rate               |
| ------------------- | ------------ | ------- | ------------------ |
| `scorePairModels`   | 216          | 7 ms    | 1.73 ms per unit   |
| `triangulateTracks` | 29           | 3011 ms | 2.33 ms per unit   |
| `bundleCg`          | 67           | 953 ms  | 0.0136 ms per unit |

The last chunk of `bundleCg` held 12198 units.

Inference: the longest chunk of `triangulateTracks` and of `bundleCg` was the first chunk of the
key, and it held the compile of the key's pipelines.

Remedy: pull request #63 makes those pipelines in an untimed warm-up plan.

Open: no record on this device has run with pull request #63.

### B4. Timed chunks cost two fences each

Observed: in `527f815b` pair verification ran 216 timed chunks. The longest took 7 ms, against a
budget of 250 ms, so the budget never limited a chunk.

Inference: the fences, two for each timed chunk, bound the time of pair verification.

Remedy: pull request #63 times the first 8 chunks of each key and runs the later ones untimed,
with no fence (`CHUNK_TIMING_SAMPLES` in `src/gpu/device.ts`).

Open: no record on this device has run with pull request #63.

### B5. The chunk rates persist for the life of the device

Observed: the warm record `1d174b95` shows 432, 51 and 132 chunks for the three keys of B3, with
the same longest values as the cold record `527f815b`.

Inference: the two passes of the standard benchmark share one device, so the warm record's
`chunks` and `longestMs` include the cold pass.

Remedy: none in the code. `docs/lab-run-records.md` ("What a record holds") states how to read a
warm record.

### B6. After a device loss Safari may drop the WebGL context

Observed (2026-10-19): after a device loss, the 3D viewer failed to initialize on the next run.

Remedy: pull request #48. The run goes on without the viewer. The map stays empty until the page
is reloaded.

## What the compiler was told

| Item                                                                         | State at the pin `d70b539` | Observations                                                     |
| ---------------------------------------------------------------------------- | -------------------------- | ---------------------------------------------------------------- |
| Change 0073, "A host picks the df64 flavor of its kernels at run time"       | `implemented`              | B1 (`789ba764`, `f82a27f5`) and the Windows loss `e5e0f937` (A1) |
| Change 0074, "A device runs long kernels as chunks under a time budget"      | `accepted`                 | B2 (the records of 2026-10-19, then `85ed0ae6` on `7eef646`)     |
| [typeshade/typeshade#703](https://github.com/typeshade/typeshade/issues/703) | open                       | A1, with A2, A3 and A4                                           |

Change 0073's record lists the three failure records of 2026-10-19 morning. It states that the
Windows loss is not explained by the flavor and needs its own cause, which A1 found.

Change 0074's record states its hardware item as observed in part. The iPhone ran pair model
generation as chunks in `85ed0ae6` and lost no device in pair verification. At that time the
Windows computer had not run.

Issue #703 asks the compiler's owner for three decisions:

1. A rule for kernel authors: in an entry with heavy loops, clamp an out-of-range lane's index and
   mask its writes, and do not return before the loops.
2. Or a lowering that rewrites the early-return guard of a compute entry into that form, which
   needs a proposal in the compiler's `changes/`.
3. A document in the compiler's tree that keeps the hardware observations of the downstream labs.

Proposal, not filed: the first timed chunk of a key on a fresh device includes the creation of its
pipelines (A2, B3). A `device.fuse(body, { chunk })` that measures the work without the pipeline
creation would be a change to the compiler (`docs/gpu-sfm.md`, "The time budget").
