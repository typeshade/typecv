# Physical-GPU runs owed

This page lists the measurements that only a physical GPU can make. The work of 2026-10-20 ran in
a container with no physical GPU: every time in its records comes from SwiftShader, a software
GPU. `docs/lab-run-records.md` ("SwiftShader against the RTX 2080") gives the ratio of each stage
between the two, so a SwiftShader time is an estimate of a physical one, not a measurement.

Each run below is a visit to the lab with a query string. The lab is `https://cv.typeshade.dev/lab/`.
A run makes a run record. Send it with the lab's "Send" button, or save it with "Download JSON".

## Procedure

1. Open the lab on the device. Use Chrome on Windows (the RTX 2080 or the RTX 5060 Ti), and Safari
   on the iPhone where a row says so.
2. Open the address of a row, with its query string.
3. Load the input of the row, then start the run.
4. Wait for the run to finish. Do not switch tabs during the run.
5. Send the record, or download it.
6. Do the next row.

Run each row twice when the device was cold: the first pass compiles the kernels. The second pass
(warm) is the one that compares with the records.

## Rows

| #   | Change measured                                       | Address (query)                                | Input                                      | Compare                                                              |
| --- | ----------------------------------------------------- | ---------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------- |
| 1   | Bundle schedule (#113)                                | `?`                                            | "Standard benchmark"                       | Bundle stage against record `944372ed` (10.4 s). Estimate: 6 to 8 s. |
| 2   | Video decode in a worker (#111)                       | `?` and `?decode=main`                         | An uploaded video (the 566-frame 4K clip)  | `timing.mainThread` long tasks and frame gaps between the two runs   |
| 3   | Phase-correlation keyframe gate (#116)                | `?` and `?gate=legacy`                         | An uploaded video with fast panning        | Kept frames, registered views, total time                            |
| 4   | CLAHE before SIFT (#114)                              | `?clahe=1` and `?`                             | The pyramid video sample                   | Features per frame, landmarks, feature stage time                    |
| 5   | Dense cleanup kNN (#129)                              | `?dense=1`                                     | COLMAP gerrard-hall                        | `dense.clean.ms.outliers` (SwiftShader 2.7 to 5.4 s, CPU 2.4 s)      |
| 6   | Dense mesh (#119)                                     | `?dense=1&denseMesh=1`                         | COLMAP gerrard-hall                        | `dense.mesh` ms per stage; look at the mesh with the "Mesh" switch   |
| 7   | Dense MVS at full size, fused points (#112 open item) | `?dense=1`                                     | TUM fr3 sample, 640 px edge                | The run finishes without a device loss                               |
| 8   | KLT tracking (#126)                                   | `?klt=1` and `?klt=0`                          | An uploaded video                          | `kltPush` time (SwiftShader 144 s for 41 frames), track length       |
| 9   | ORB front end (#123, #127)                            | `?features=orb` and `?`                        | COLMAP south-building (12) and the pyramid | Registered views, feature and matching stage times                   |
| 10  | Hough lines (#125, #128)                              | none in the lab; `scripts/benchmark-hough.mjs` | south-building frame 0                     | WebGPU total against the CPU reference (55 to 67 ms)                 |
| 11  | Automatic focal (#162)                                | `?` (focal "Auto")                             | The pyramid video sample                   | Time to the first map, `result.focalSearch`, against `4a8aa2ff`      |

## Notes

- Row 1 also gives the physical rate of every other stage of the standard benchmark. Add the record
  next to `944372ed` in `docs/lab-run-records.md` to update the ratio table.
- Row 7 is the run that crashed the headless browser after 55 minutes in the container. A device
  loss on a physical GPU is a finding: send the record with its `failure` block.
- Row 10 needs a checkout and `npm ci`. Run `node scripts/benchmark-hough.mjs` in Chrome's WebGPU
  through Playwright with the default adapter.
- KLT step 7 (`docs/klt-tracking.md`) is the physical-device step of the KLT record. Row 8 is its
  first measurement.
