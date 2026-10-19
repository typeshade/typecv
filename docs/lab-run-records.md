# Lab run records

The SfM lab at `/lab/` can send the measurement of one run to the site's collector,
so that runs on hardware the maintainers do not own can be compared. Sending is a
choice the visitor makes for each record. Nothing is sent without the checkbox.

## What a record holds

`website/src/lib/run-record.ts` defines the record and its validator. One record is
one completed run:

| Field       | Content                                                                                      |
| ----------- | -------------------------------------------------------------------------------------------- |
| `site`      | The site commit (`/version.json`) and the runtime path                                       |
| `client`    | Browser user agent, platform, language, core count, device memory class, screen size and DPR |
| `gpu`       | WebGPU adapter vendor, architecture, device, description, fallback flag, two buffer limits   |
| `input`     | Kind (sample, prepared JSON, video), frame count, analysis and source resolution, duration   |
| `options`   | The front end (`corner` or `sift`), the lab's numerical controls and the assumed focal ratio |
| `result`    | Registered views, landmarks, observations, mean reprojection error, matched pairs, features  |
| `timing`    | Total wall time, device reuse (cold or warm), per-operation totals, per-frame times          |
| `transfers` | Application upload and readback bytes                                                        |
| `benchmark` | `pyramid-8` and `cold` or `warm` when the standard benchmark produced the run, else null     |
| `reference` | Agreement with a sample's reference cameras (below), else null                               |
| `note`      | Up to 500 characters the visitor typed                                                       |

A record holds no video, image, file name or network address. The page shows the
exact JSON before it is sent, and offers the same JSON for download or copy when the
visitor prefers to share it another way. The collector adds `server.id`,
`server.receivedAt`, the Cloudflare country code and colo. The sender's address is
hashed with the hour for rate limiting only and is not stored.

## The standard benchmark

The lab's **Standard benchmark** button loads the eight-frame pyramid sample and runs
it twice in one worker: a cold pass that includes shader compilation, then a warm pass
on the retained device. Each pass produces one record with `benchmark.pass` set. With
the consent box checked before the benchmark starts, both records are sent as they
complete.

## The COLMAP sample

The lab's two **COLMAP south-building** buttons load twelve photographs of the
[COLMAP demo dataset](https://demuc.de/colmap/datasets/) south-building (image credit
Christopher Zach), resized to 640 × 480, with the dataset's refined intrinsics scaled to
that size and the poses of its bundled sparse model
(`website/public/sample/south-building/`, derived by
`scripts/prepare-lab-colmap-sample.py`). The first button runs the whole GPU pipeline
on the photographs with the lab's front end, scale-space SIFT
(`extractSiftGpuResident`) and the tiled L2 matcher with a 0.8 ratio test, and the
calibrated intrinsics held fixed (`PHOTO_PROFILE` in `sfm.worker.ts`). The second feeds the dataset's own SIFT keypoints and raw matches
(`colmap-sift.bin`, the matched keypoints only) to the GPU mapper, so the mapper is
measured apart from the lab's feature front end.

After each snapshot and at the end, the page similarity-aligns the registered cameras
to the COLMAP poses on their common views (`website/src/lib/pose-alignment.ts`, the
method of `colmap_dataset.py compare`) and shows the common view count, the median and
maximum rotation difference in degrees, and the median and maximum camera center
difference as a fraction of the reference trajectory's radius. The record's `reference`
field carries the same numbers. The COLMAP model is a reference, not ground truth: both
systems hold the same intrinsics fixed, and with the second button they also share the
same features and matches.

## A lost device

When the browser loses the WebGPU device during a run (on Windows,
`DXGI_ERROR_DEVICE_HUNG` is the GPU watchdog resetting a device whose command stream ran
too long), the page reports the operation and the kernel that started last. Open the lab
with `?bounded=1` to run the same job with fenced submissions: the worker waits for the
queue after each group of dispatches and verifies eight models per batch
(`ShadeDevice.open({ boundedSubmissions: true, maxModelsPerBatch: 8 })`). The run is
slower. Report the kernel named in the message and whether the bounded run completes.

## Collector

`website/worker.mjs` runs in front of the static assets for `/api/*` only
(`wrangler.jsonc`: `run_worker_first`). Routes:

| Route                   | Behavior                                                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| `POST /api/records`     | Same-origin only; body at most 256 KiB; validated; 30 per address per hour; stored in KV                       |
| `GET /api/records`      | `Authorization: Bearer <RECORDS_READ_TOKEN>`; lists `limit` (at most 100) records with metadata and a `cursor` |
| `GET /api/records/<id>` | Same token; returns one stored record                                                                          |

Records live in the KV namespace bound as `RECORDS` under `record/<id>`, with
listing metadata (received time, GPU, input kind, frames, registered views, total ms,
device reuse, benchmark name). `website/worker.test.mjs` tests the handler with an
in-memory KV.

## Setup

The deploy workflow (`.github/workflows/deploy.yml`) runs `scripts/ensure-run-records.mjs`
before `wrangler deploy`. The script lists the account's KV namespaces with the deploy
token, creates `typecv-records` when it is missing, writes the namespace id into
`wrangler.jsonc` for that deploy, and stores the `RECORDS_READ_TOKEN` Actions secret as
the Worker's secret when the repository has one. The deploy token
(`CLOUDFLARE_API_TOKEN`) needs **Workers KV Storage: Edit** for this; without it the
script removes the binding, warns, and the site deploys with the collector answering 503. `scripts/setup-run-records.ps1` does the same steps by hand from the owner's
machine. The read routes stay closed until a `RECORDS_READ_TOKEN` secret exists.

## Reading the records

```sh
curl -H "Authorization: Bearer $TOKEN" 'https://cv.typeshade.dev/api/records?limit=50'
curl -H "Authorization: Bearer $TOKEN" 'https://cv.typeshade.dev/api/records/<id>'
```

Or on the Cloudflare dashboard: Workers & Pages → KV → `typecv-site-typecv-records`.
