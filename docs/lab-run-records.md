# Lab run records

The SfM lab at `/lab/` can send the measurement of one run to the site's collector,
so that runs on hardware the maintainers do not own can be compared. Sending is a
choice the visitor makes with the consent checkbox. Nothing is sent without it. With
the checkbox selected before a run, the record is sent as soon as the run completes
or fails; selected after a run, the **Send record** button sends it.

## What a record holds

`website/src/lib/run-record.ts` defines the record (`typecv-run-record-v2`) and its
validator. One record is one run, completed or failed:

| Field       | Content                                                                                                                                |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `site`      | The site commit (`/version.json`) and the runtime path                                                                                 |
| `client`    | Browser user agent, platform, language, core count, device memory class, screen size and DPR                                           |
| `gpu`       | WebGPU adapter vendor, architecture, device, description, fallback flag, two buffer limits                                             |
| `input`     | Kind (sample, prepared JSON, video), frame count, analysis and source resolution, duration                                             |
| `options`   | The front end (`corner` or `sift`), the lab's numerical controls, the assumed focal ratio and the submission mode                      |
| `result`    | Registered views, landmarks, observations, mean reprojection error, matched pairs, features                                            |
| `timing`    | Total wall time, device reuse (cold or warm), per-operation totals, per-frame times, the GPU kernels in first-use order                |
| `transfers` | Application upload and readback bytes                                                                                                  |
| `benchmark` | `pyramid-8` and `cold` or `warm` when the standard benchmark produced the run, else null                                               |
| `reference` | Agreement with a sample's reference cameras (below), else null                                                                         |
| `failure`   | Null for a completed run; else the error, the operation, the kernel that started last, the device-loss reason and the frames processed |
| `note`      | Up to 500 characters the visitor typed                                                                                                 |

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
too long), the page reports the operation and the kernel that started last, and builds
a failure record with everything measured before the loss: the counts of the last map
snapshot, the per-operation totals, the kernels in first-use order and the `failure`
field. With consent given before the run, the record is sent at once. Open the lab
with `?bounded=1` to run the same job with fenced submissions: the worker waits for the
queue after each group of dispatches and verifies eight models per batch
(`ShadeDevice.open({ boundedSubmissions: true, maxModelsPerBatch: 8 })`). The run is
slower. The record's `options.boundedSubmissions` says which mode ran.

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
device reuse, benchmark name, and for a failed run the kernel or operation). `website/worker.test.mjs` tests the handler with an
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

The **Export lab run records** workflow (`.github/workflows/lab-records.yml`, run by
hand from the Actions tab or by an agent through the API) reads the KV namespace with
the deploy's `CLOUDFLARE_API_TOKEN` (`scripts/export-run-records.mjs`, the Cloudflare
REST API) and needs no read token. It prints one summary line per record to the job
log, prints the failed runs in full (`print: all` prints every record), and uploads the
JSON files as the `lab-records` artifact for 14 days. The inputs are `limit` (newest
records, default 50) and `print` (`failures`, `all` or `none`).

With a `RECORDS_READ_TOKEN` secret, the collector's read routes answer too:

```sh
curl -H "Authorization: Bearer $TOKEN" 'https://cv.typeshade.dev/api/records?limit=50'
curl -H "Authorization: Bearer $TOKEN" 'https://cv.typeshade.dev/api/records/<id>'
```

Or on the Cloudflare dashboard: Workers & Pages → KV → `typecv-site-typecv-records`.
