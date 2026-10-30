# Compiler adoption: complete sparse-pipeline comparison

This record compares the TypeShade compiler pins `725b2969a9cac37400980c2c0058680368407602`
and `49f4f2884c0c38dc6134c837efd5565e5395b721` on TypeCV's library source at
`451e12ccea28397fcdec0594f1194268a57f814a`. The latter includes
[TypeShade #839](https://github.com/typeshade/typeshade/pull/839). This is a larger pin change
than the last incremental compiler benchmark: percentages from those experiments must not
be added to these results.

## Result

**These measurements do not establish a repeatable end-to-end speedup.** The new compiler
preserves the tested outputs, but these pipeline timings do not justify describing the pin
as a TypeCV application performance improvement. A small regression is not ruled out.

Two fresh browser sessions per workload each contain one warmup pair and five measured
pairs. B reverses the initial order. Times below are milliseconds; each time is the median
for that version, while the final column is the median of the five paired changes.

| Pipeline/session  | Baseline median | Candidate median | Median paired change |
| ----------------- | --------------: | ---------------: | -------------------: |
| SIFT A, 3 images  |        16,033.4 |         16,095.4 |               +0.07% |
| SIFT B, 3 images  |        15,957.2 |         15,883.3 |               −0.79% |
| BRIEF A, 8 images |         2,456.4 |          2,440.1 |               +0.38% |
| BRIEF B, 8 images |         2,381.1 |          2,412.2 |               +1.29% |

The SIFT mapper-stage paired medians are −3.55% and +0.21%; the BRIEF mapper-stage
medians are −3.56% and +0.27%. The stage-level gains also fail to repeat in the reversed
session. BRIEF feature-stage medians are +1.30% and +1.70%, which should be retained as
a possible regression signal, not hidden by a favorable mapper sample.

The identical-compiler controls use three measured pairs each. Their paired changes are
**−1.42% to +1.56%** for SIFT (median +0.20%) and **−3.17% to +2.71%** for BRIEF
(median +0.60%). These short controls demonstrate measurement variation; they are not
confidence bounds or proof that a small regression cannot exist.

All **26 measured pairs and six warmup pairs** have identical outputs. Each workload's
complete output also agrees across every session, including its control. No WebGPU validation
or page error was observed. Host transfer counters are unchanged in every run:

| Pipeline | Application upload bytes | Application readback bytes |
| -------- | -----------------------: | -------------------------: |
| SIFT     |                5,292,144 |                    243,988 |
| BRIEF    |                8,090,956 |                     62,848 |

Raw receipts: SIFT [A](experiments/compiler-performance/sift-a.json),
[B](experiments/compiler-performance/sift-b.json),
[control](experiments/compiler-performance/sift-control.json); BRIEF
[A](experiments/compiler-performance/brief-a.json),
[B](experiments/compiler-performance/brief-b.json),
[control](experiments/compiler-performance/brief-control.json).

Measured on 2026-10-30 with Node 24.19.0 and Chromium 156.0.8078.4, Linux x64,
SwiftShader, and a four-CPU container quota on an AMD EPYC 9V74 host. No other local
test or benchmark ran concurrently with the recorded measurements. Early fixture-setup
smoke runs are excluded.

## Method

`scripts/benchmark-compiler-pipeline.mjs` serves the normal built JavaScript chunks from two
independently installed and built checkouts. It checks identical library source trees, build
configuration, other dependency versions and input JPEG bytes before running. Every receipt
records the compiler URLs, input and runtime SHA-256 hashes, script hash, browser identity,
machine, options, individual timings and output hashes.

The timed interval starts with the decoded RGB frames and ends after feature extraction,
descriptor matching, native incremental SfM and a GPU completion fence. Ordinary uploads,
readbacks and host orchestration are included. JPEG decoding and module import are reported
separately; diagnostic output hashing and disposal are excluded. This measures the complete
sparse library pipeline for these inputs. It does not measure the browser application, video
decoding, dense MVS, network loading or a physical GPU.

Both versions run on one WebGPU device backed by SwiftShader, with separate TypeShade device
owners and buffer pooling disabled. Each fresh browser session warms each version once,
then alternates their execution order in paired rounds. The reversed session starts in the
opposite order. The identical-compiler control serves the baseline build under both labels.
First calls share the browser/device environment and are not isolated startup measurements.

The inputs are the attributed, committed 640 × 360 pyramid images in
`tests/fixtures/lab-images/pyramid/`. Calibration and the fixed random seed follow the existing
pyramid benchmark and library defaults. Both versions use the same lab mapper controls:
500 RANSAC iterations, 3 px error, 1° initial and 0.5° triangulation angle, four local BA
iterations and eight final BA iterations, with shared focal refinement. No solver quality
threshold is relaxed to make the comparison pass.

For every warmup and measured pair, the script requires exactly equal keypoint, descriptor,
match and complete reconstruction hashes, registration decisions, reprojection error and
GPU BA counts. A mismatch is written into the receipt and fails the command. This comparison
complements the independent CPU-oracle and base/head digest checks in CI (Rules 1.2, 1.4
and 2.1); agreement between two versions alone is not an oracle.

## Build cost and affected code

The public `dist/gpu/` JavaScript chunks grow from **6,914,179 to 7,272,198 bytes**
(+5.18%), or **1,137,390 to 1,187,694 bytes** when each file is gzip-compressed at level 9
(+4.42%, 50,304 bytes). The receipts contain every file's size and hash. These are the public
GPU chunks, including the dynamically loaded fusion chunk, not an application's tree-shaken
download. The benchmark excludes network loading and cannot offset this cost with a loading
speed claim.

Of 30 kernel modules, the recorded shader-text digests change for 21 WGSL modules, 11
integer-fp64 WGSL variants and nine GLSL records. Entry and program counts are unchanged.
`tests/digests/kernels.json` is renewed because it records emitted text and the compiler pin.
Numerical output fixtures and `tests/digests/expected-changes.json` are not changed.

The recorded WGSL of `sift.shade.ts` and `float-matching.shade.ts` is unchanged. In the first
SIFT session, feature extraction and matching account for about 96% of baseline wall time;
the mapper accounts for about 4%. This explains why a targeted BA-kernel gain need not give
a measurable whole-pipeline gain here. The static digests do not describe every shader that
the runtime may fuse, and stage timings include both host and device work.

The BRIEF configuration retains 400 features at most per image. On this fixture and preset,
both versions register only views 2 and 4 of the eight inputs and produce 14 landmarks. It is
a limited reconstruction case, not evidence of successful eight-view reconstruction. The SIFT
configuration retains 4,096 features at most per image and registers all three input views,
with 535 landmarks. Both runs use the same acceptance thresholds before and after the pin.

Application upload/readback counters describe host transfers, not GPU memory bandwidth.
No reduction in physical memory traffic, register pressure or native GPU call overhead is
claimed from these measurements.

## Correctness checks

- Local library CPU tests: 477 pass; renewed compiler capability probes: 43 pass;
  declaration/API/shader surface checks: six pass.
- [CI run 41013384482](https://github.com/typeshade/typecv/actions/runs/41013384482),
  on the compiler-pin commit `2761e824`: all three SwiftShader shards pass, with
  128 tests passing and two native integer-fp64 cases skipped. The TypeShade audit passes.
- In that run, the independent base/head digest comparisons on Node 22 and 24 both
  report zero differing cases and zero missing cases. Node 24 uses the repository's
  existing `--single-threaded` setting for reproducibility. No expected-output exception
  is added for this pin.
- The PR's required checks still gate adoption of its final commit. The shader-text
  receipt update records the new compiler; the numerical fixtures retain their old bits.

## Reproduction

Use Node 24, install dependencies with `npm ci` and run `npm run build` in both checkouts.
The baseline checkout is TypeCV `451e12c`; the candidate uses the new compiler pin. Keep other
CPU/GPU workloads off the measurement machine. Set `TYPESHADE_CHROMIUM` when Chromium is not
in Playwright's default location.

```sh
node scripts/benchmark-compiler-pipeline.mjs \
  --baseline /path/to/baseline --output /path/to/sift-a.json \
  --features sift --frames 0,3,7 --pairs 5
node scripts/benchmark-compiler-pipeline.mjs \
  --baseline /path/to/baseline --output /path/to/sift-b.json \
  --features sift --frames 0,3,7 --pairs 5 --reverse
node scripts/benchmark-compiler-pipeline.mjs \
  --baseline /path/to/baseline --output /path/to/sift-control.json \
  --features sift --frames 0,3,7 --pairs 3 --control
```

For BRIEF, repeat the three commands with `--features brief --frames 0,1,2,3,4,5,6,7`
and separate output filenames. Its feature limit is 400; SIFT's is 4,096. The script fixes
matching and mapper options identically for both versions and records them through the
script hash and mapper controls.

Each summary reports the median baseline and candidate time separately, and the median of
the paired percentage changes `100 × (candidate / baseline − 1)`. These are different
statistics: the percentage is not calculated by dividing the two displayed medians.
Negative percentages indicate lower elapsed time. The individual samples and identical-build
control must be considered before attributing a small change to the compiler.
