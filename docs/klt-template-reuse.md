# Reusing KLT template samples

This stage compares against `1af3a5868d5762b36e193f366685c5e6bd638519`, after
the shared work-elision/read-overlap and compact-output changes. The retained
baseline GPU bundle has SHA256
`75c2b44fb281052449e8e3c77eeffcc3b7d141ae046739ec39145cf4516dcae8`.

- [x] T1: Audit repeated template work and existing numeric/runtime contracts.
- [ ] T2: Reuse bounded template samples without changing image measurements.
- [ ] T3: Compare full KLT outputs against the unchanged exhaustive f32 reference.
- [ ] T4: Measure repeated ordinary RGB workloads and separate GPU profiles.
- [ ] T5: Record default decision, memory limits and final verification.

## Existing implementation and bounded change

`src/gpu/klt.shade.ts::kltTrack` already calculates the source Hessian and
mean/variance once per pyramid level. The repeated work is bilinear source
sampling inside each residual iteration. Target samples, displacement, target
mean/variance and photometric gain change each iteration and cannot be frozen.

The first candidate retains source intensity in a 441-element invocation-local
array, filled by the existing template statistics loop. Each level overwrites
it in the original row order. Residuals and final level-zero NCC read those same
measured samples. Half-windows through10 fit; larger windows use the existing
sampler without truncation. Forward and backward invocations fill their own
templates. The exhaustive f32 oracle in `src/gpu/klt.ts` remains unchanged.

For a P-pixel window and T executed iterations at one level, source analysis
counts approximately P(12+20T) global float reads before reuse and P(12+16T)
after intensity reuse. Reusing the final source patch in NCC avoids another8P
reads per direction. These are source-derived upper operation counts, not
measured physical transactions or a speed guarantee.

The local array holds1,764 bytes per active invocation. It adds no resident
binding, dispatch or readback, and uses no workgroup memory/barrier. Private
register/spill storage is outside application allocator statistics and may
reduce occupancy on a physical GPU. Even oversized-window fallback can carry
private-array compiler overhead; unchanged sampling does not establish equal
performance. Hardware performance is not inferred from SwiftShader.

## Validation and decision criteria

Keep bilinear arithmetic, accumulation order, residual grouping, affine target
warps, status and iteration counts. Do not replace centered NCC covariance with
an algebraically equivalent variance formula that changes f32 rounding.

Existing chunk/affine/brightness/end-state/readback/cache tests remain in place.
Additional cases cover odd image dimensions, fractional positions, pyramid
border clamping, the cache boundary10/11 and a large-window fallback. Compare
all forward/backward/checked values to the unchanged reference, including ended
tracks rather than only accepted reconstruction observations.

Performance comparison uses the production32MiB pool and stateless policy in
both modes, three fresh-browser repeats with alternating order, all six existing
RGB cases and exact accepted-measurement/model/cohort checks. Profiling runs stay
separate. Default application depends on measured benefit, regressions and memory
limits; the initial candidate alone does not establish acceleration.
