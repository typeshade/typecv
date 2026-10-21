# TypeCV design rules

Status: §1 to §3 accepted on 2026-10-21, by the owner's go-ahead in the session. §4 is a
proposal, and §5 lists its open decisions. A rule is numbered `Rule N.M` so that an issue, a pull
request or a code comment can cite it. "Must", "must not", "should" and "may" are read as in
RFC 2119.

This document governs what a user of TypeCV can write and what a call gives back. The
architecture (`architecture.md`) and the roadmap (`roadmap.md`) describe the parts and their
order. Where they disagree with a rule here, they are the ones to fix.

## 1. The first principle: one answer, everywhere, and proved

**Rule 1.1.** One call of TypeCV, on the same input, must give the same answer on every tier that
runs it: WebGPU, WebGL2 and the CPU tier, on any device.

- Rationale: every other property of a vision library depends on this one. An answer that changes
  with the device cannot be reproduced, tested or debugged, and an easier API only reaches a
  wrong answer faster.
- Evidence: the measurements of 2026-10-19 (`engine-v2.md`, "What the measurements say"). The
  eight-frame pyramid sample registered 8 of 8 views in one container and 5 of 8 on the CI runner,
  with identical features and pair inliers. PnP found 17 to 19 inliers on one machine and fewer
  than 12 on the other, on the same 29 correspondences. Emulated `f64` on `f32` hardware differs
  across instruction sets, and each thresholded decision amplifies the difference.
- Derives from: TypeShade's one IR, three backends and CPU oracle (TypeShade `AGENTS.md`; its
  design rules, §1.6 "Relation to the CPU oracle" and Rule 11.13).

**Rule 1.2.** An operation is proved before it is offered: a test holds its result on each tier to
the CPU oracle on the same input, bit for bit where the operation has one answer, and within a
stated bound where it does not. "It runs" is not proof.

- Rationale: a claim of Rule 1.1 that no test makes is not a claim.

**Rule 1.3.** An operation whose answer the platform does not fix must say so: in its
documentation, in the type or the options of its call, and in the determinism report of the
TypeShade program it runs. Texture filtering, a floating-point atomic, and an output whose order
depends on scheduling are such operations.

- Rationale: where one answer is not possible, the user decides with the difference in view, not
  after it has cost an experiment.
- Example: TypeShade change 0077 gives the CPU tier `textureLoad` and not `textureSample`, since
  WebGPU does not fix the precision of filtering.

**Rule 1.4.** A fused or lazily scheduled pipeline must give the bits that the same calls give run
apart (TypeShade Rule 11.13). A convenience that changes a result is not allowed.

## 2. The order of priorities

**Rule 2.1.** When two goals conflict, the earlier one in this list wins:

1. One answer everywhere, and proved (§1).
2. Where one answer is not possible, a stated difference (Rule 1.3).
3. Ease of use (§4).
4. Speed.

- Rationale: a fast or easy call that gives a device-dependent answer fails the user at the
  point that matters most. Speed that keeps the answer can come later. An answer lost for speed
  cannot.

**Rule 2.2.** A pull request that trades an earlier goal for a later one must say so, cite the
rule, and wait for the owner's decision.

## 3. Composition with the TypeShade family

**Rule 3.1.** TypeCV must run on a TypeShade device owner (`createDevice()` of
`typeshade/runtime`). It may open one, and it must take one that the host hands in, so that
TypeCV, TypeNN and Radiance can share one device. A device that the host hands in is never
disposed by TypeCV.

**Rule 3.2.** Data crosses between the libraries as TypeShade's own values: a `Texture` for an
image, a `Resident` for an array. On one device a crossing makes no copy. TypeCV adds no device
layer and no resource type of its own at that boundary.

**Rule 3.3.** What TypeShade cannot do yet becomes a proposal in TypeShade's `changes/`, not a
workaround inside TypeCV. TypeShade change 0077 (a kernel function that writes a texture) is the
first one.

## 4. The front end (proposed)

This section is a proposal. It shows the shape the front end takes under §1 to §3. §5 lists the
decisions it still needs.

**4.1. One namespace.** A user imports the library as one namespace, as three.js is imported:

```ts
import * as cv from '@typeshade/cv';
```

Every operation, class and alias is a member of it, so an editor lists the whole API after
`cv.`, and a bundler removes what the program does not use.

**4.2. No setup before the first call.** The device opens on first use. A host that shares a
device with another library sets it once, with `cv.setDevice(device)`. Only a read of a result is
awaited.

**4.3. An image is a value, and a chain of operations is one plan.** `cv.image(source)` takes any
image the web platform has: an `<img>`, a `<video>`, a canvas, an `ImageBitmap`, a `VideoFrame`
or an `ImageData`. Each operation returns a new image and changes none. Nothing runs until a
result is read. The chain then runs as one TypeShade plan, which gives the bits the calls give run
apart (Rule 1.4).

```ts
const edges = cv.image(photo).gray().blur(5).canny(50, 150);
await edges.draw(canvas);
```

**4.4. Configured algorithms are classes.** An algorithm with settings is a class, as a three.js
material is:

```ts
const sift = new cv.SIFT({ features: 4096 });
const [a, b] = await Promise.all([sift.detect(left), sift.detect(right)]);
const H = await a.match(b).ratio(0.75).homography();
```

**4.5. Video is a stream.** `cv.camera()` and `cv.frames(video)` give frames to a `for await`
loop. `cv.pipeline(f)` plans `f` once and runs the same plan on each frame.

**4.6. Results read as JavaScript.** `await img.pixels()` gives a typed array. A set of keypoints
has a `length` and iterates as objects with `x` and `y`. `img.texture` and `points.resident` give
the TypeShade values for another library (Rule 3.2), and `cv.from(value)` takes them back.

**4.7. Names come from OpenCV.** An operation is named as OpenCV names it, in camelCase on an
image (`gray`, `blur`, `canny`, `warpPerspective`). The OpenCV spellings (`cv.cvtColor`,
`cv.GaussianBlur`, `cv.findHomography`) are aliases in the same namespace, which build the same
plan.

## 5. Open decisions

1. **The package path.** `@typeshade/cv` (recommended), `typeshade/cv` as a subpath of the
   compiler's package, or `typecv`. A subpath of the compiler's package conflicts with TypeShade's
   decision that an engine lives in its own repository (#335), and needs a TypeShade proposal
   first. npm refused the unscoped name `typenn` as too similar to an existing package
   (GitHub Support, 2026-10), which argues for one `@typeshade` scope for the family.
2. **Channels in the type.** `Img<'rgba8'>`, `Img<'r8'>` and so on would let the editor refuse
   `canny` on a colour image before the program runs. The cost is more complex type declarations.
3. **The opening of the device.** On first use (recommended, with `cv.ready` to await it) or by an
   explicit `await cv.open()`.
4. **Intermediate memory.** A `cv.scope(fn)` that frees what `fn` made and did not return, as
   TensorFlow.js's `tidy` does, or release by `using` declarations only.

## 6. Change control

**Rule 6.1.** A change to a rule of §1 to §3 changes this document first, in its own pull
request, and waits for the owner's decision. An implementation that rests on a rule cites it by
number.

**Rule 6.2.** A proposal of §4 becomes a rule when the owner accepts it. Until then the current
API stands, and nothing in §4 is an available API.
