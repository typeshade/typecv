# Where TypeCV's GPU code had to leave TypeScript

Record: a read-only audit of 2026-10-27 by a Claude Code sub-agent at the coordinator's request. It
is the inventory that TypeShade proposals cite: for each construct a TypeScript author reaches for,
what TypeShade's surface says at TypeCV's pin, where TypeCV wrote a hand-made form instead, what that
form costs, and how a compiler could lower the construct. No file of either repository was changed,
except this record and the index line of [`../README.md`](../README.md). It states facts at the
commits of section 0.

Written in two sessions on 2026-10-27. The first wrote section 0 with probes Q1 to Q60, and a
container restart ended it. The second reran the first session's census scripts, added probes Q61
to Q63 and wrote sections 1 to 8.

Each statement carries a label: **Fact** (read in a source, counted by a script, or measured by a
compiler probe), **Inference** (a conclusion from facts) or **Proposal** (a choice for the owner).
A file:line names the evidence. A claim of "absent" or "refused" cites a probe of section 0.1; a
claim with no probe says that it is read from the surface document only.

## 0. Configuration and method

- **Fact.** TypeCV: `73c84dc` (`main`), on branch `claude/audit-typescript-surface`. `src/` holds 30
  kernel modules (13,186 lines) and 7 kernel aliases of 3 lines each (`export * from …`).
- **Fact.** TypeShade pin: `725b2969` (`package.json`), installed by `npm ci` as
  `node_modules/typeshade`, package version 0.1.0. The surface document, the design rules and the
  roadmap were read there, so they are the pin's.
- **Fact.** TypeShade `main` was read at `d0ce8843` on 2026-10-27. It is 10 commits after the pin.
  It implements change 0096 (typeshade/typeshade#782) and change 0097 (typeshade/typeshade#779), and
  accepts change 0098, `null` as an absent value (typeshade/typeshade#784).
  `git diff 725b2969 d0ce8843 -- docs/use-typeshade-surface.md` changes §31, §65 and §76 only, for
  0096 and 0097.
- **Fact.** Node 22.22.0, TypeScript 5.9.3, esbuild 0.25.12. Date: 2026-10-27.
- **Fact.** Method, in four parts. The scripts are outside the repository.
  1. A census of the 30 kernel modules with the TypeScript compiler API: constructs, index forms,
     loop shapes and entries. Each count in this record names its script: `idioms.mjs`,
     `entries.mjs`, `loops.mjs`, `runs.mjs`, `hostpack.mjs` and `neardups.mjs` of the first session;
     `structs.mjs`, `forms.py`, `localpush.py` and `csr.py` of the second.
  2. Line counts: the code lines (not blank, not only a comment) of the cited ranges, by
     `count.py`. Each count lists its ranges; `rows.sh` holds the ranges of every row of section 2.
  3. 72 probe programs under 63 ids (section 0.1). Each compiles a "use typeshade" source of 5 to
     20 lines with `compile()` and `hostFace()`, and asks the language service
     (`createTypeshadeLanguageService().getDiagnostics`) for the editor's half. The compiler is
     bundled with esbuild, as `tests/typeshade/compiler-probes.test.mjs` does. Nine probes also
     run the generated module on the CPU tier (`configure({ prefer: ['cpu'] })`).
  4. Reading: the surface, the rules, the roadmap, changes 0076 to 0098, issues
     typeshade/typeshade#761, #775 and #780, and TypeCV's records
     ([`2026-10-26-kernel-style.md`](2026-10-26-kernel-style.md),
     [`../library-structure.md`](../library-structure.md) sections 3.6 and 6,
     [`../typeshade-capabilities.md`](../typeshade-capabilities.md), `compiler-changes.md`).
- **Fact.** The probe suite's 35 facts (`tests/typeshade/compiler-probes.sources.mjs`, P1 to P30
  and S1 to S6) are cited by id and not repeated.
- **Fact.** Not run: no TypeCV test but the site check of the pull request, no WebGPU device, no
  Tint, no hardware. No performance number is claimed.
- **Fact.** Pull request #257 (draft, branch `claude/compiler-probes-knn`, head `f67a31f`) measures
  `src/mvs/gpu/cloud-knn.shade.ts` at the same pin with 8 probes, K1 to K6b; its suite passes 43
  of 43. This record cites them as "measured in #257". Accepted: K1, a helper that takes a local
  `array<f32, 32>` as `@inout` (WGSL `ptr<function, array<f32, 32>>`, GLSL `inout float[32]`); K2,
  a helper that reads bindings by name and hands the list on; K3, `break`; K4, `while`; K5, an
  early `return` from nested loops of an entry; K6b, a kernel function that writes `sums[lane]`.
  Refused: K3b, a labelled `break` (`TS8099`); K6, `sums[visitOrder[lane]]` in a kernel function,
  which runs on the CPU tier (`TS8070`, rule R3).

### 0.1 Compiler probes (pin `725b2969`, 2026-10-27)

"Editor" is the language service's half. "Same" means the same code and text. A row with two
verdicts names its variants.

| Id | Source shape | `compile()` | Editor |
| --- | --- | --- | --- |
| Q1 | `const m = new Map<u32, u32>()` in a helper | `TS8035` "Map" is a type, not a value | same |
| Q2 | kernel `Map` parameter: `if (!m.has(k)) m.set(k, i)` in one loop | `TS8070` R4 ("reads "owners", a Map the loop adds to"); CPU tier | same |
| Q3 | kernel `Set` parameter: `add` in a loop, `size` after | GPU loop (push records); CPU run: keys `[5, 3, 9]`, size 3 | none |
| Q4 | `declare const t: storage<Map<u32, u32>>` in an entry | `TS8002` "A Map<K, V> is a kernel function's parameter only" | same |
| Q5 | `m.delete(k)` on a kernel `Map` | `TS8099` "has set, has, get, size (change 0080)" | `TS2339` and `TS8099` |
| Q6–Q9 | `sort()`, `find(p)`, `filter(p)` on `array<f32, 8>`; `slice(a, b)` on a storage array | `TS8099` ".sort is not one of an array's methods here, which are map, forEach, some, every, reduce, fill, findIndex, findLastIndex, indexOf, lastIndexOf and includes…" | `TS8099` with `TS2339` (and `TS7006` for Q7, Q8) |
| Q10 | `list.push(v)` on a function-scope `array<f32, 2>` | `TS8099`, the sentence of Q6 | same |
| Q11 | `fill`, `reduce`, `findIndex`, `some`, `includes` on `array<f32, 36>` in an entry | accepted | none |
| Q12 | `new Array(32).fill(1e30)` | `TS8035` "Array" is a type, not a value | same |
| Q13 | one kernel loop pushes onto two arrays | no diagnostic; CPU tier ("pushes onto "lo" and "hi"") | none |
| Q14 | a push inside a nested loop of a kernel function | GPU loop | none |
| Q15 | `out.push(i)` in a `@compute` entry | `TS8099`, the sentence of Q6 | same |
| Q16 | `outer: for … break outer;` | `TS8099` "A labelled statement is not supported … Restructure with a flag, or hoist the inner loop into a function and return from it." | same |
| Q17 | `do { … } while (c)` | `TS8099` "do…while is not supported: the IR has one loop shape, a top-tested "for"…" | same |
| Q18 | binary search `while (lo < hi)` in a helper that reads a storage binding | accepted | none |
| Q19 | `const [x, y, z] = points[i]`, `points: storage<array<array<f32, 3>>>` | accepted | none |
| Q20 | `const { x, y, z } = points[i]`, `points: storage<array<Point>>` | accepted; WGSL `array<Point>` | none |
| Q21 | parameter pattern `function len({ x, y }: P)` | `TS8020` "Parameter must be a simple identifier." | same |
| Q22 | `const { x = 1., ...rest } = p` | `TS8099` "A default in a pattern has no shader form…" | same |
| Q23 | `n?: f32` in a struct of a storage binding | `TS8002` ""array<Hit>" holds "[i].n: f32 \| undefined" … a binding may not. Carry a u32 flag beside the value" | same |
| Q24 | kernel function returns `best > 0. ? best : undefined` | no diagnostic; host view `export declare const firstPositive: never;` | none |
| Q25 | `f32 \| null` | `TS8002` "names null … Write "f32 \| undefined"" | same |
| Q26 | helper returns `vec4f64 \| undefined` | accepted | none |
| Q27 | helper returns `[f32, u32]` | `TS8002` "A list of one type is array<T, N>; a list of several is a struct" | same |
| Q28 | helper returns `[f32, f32]`, destructured at the call | accepted; CPU run gives 2 | none |
| Q29 | `sweep: "pcg" \| "update"` in a uniform struct, `switch` on it | accepted | none |
| Q30 | `sweep: Sweep` (an `enum`) in a uniform struct | accepted | none |
| Q31 | `throw new Error("table full")` | `TS8013` "try/catch/throw are JS exceptions. TypeShade has no exception path." | same |
| Q32 | recursion with a depth bound | `TS8031` "Recursive call … WGSL has no call stack" | same |
| Q33 | `x: number` | `TS8002` "A number on the GPU has a width. Write f32 for a float, i32 or u32 for an integer." | same |
| Q34, Q35 | `xs[gid.x + d]` with `d: i32`; `x * n` with `x: f32`, `n: u32` | `TS8003` "WGSL has no implicit integer conversion"; `TS8003` "no implicit int/float conversion" | same |
| Q36 | `let k = 0; k = gid.x;` / Q36a `k = u` with `const u: u32 = gid.x` / Q36c `const k = 0; xs[k]` | `TS8003` "cannot assign u32 to f32" / accepted / `TS8003` "Index must be i32 or u32." | same |
| Q37 | `` console.log(`x = ${x}`) `` | `TS8013` "Pass the text and the value as two arguments: console.log("x =", x)." | same |
| Q38 | `console.log` in a kernel loop | `TS8070` R6; CPU tier | same |
| Q39 | module `let calls: u32 = 0` that an entry increments | accepted; WGSL `var<private>`; CPU tier: `out[0]` is 1 after each of two calls | none |
| Q40 | a kernel loop calls a helper that writes a module `let` | `TS8070` R5; CPU tier | same |
| Q41 | `const g = sq; return g(x);` | `TS8099` "a shader has no function values" and `TS8004` Unknown function "g" | same |
| Q42 | a method on a uniform struct, `control.last()` | accepted | none |
| Q43 | `atomicCompareExchangeWeak` on a kernel function's `array<atomic<u32>>` | `TS8099` "atomic<u32> lives in storage or workgroup memory only … not as a parameter." | same |
| Q44 | `let best = Infinity` | `TS8022` Unknown identifier "Infinity". | same |
| Q45 | `import { hash32 } from "typeshade/random"`; `cellBucket` through `hash32` | accepted (with `readDocument`); CPU values equal Q45b's | not run |
| Q45b | `cellBucket` as written in `cloud-knn.shade.ts:45-52` | accepted; CPU values `198770832`, `3399120504` for (1, 2, 3), (7, 9, 11) | none |
| Q46 | module constant `array<vec2i, 4>` indexed at run time | accepted | none |
| Q47 | `keep: bool` in a struct of a storage binding | `TS8051` "a storage struct holds numeric scalars only (WGSL's host-shareable rule). Use u32." | same |
| Q48 | a storage struct of `f64` fields | accepted; each field `vec2<f32>` | none |
| Q49 | `storage<array<array<u32>>>` | no diagnostic; WGSL `var<storage, read> cells: array<array<u32>>;` | none |
| Q50 | `storage<array<atomic<f32>>>` | `TS8002` "atomic<T> T must be u32 or i32." | same |
| Q51 | a class whose method changes its object (best and second best), called in an entry loop | accepted | none |
| Q52 | `function trace<N extends number>(a: array<f64, N>, n: u32)` / Q52b `trace<9>(…)` / Q52c `<T, N>` | `TS8002` twice "This call does not say what "N" is … write it instead, as "trace<f32>(…)"" / `TS8002` "Unsupported type syntax "9"" / as Q52 | same, twice |
| Q53 | one bilinear helper that takes `at: (k: u32) => f32`, called with arrows over two bindings | accepted; one copy per arrow | none |
| Q54 | `for (const w of weights)` over a storage binding in an entry | accepted | none |
| Q55 | `let best: f32 = Number.MAX_VALUE` | `TS8022` "Number" is a type, not a value | same |
| Q56 | kernel function pushes a struct, `out.push({ x, y, z, w })` | GPU loop; CPU run: count 3, `w` = `[0, 2, 4]` | none |
| Q57 | layouts by `reflect()`: `storage<S>` of three `f64`; `array<array<f32, 3>>`; `array<P3>` of three `f32`; `array<vec3f>` | `S`: size 24, offsets 0, 8, 16; WGSL `array<array<f32, 3>>` and `array<P3>`; `array<vec3<f32>>` (16-byte elements) | none |
| Q58 | host value of `storage<array<P3>>`, CPU tier | a `Float32Array`: `norm(): binding "recs" (array<P3>): got a Float32Array, not an array.`; an array of objects: accepted, `[5, 3]` | none |
| Q59 | exported `enum P3pStatus`, stored as `u32(P3pStatus.Overflow)` | accepted; the host imports `P3pStatus.Overflow` = 5 | none |
| Q60 | the same store without `u32(…)` | `TS8003` "cannot assign i32 to u32" | same |
| Q61a–Q61d | a helper returns `isFinite(x)` / `Number.isFinite(x)` / `isNaN(x)` / `Number.isNaN(x)` | `TS8004` Unknown function "isFinite". Did you mean "finite"? / `TS8022` ""Number" is a type, not a value: the library declares it for TypeScript's own use." / `TS8004` Unknown function "isNaN" / `TS8022`, the text of Q61b | same |
| Q62 | `pts: storage<array<array<f32, 3>>>`, `const [x, y, z] = pts[gid.x]` in an entry; its host value on the CPU tier | accepted; the view types `pts` as `readonly (readonly number[])[]`; a `Float32Array`: `norm(): binding "pts" (array<array<f32, 3>>): got a Float32Array, not an array.`; an array of arrays: accepted, `[5, 3]` | none |
| Q63 | `select(v * 2., -v, v < 0.)` against `v < 0. ? -v : v * 2.` | both accepted; the WGSL and the GLSL are identical | none |

- **Fact.** The two halves give the same codes on every probe. The editor adds TypeScript's own
  `TS2339` beside `TS8099` on Q5 to Q9, and `TS7006` on Q7 and Q8. Q41 and Q52 give two
  diagnostics for one mistake in both halves.
- **Fact.** Q45's editor half was not run: the harness's file name could not resolve the package.
- **Inference.** WGSL requires an array's element type to have a creation-fixed footprint, so
  Tint refuses Q49's WGSL. Not run on Tint.

## 1. The answer in short

- **Fact.** The surface at the pin takes most of what a TypeScript author writes in a kernel:
  `while`, `break`, an early `return` in an entry, inlined arrow functions, classes with methods,
  `for…of`, `switch`, `enum`, string literal unions, optional values in function scope,
  destructuring of fixed arrays and of storage elements, the array methods of §63, `push` in a
  kernel loop, and `Map` and `Set` parameters (Q3, Q11, Q14, Q18 to Q20, Q26, Q28 to Q30, Q39, Q42,
  Q51, Q53, Q54, Q56, Q59, Q62; K1 to K5 measured in #257). Generics over types are read from §30
  only.
- **Fact** (`idioms.mjs`, `entries.mjs`). TypeCV's 30 kernel modules use few of them: 0 array
  methods, 0 `for…of`, 0 `switch`, 0 `enum`, 0 class methods, 1 `while` and 3 arrow functions.
  They write 1,044 casts and 970 storage accesses at a stride.
- **Fact** (`compiler-changes.md`; `library-structure.md` section 5). The array methods (change
  0079) reached TypeCV's pin with #164, and optional values (0088, 0093) with #246.
- **Inference.** Most of the distance from TypeScript is not forced by the pinned compiler. Part of
  it is older than the facility. Section 7 lists the refactors that today's surface allows.
- **Fact.** Six forms are forced at the pin. Each is a row of section 2.
  1. T7, records in storage at a stride: the CPU-tier host value of a struct-array or a
     fixed-array binding is an array of objects or of arrays, not a typed array (Q58, Q62).
  2. T1, hash tables by hand, 188 code lines: a `Map` is a kernel function's parameter only, and
     WebGPU applies its adds on the host (§74).
  3. T4, sorting and ranking by hand, 122 lines: `sort` is refused (Q6).
  4. T6, bounded lists with a counter, 77 lines: `push` is refused outside a kernel loop (Q10).
  5. T5, compactions as flag, scan and scatter entries, 397 lines: change 0094 is not implemented
     at the pin.
  6. T11, the clamp-and-mask idle lane, 153 lines in 33 entries: the A1 rule. Change 0096 came
     after the pin, and it leaves an author's entry as written (its decision D4).
- **Inference.** Accepted changes cover forms 5 and 6. Forms 1 to 4, a generic over an array's
  length (T17) and `Infinity` with the finiteness tests (T22) are the six proposals of section 6,
  in the order of the ranking (section 3).
- **Fact.** Some TypeScript meanings cannot map: a collection with no capacity, a string at run
  time, an exception, recursion, a function value, an infinity where WGSL lets a driver assume
  none. Section 5 gives the refusal that each should carry. Two probes compile with no diagnostic to
  a wrong or an invalid result, Q24 and Q49 (section 8).

## 2. The inventory by construct

Each row stands alone, so that a proposal can cite it by its id. Columns: (a) what the surface says
at the pin, with its probe; (b) the TypeCV sites of the hand-made form; (c) the code lines of the
cited ranges, kernel plus host (`count.py` over the ranges of `rows.sh`); (d) a lowering sketch for
WGSL, GLSL ES 3.00 and the CPU oracle, with the declaration that the author adds. Column (e), what
cannot map, is section 5. A file name alone is a kernel module, since the 30 names are unique. A
host module is named by its path under `src/`. Each census count names its script.

### 2.1 Collections and lists

| Id | Construct | (a) Surface at the pin | (b) TypeCV sites of the hand-made form | (c) Lines | (d) Lowering and declaration |
| --- | --- | --- | --- | --- | --- |
| T1 | `Map` and `Set` with integer keys | §74 (change 0080's code; its record says `accepted`): a kernel function's parameter only. On WebGPU each add pushes a record that the call applies to the host's `Map`. A later loop finds a key by a binary search over the uploaded keys. An add in a nested loop runs the function on the CPU tier. A local map `TS8035` (Q1); a binding `TS8002` (Q4); a read in the loop that adds `TS8070` R4 (Q2); `delete` `TS8099` (Q5); a `Set` that a loop fills, then its `size`, accepted (Q3) | Open-addressing tables claimed by `atomicCompareExchangeWeak`, with an `EMPTY` sentinel and a hand-written hash: `cloud-voxel.shade.ts:33,40-47,57-76`; the TSDF block table, its claim, its assignment and its lookup, `tsdf.shade.ts:117-118,191-215,430-444,449-461`. Bucket lookups in a host-built grid: `cloud-knn.shade.ts:45-52,102-111`, `dense-mesh.shade.ts:71-78,140-146`. Host: the table's growth loop and key list `mvs/gpu/tsdf.ts:121,453-475`; a host rebuild of the table `mvs/gpu/tsdf-box-table.ts:1-39`; the grid upload and the table fill `mvs/gpu/cloud-clean.ts:56-63,101-106`; the mesh weld `mvs/gpu/dense-mesh.ts:179-187`. The CPU reference keys a `Map<string, number>` with a template string (`mvs/cloud-clean.ts:466-483`) | 114 + 74 = 188; the CPU reference's grid (57) stays | A device-resident `Map<u32, V>` or `Set<u32>` that a kernel function fills and later calls read. Open addressing over `array<atomic<u32>>` with one reserved key word; an insert by compare-exchange, bounded by the table size; `has` and `get` by linear probing. The insertion order is each key's least iteration (`atomicMin`), compacted by 0078's scan only when the host reads the map. The value is the last iteration's, as 0080's draft has it. GLSL ES 3.00: no atomics, so no tier. Oracle: JavaScript's `Map`. Declaration: the capacity |
| T2 | Array methods: `reduce`, `fill`, `map`, `some`, `every`, `findIndex`, `indexOf`, `includes` | §63 (change 0079): eleven methods on `array<T, N>`, all but `map` on storage and kernel arrays. `fill`, `reduce`, `findIndex`, `some` and `includes` accepted in an entry (Q11). `find` refused with the methods list (Q7), and §63 names `findIndex` as the fix | 0 method calls in the 30 modules (`idioms.mjs`). 159 counted loops run over a whole local `array<T, N>` (`loops.mjs`), 43 of them one-line folds such as `sift.shade.ts:383` (`maximum = max(maximum, hist[b])`). `every` as a loop with an early return: `bundle.shade.ts:982`, `joint-focal.shade.ts:989` | 0 by a compiler change: a method keeps a one-line loop at one line | None needed. A function-scope `reduce` folds left to right (§63), the order of the loop it replaces |
| T3 | `slice` and a sub-range copy | §63 refuses `slice`: `TS8099` with the methods list (Q9). A copy is a loop | 44 one-line loops copy between a local array and a storage record (`forms.py`), e.g. `pnp.shade.ts:145` (`p[j] = poses[model * 12 + j]`), `pair-models.shade.ts:148,167` (81 words in and out), `recovery.shade.ts:310-312` | 44 held, 0 removed: a `slice` is one line too | `xs.slice(a, a + N)` with a constant `N` lowers to an `array<T, N>` copy loop per (T, N); every target and the oracle run that IR. Declaration: none |
| T4 | `sort`, sorted insertion, ranking | `sort` refused (Q6). `typeshade/algorithms` lists sort as "Not yet" (§72) | The k-NN insertion into `best: array<f32, 32>`, written twice with a 31-step loop: `cloud-knn.shade.ts:116-126,146-155` (K1 and K4 measured in #257). The MVS cost list `mvs.shade.ts:1638-1648`. An insertion sort and merge of roots `p3p.shade.ts:84-93`. A quickselect in one invocation, `selectAngle`, `recovery.shade.ts:600-631`. A histogram top-k in one invocation, `orbPickTop`, `orb.shade.ts:245-286`. Host sorts after a readback: `mvs/gpu/tsdf.ts:722-733`, `analysis/gpu/hough.ts:551`, `motion/gpu/klt-tracker.ts:497`. The best-and-second-best updates (`features.shade.ts:323,391`; 8 in `float-matching.shade.ts`) are D1 and not counted | 108 + 14 = 122 | `xs.sort(cmp)` on a function-scope `array<T, N>`: a stable insertion sort per (T, N, comparator), the comparator handed as §14's functions are. The oracle runs the same IR, so equal keys keep their order on every tier, as ES2019's stable sort does. A second part sorts a kernel function's array on the device (§72). Declaration: none |
| T5 | `filter`, and `push` across a loop | `push` in a kernel function's loop (change 0078; Q14, Q56). Refused in an entry (Q15). A loop that pushes onto two arrays runs on the CPU tier with no diagnostic (Q13). Change 0094 (accepted, not implemented at the pin) keeps the count on the device | 29 entries flag, scan, scatter or move a count: the list of `library-structure.md` section 3.6 (SIFT 5, ORB 6, `features.shade.ts` 4, float matching 2, `kltCompact`, MVS fusion 4, `scan.shade.ts` 3, the pair-preparation scan copy 4). Host: 35 lines in 8 modules name `planFlagScan` or `scanBlockCount`, and `device/gpu/scan.ts` | 336 + 61 = 397; `scan.shade.ts` and `scan.ts` (82) stay until `library-structure.md` section 7, item 14 | Changes 0078 and 0094: count, scan, store, with the count held in a `Resident`. Declaration: the capacity, which the host array already is |
| T6 | A bounded list in function scope | `push` on a function-scope `array<T, N>` refused (Q10). `new Array(n)` `TS8035` (Q12) | `RootSet { values: array<f64, 8>; count; overflow }`, its pushes behind a capacity test, its dedupe and its overflow flag: `p3p.shade.ts:10-14,38-41,47-58,77-81,94-103`. Up to four poses per problem, with a duplicate test and status 5 on overflow: `p3p.shade.ts:332-358`. Matrix rows `triangulation.shade.ts:172-173`; two orientations per keypoint `sift.shade.ts:384-388,399-401`; angles `recovery.shade.ts:642-645` (`localpush.py`) | 77 | A struct of `array<T, N>` and a `u32` length. `push` stores below N and counts past it, as 0078's count may pass the capacity, so `length > N` is the overflow test. Reads below `min(length, N)`; `pop`, `includes` and `for…of` over that range. Every target and the oracle run that IR. Declaration: the capacity N |
| T7 | Records in storage; nested and jagged arrays | Struct-array and fixed-array-element bindings compile and destructure (Q19, Q20, Q48, Q57, Q62). Their CPU-tier host value is an array of objects or of arrays, and a `Float32Array` is refused (Q58, Q62). `storage<array<array<u32>>>` compiles with no diagnostic to WGSL that WGSL forbids (Q49). Change 0095 (C6) widens the record writes the proof accepts | Flat `array<f32>` and `array<u32>` bindings read at a stride: 970 storage accesses at `b * N + c` and 211 local ones (`idioms.mjs`). 119 runs of two or more statements read or write one record's fields in order, 342 lines (`runs.mjs`), e.g. `cloud-knn.shade.ts:69-74`. Host packers: 208 lines in 20 modules write or read a field at a constant stride (`hostpack.mjs`; most in `motion/gpu/klt.ts` 47, `analysis/gpu/hough.ts` 22, `motion/gpu/klt-tracker.ts` 22, `sfm/tracks/gpu/observation-expansion.ts` 22). Jagged rows by offsets: `cloud-knn.shade.ts:104-105`, `dense-mesh.shade.ts:141-142`, `triangulation.shade.ts:354-355`, `bundle.shade.ts:837`, `joint-focal.shade.ts:799` (`csr.py`'s hits, read one by one) | Up to 223: one destructuring per run (`runs.mjs`) | The flat typed array of the element's layout as the host value of `array<S>` and `array<array<T, N>>`, checked against `reflect()`'s stride. WebGPU binds its bytes. The CPU tier and the oracle read a field at its offset. WebGL2 as today. A jagged `array<array<T>>` lowers to offsets and values (CSR), or is refused naming that form. Declaration: none |

### 2.2 Control flow

| Id | Construct | (a) Surface at the pin | (b) TypeCV sites of the hand-made form | (c) Lines | (d) Lowering and declaration |
| --- | --- | --- | --- | --- | --- |
| T8 | `for…of` | Accepted over a storage binding (Q54) and over a kernel function's array (§65) | 0 in the 30 modules; 22 counted loops are bounded by `.length` (`loops.mjs`) | 0 | None needed |
| T9 | Labelled `break` and `continue` | Refused by design, since neither target has a label: `TS8099` names two restructurings (Q16; K3b measured in #257) | 0 labels. Three flags stand for a plain `break`, which K3 accepts: `cloud-knn.shade.ts:80-82,135-138`, `klt.shade.ts:339-343`, `tsdf.shade.ts:430-432` | 13 held; 0 by a compiler change | None needed |
| T10 | `while` and `do…while` | `while` accepted (Q18; K4 measured in #257). `do…while` refused (Q17). The `TS8008` text names an `i32` counter only, though Rule 7.5 takes `u32` too (typeshade/typeshade#788) | 1 `while` (`recovery.shade.ts:612`). 15 counted loops open with `if (…) break` (`idioms.mjs`): a `while` with a safety cap, e.g. `tsdf.shade.ts:421`, `bundle.shade.ts:169`. Two 31-step loops (`cloud-knn.shade.ts:119,148`). 0 `do…while` | 0 by a compiler change | None needed |
| T11 | Early `return`; the idle lane | An entry returns from nested loops (K5, #257); a kernel loop may not (R2). The kernel-function scaffold returns early at the pin (S2), and it clamps and masks after change 0096 (TypeShade `a21f7024`, after the pin). An author's entry keeps its own guard (0096, D4) | 148 of 251 entries open with an early-return guard (`entries.mjs`). 33 entries clamp and mask for A1 (`docs/gpu-sfm.md`, "Rule for kernel authors"): 153 lines name `active` in 12 modules | 153, if an entry's guard were lowered as 0096 lowers a kernel loop | 0096's scaffold for an author's guard at the top of an entry (its decision D4): clamp the index, mask the writes, and return no lane before the end. Declaration: none |
| T12 | `switch`, `enum`, string literal unions | `switch` on a string union or an `enum` accepted (Q29, Q30). An exported `enum` reaches the host (Q59). A member stored into a `u32` needs `u32(…)` (Q60, `TS8003`) | 0 `switch`, 0 `enum`. Numeric modes: `control.kind === 0` or `1` 9 times (`consensus.shade.ts`, `pair-models.shade.ts`), `control.sweep` 4 times (BA), `pick.stage` and `streamControl.phase` once each. Status codes as literals: `status[h] = 0` to `6` in 11 lines (`p3p.shade.ts:202-360`), `statuses[h] = 0` to `5` in 6 lines (`triangulation.shade.ts:358-456`), `reason` codes (`klt.shade.ts:438-441`). The host copies the P3P codes as `GpuP3pStatus` (`geometry/gpu/p3p.ts:5-13`, a public export) | 9 host lines by a refactor; 0 by a compiler change | An enum member takes `u32` where a `u32` is due, as an integer literal does. Declaration: none |
| T13 | `throw` | Refused, `TS8013` (Q31) | No exception path. The TSDF table's overflow word (`tsdf.shade.ts:443`) and the host's retry loop (`mvs/gpu/tsdf.ts:462-468`), counted in T1. Per-element status codes are data (T12) | 0 of its own | None recommended (section 5) |
| T14 | Recursion | Refused, `TS8031` (Q32): WGSL has no call stack (surface §4) | 0 sites | 0 | None |

### 2.3 Functions and values

| Id | Construct | (a) Surface at the pin | (b) TypeCV sites of the hand-made form | (c) Lines | (d) Lowering and declaration |
| --- | --- | --- | --- | --- | --- |
| T15 | Closures and higher-order functions | A local arrow function captures (P3). A helper takes `at: (k: u32) => f32`, one copy per arrow (Q53). A function value is refused, with two diagnostics (Q41) | 3 arrow functions (`mvs.shade.ts:719,945,950`). Three families of helpers that are equal but for the binding they read: `features.shade.ts:108,113`, `klt.shade.ts:133,138`, `klt.shade.ts:200,214`; 49 lines, 25 past the first copy (`neardups.mjs`) | 25 by a refactor (Q53); 0 by a compiler change | None needed |
| T16 | Classes with methods and `this` | Accepted (surface §26; Q42, Q51) | 75 classes, 0 with a method (`entries.mjs`); 16 value structs (`structs.mjs`); the best-and-second-best update written 10 times (T4) | 0 | None needed |
| T17 | Generics; a generic over an array's length | Generics over types (§30). Over a length: `N extends number` is not settled from `array<f64, N>`, `TS8002` twice (Q52); `trace<9>(…)` refused (Q52b) | One Cholesky at two sizes: `solve3`, unrolled (`numerics.shade.ts:25-45`), and `solve6` (`numerics.shade.ts:221-253`), with `Linear3` and `Linear6`. BA's `array<f64, 6>` and `array<f64, 7>` tiles, inside C1's copy | 21 (`solve3`), an **Inference**; C1's ~900 belong to change 0091 | A numeric type parameter, settled from an argument's `array<T, N>` or written `f<9>(…)`, monomorphised per value as §30 does per type. Declaration: none |
| T18 | Destructuring beyond change 0086 | Fixed arrays (0086) and storage elements (Q19, Q20, Q62) accepted. A parameter pattern `TS8020` (Q21); a default or a rest `TS8099` (Q22) | No site needs a pattern parameter, a default or a rest. The 119 runs of T7 need a record element first | 0 | None needed |
| T19 | Optional values beyond 0088 and 0093: `?.`, `??`, `null` | Function scope accepted (Q26); `?.` and `??` are read from §76 only. In a binding refused: "Carry a u32 flag beside the value" (Q23). A kernel function that returns `undefined` compiles with no diagnostic to a host view `never` (Q24). `null` refused at the pin (Q25); change 0098 accepts it after the pin | 30 lines use `1e30` or `1e20` outside comments, some of them as bounds, and 30 lines use `0xffffffff` for "no index" (`features.shade.ts` 10, `float-matching.shade.ts` 20). Flags `ok: u32` in `Projection`, `Linear3` and `Linear6` (`numerics.shade.ts:11-24`) and in `Basis` (`p3p.shade.ts:15-20`). Storage sentinels: `EMPTY` (`tsdf.shade.ts:117`, `cloud-voxel.shade.ts:33`), `NONE` and `NO_VALUE` (`dense-mesh.shade.ts:67-68`) | 0 removed; 60 lines change form | An optional element in storage as the value plus a presence bit in a word array that the compiler adds; the host sees `T \| undefined`. Declaration: none |
| T20 | Tuples and several return values | A pair of one type accepted (Q28). A mixed tuple refused: "a list of several is a struct" (Q27) | 13 value structs that a function returns (`structs.mjs`), e.g. `MvsFused`, `TsdfSample`, `Basis3`. Each names its fields, as TypeScript code would | 0 | An anonymous struct for a mixed tuple; no TypeCV site needs it |
| T21 | `number`; implicit conversions | `number` refused, and `TS8002` names the widths (Q33). No implicit integer or integer-to-float conversion, `TS8003` (Q34, Q35, Q60). An unannotated `0` is an `f32` (Rule 5.1, Q36) | 1,044 casts `f32(…)`, `u32(…)`, `i32(…)` and `f64(…)`; 332 literals written `N.`; 4,186 annotated declarations against 509 unannotated (`idioms.mjs`) | 0 | None: the widths are WGSL's (section 5) |
| T22 | `Infinity`, `Number.MAX_VALUE`, `isFinite`, `isNaN` | `Infinity` `TS8022` (Q44). `Number.MAX_VALUE` `TS8022` (Q55). `isFinite` and `isNaN` `TS8004`, `Number.isFinite` and `Number.isNaN` `TS8022`, in both halves (Q61) | The largest finite `f32`, spelled three ways in 17 lines: `3.4e38` (`slic.shade.ts:102,141`), `bitcast<f32>(u32(0x7f7fffff))` (`global-descriptor.shade.ts:37`; 13 lines of `float-matching.shade.ts`) and `FAR = 3.0e38` (`tsdf.shade.ts:126`). Finiteness tested on the exponent bits, `& 0x7f800000`, in 15 lines of `float-matching.shade.ts`. `1e30` for "no distance" (`cloud-knn.shade.ts:40`) | 0 removed; 32 lines change form | `Infinity` in an `f32` position as the largest finite `f32`, the start that §65 gives `min` and `max`, a stated difference from TypeScript (Rule 7.2). `Number.isFinite(x)` and `Number.isNaN(x)` as integer tests of `bitcast<u32>(x)`, which no finite-math assumption removes. WGSL and GLSL: the bit test. Oracle: a test of the `f32` bits. Declaration: none |

### 2.4 Memory, modules and debugging

| Id | Construct | (a) Surface at the pin | (b) TypeCV sites of the hand-made form | (c) Lines | (d) Lowering and declaration |
| --- | --- | --- | --- | --- | --- |
| T23 | Module state, constants and shared helpers | A module `let` is `var<private>` (Q39); a kernel loop that calls a writer of one runs on the CPU tier (Q40, R5). An exported constant reaches the host (S4). A helper module is imported (§68). `TAU` is a §9.3 name. `hash32` of `typeshade/random` (§73) is the `lowbias32` finaliser | 0 module `let` besides workgroup arrays. Constants copied into hosts: `analysis/gpu/hough.ts:108-110`, `mvs/gpu/tsdf.ts:121`, `mvs/gpu/tsdf-box-table.ts:3`. `TWO_PI` defined twice (`sift.shade.ts:333`, `mvs.shade.ts:229`). Helpers equal across files (`entries.mjs`, mode `dups`): `cellBucket` and `voxelSlot` (3 copies), `encodeNormal`, `decodeNormal` and `hash` (2 each), 42 lines past the first copy; six BA helpers twice (C1). The avalanche of `cellBucket` is `hash32` (Q45: equal CPU values) | 0 by a compiler change; 42 + 7 by a refactor | None needed |
| T24 | `+=` into an `f32` array at a computed index | An `f32` scatter runs on the CPU tier only (§65; P19). `atomic<f32>` refused (Q50) | Weights summed as fixed point in `u32` atomics: `tsdf.shade.ts:121` (`WEIGHT_SCALE`), `tsdf.shade.ts:539-546` | 9 | An `f32` scatter into an `i32` fixed point of a declared scale, exact in any order; or a compare-exchange loop on the bits, whose order the determinism report lists. Declaration: the scale |
| T25 | `boolean` and narrow fields in storage | A `bool` field in storage refused: "Use u32" (Q47) | Counters of 12, 8 and 12 bits packed by a compare-exchange loop, `tally` (`tsdf.shade.ts:475-486`). A status, a reason and a use count packed into one word (`klt.shade.ts:438-452`) | 27 | A storage struct of `bool` and narrow integer fields, packed into `u32` words by the compiler and unpacked in the host view. Declaration: each field's width |
| T26 | A write through a permutation, `out[perm[i]]` | In a kernel loop it runs on the CPU tier, `TS8070` R3 (K6, measured in #257); `out[i]` lowers (K6b). Change 0095 excludes it by name | `knnMean[i]` with `i = visitOrder[lane]` (`cloud-knn.shade.ts:66,161`), in an entry, where it is allowed. The host un-permutes (`mvs/gpu/cloud-clean.ts:85-86`) | 1, per #257 | #257's row C15: a parameter declared a permutation of the loop's range, checked at the call |
| T27 | Template literals and `console.log` | A template literal refused, with the two-argument form named (Q37). `console.log` in a kernel loop runs on the CPU tier (Q38, R6) | 0 `console` calls; the kernels report through status words (T12) | 0 | None needed |

## 3. The ranking

**The departure scale.** One test for every row: how far the TypeCV site is from what a TypeScript
author writes for the same intent.

- D0: the site is the TypeScript form.
- D1: valid TypeScript in a longer spelling. A counted loop stands for a method, a number for an
  enum member, a cast for a conversion, a named struct for a tuple. The line still says what it
  does.
- D2: a structure or an encoding built by hand inside the kernel: a record flattened by stride, a
  sentinel, a flag for an exit, a counter with a capacity test, a hand-written hash, sort or bit
  field. The intent is in a comment.
- D3: the work leaves the kernel language: an extra entry, a host step after a readback, a table
  built on the host, a retry loop.

A row takes the level of most of its counted lines. The order: (i) the lines that a compiler change
removes, then (ii) the sites, then (iii) the level, each descending. A row with no compiler gap
removes 0 lines and ranks by its sites.

| Rank | Id | Construct | (i) Lines a compiler change removes | (ii) Sites | (iii) Level | Covered by (section 4) |
| --: | --- | --- | --- | --- | --- | --- |
| 1 | T5 | `filter`, `push` across a loop | 315 (397 held; the scan's 82 stay) | 29 entries, 8 host modules | D3 | 0078, 0094, 0096 |
| 2 | T7 | Records; nested and jagged arrays | up to 223 | 970 accesses, 119 runs, 5 jagged rows | D2 | 0095 in part; P1 |
| 3 | T1 | `Map` and `Set` | 188 | 8: 4 kernels, 4 host modules | D2 | 0080 in part; P2 |
| 4 | T11 | Early `return`; the idle lane | 153 | 33 entries | D2 | 0096 in part (D4) |
| 5 | T4 | `sort`; ranking | 122 | 9 | D2 | none; P3 |
| 6 | T6 | A bounded list | 77 | 7 | D2 | none; P4 |
| 7 | T25 | Narrow fields in storage | 27 | 2 | D2 | none |
| 8 | T17 | A generic over a length | 21 | 2, and C1's tiles | D1 | 0091 in part; P5 |
| 9 | T24 | An `f32` scatter | 9 | 1 | D2 | none |
| 10 | T26 | A permutation write | 1 | 1 | D0 | C15 (#257) |
| 11 | T21 | `number`; conversions | 0 | 1,376 | D1 | none: by design |
| 12 | T2 | Array methods | 0 | 159 | D1 | 0079 |
| 13 | T19 | Optional values beyond 0088 and 0093 | 0 | 60 | D2 | 0088, 0093 in part |
| 14 | T3 | `slice` | 0 | 44 | D1 | none |
| 15 | T12 | `switch`, `enum` | 0 | 34 | D1 | §52; 0081 |
| 16 | T22 | `Infinity`; finiteness | 0 | 32 | D2 | none; P6 |
| 17 | T10 | `while`, `do…while` | 0 | 18 | D1 | supported |
| 18 | T23 | Module state, constants, helpers | 0 | 9 | D1 | supported |
| 19 | T9 | Labelled exits | 0 | 3 | D1 | supported (`break`) |
| 20 | T15 | Closures | 0 | 3 | D1 | supported |
| 21 | T13 | `throw` | 0 | 1, counted in T1 | D3 | none |
| 22 | T8 | `for…of` | 0 | 0 | D1 | supported |
| 23 | T20 | Tuples | 0 | 0 | D1 | supported |
| 24 | T16 | Classes with methods | 0 | 0 | D0 | supported |
| 25 | T18 | Destructuring beyond 0086 | 0 | 0 | — | 0086 |
| 26 | T14 | Recursion | 0 | 0 | — | none |
| 27 | T27 | `console`; template literals | 0 | 0 | — | supported |

- **Inference.** Ranks 1 and 4 rest on accepted changes: rank 1 on 0094 and 0096, rank 4 on 0096
  and its open decision D4. Their lines go when a pin carries those implementations and the phase
  10 steps run (section 4).
- **Inference.** The proposals of section 6 follow this order. They skip T25 and T24: each has two
  sites or fewer, and T25 needs a field width that neither WGSL, ECMAScript nor §9.3 spells (Rule
  9.6). P6 (T22) has no line to remove; it is in the list for its 32 sites at D2.
- **Fact.** Column (i) is an upper bound of what a lowering deletes: each replacement adds some
  lines back. The kernel audit's estimates for C1 to C9 are separate (its section 6).

## 4. The map to existing changes and rows

**Fact** (the change records at TypeShade `main` `d0ce8843`; `library-structure.md` section 6).
"Fully" means that the construct is covered as TypeCV needs it; "in part" names what is left. The
C-rows are TypeCV's requests, and C15 is the row that #257 proposes.

| Id | Covered by | Fully or in part | What is left |
| --- | --- | --- | --- |
| T1 | 0080 (§74; its record says `accepted`), 0081 (key types) | in part | A device-resident map with a capacity, an add in a nested loop, a lookup on the device, a key of three coordinates (P2). 0094's exclusions run a loop over a device range that adds to a map on the CPU tier |
| T2 | 0079 (implemented) | fully | the refactor of section 7, step 9 |
| T3 | none | — | a constant-length `slice` |
| T4 | none; §72 lists sort as "Not yet" | — | `sort` on a fixed array, then a device sort (P3) |
| T5 | 0078 (implemented); 0094 (C3 and C8, accepted); 0096 (C7) | fully, once implemented and pinned | phase 10's step C3; the pair scan copy after R03; `scan.shade.ts` until `library-structure.md` section 7, item 14 |
| T6 | none | — | P4 |
| T7 | 0082 (private input layout); 0095 (C6, accepted) | in part: the kernel side | the host value of a record binding and jagged arrays (P1) |
| T8 | §17, §65 | fully | nothing |
| T9 | §52 (a label is refused by design) | fully, by a plain `break` | nothing |
| T10 | §17 (#203); §52 defers `do…while` | fully for TypeCV | the `TS8008` text (typeshade/typeshade#788) |
| T11 | 0096 (C7; implemented at `a21f7024`, after the pin) | in part | an author's entry (0096, D4); the A1 hardware gate (`library-structure.md` section 7, item 11) |
| T12 | §52; 0081 (§75) | fully | the `u32(…)` of Q60 |
| T13 | none | — | nothing recommended |
| T14 | surface §4 | — | nothing |
| T15 | §14, §63; 0087 (the oracle's copies by the written name); 0089 (functions that a program supplies) | fully for TypeCV | nothing |
| T16 | §26 | fully | nothing |
| T17 | §30; 0091 (C1: one helper for every binding a call passes) | in part | a numeric type parameter (P5) |
| T18 | 0086 (implemented) | fully for TypeCV | pattern parameters, defaults and rests, which no site uses |
| T19 | 0088 and 0093 (implemented at the pin); 0098 (`null`, accepted after the pin) | in part | an optional value in storage; the host view of Q24 |
| T20 | §28 | fully for TypeCV | nothing |
| T21 | Rule 5.1; §45 | by design | nothing |
| T22 | none | — | P6 |
| T23 | §24, §64 (S4), §68, §73 | fully | the refactors of section 7 |
| T24 | §65 (the CPU tier only). C9 (typeshade/typeshade#761) is the integer `min` and `max` scatter, not this one | — | a fixed-point `f32` scatter |
| T25 | none | — | a bit-field struct in storage |
| T26 | 0095 excludes it by name | — | C15 (#257) |
| T27 | §66 | fully | nothing |

- **Fact.** No construct row needs these: 0085 (host calls without setup) and 0097 (a costly
  conditional arm lowered to an `if`) change no TypeScript form; 0090 (C14) places files; 0092 (C2)
  runs a barrier entry on the CPU tier, which retires the 18 oracle twins, a tier and not a
  construct; C4, C5 and C10 to C13 are runtime and lab rows.
- **Inference.** So the proposals of section 6 add to 0078 to 0098 and to C1 to C15. None duplicates
  one. P2 takes up the form that 0080's decision 3 left open.

## 5. What cannot map, and the refusals the compiler should give

**Fact** for each "Today" cell, with its probe. **Proposal** for each "Should" cell: a refusal that
names the bounded form, as the texts of Q16, Q17 and Q37 already do.

| Id | The TypeScript meaning that a GPU cannot hold | Today | Should |
| --- | --- | --- | --- |
| T1 | a collection that grows with no bound; a key compared by identity; a `string` key | `new Map()` `TS8035` "Map is a type, not a value" (Q1); a binding `TS8002` (Q4); `Map<string, V>` `TS8002` (§74) | name the bounded forms: a kernel function's `Map` parameter (§74) and, after P2, a device map with a capacity; for a string or an object key, a `u32` key or a key type (§75) |
| T3 | `slice` of a run-time length | `TS8099` with the methods list (Q9) | "`slice` needs a constant length here: write `xs.slice(a, a + N)`, or an `array<T, N>` and a loop" |
| T4 | `sort` of a runtime-sized array in one invocation | `TS8099` with the methods list (Q6) | name `sort` on an `array<T, N>` (P3) and, for a kernel function's array, the device sort |
| T5, T6 | `push` with no capacity | `TS8099` with the methods list (Q10, Q15) | name the two capacities: a kernel function's array parameter (0078) and a bounded list (P4) |
| T6 | `new Array(n)`; an array literal that grows | `TS8035` "Array is a type, not a value" (Q12) | name `fill<T, N>(v)` (§9.3) for a fixed length, and the bounded list for a growing one |
| T7 | an array of arrays of run-time lengths in storage | no diagnostic, and WGSL that WGSL forbids (Q49) | `TS8002` naming offsets and values, or P1's lowering |
| T9, T10 | a labelled exit; `do…while` | `TS8099` with the forms that work (Q16, Q17) | keep |
| T13 | an exception that unwinds the call | `TS8013` "TypeShade has no exception path" (Q31) | add the bounded forms: a status code in an array, or a `T \| undefined` result |
| T14 | a call stack | `TS8031` (Q32) | add the bounded form: an explicit stack in an `array<T, N>`, and a `while` |
| T15 | a function as a value | `TS8099` and `TS8004`, two diagnostics for one mistake (Q41) | one diagnostic, as §28 asks of one mistake |
| T17 | a type argument that nothing settles | `TS8002` twice (Q52); `trace<9>` "Unsupported type syntax" (Q52b) | one diagnostic; after P5, the settled forms |
| T19 | an absent value in host-shareable memory | `TS8002` "Carry a u32 flag beside the value" (Q23) | keep. Q24's kernel function result should be refused in the same words, or lowered |
| T20 | a tuple of several types | `TS8002` "a list of several is a struct" (Q27) | keep |
| T21 | a number with no width; an implicit conversion | `TS8002` (Q33); `TS8003` (Q34 to Q36, Q60) | keep. Q36's text could add that the literal `0` made `k` an `f32` (Rule 5.1), and name `let k: u32 = 0` |
| T22 | an infinity or a NaN, which WGSL lets a driver assume away | `TS8022` "Unknown identifier" (Q44); `TS8022` for `Number` (Q55, Q61b); `TS8004` "Did you mean "finite"?" (Q61a) | say what the name is, an ECMAScript value with no portable `f32` form, and name the largest finite `f32` and the bit test, or P6's forms |
| T24 | an `f32` sum whose order follows the schedule | `TS8002` "atomic<T> T must be u32 or i32" (Q50); a scatter on the CPU tier (P19) | name an `i32` fixed-point sum and its scale, or the CPU tier |
| T26 | a write through indices read from data | `TS8070` R3 (K6) | keep; C15 |
| T27 | a string at run time | `TS8013` names `console.log("x =", x)` (Q37) | keep |

- **Fact** (read from surface §28 only, with no probe of this audit). An `async` function, a
  generator, `try` and a `var` are refused once each, with their reason.

## 6. The recommended proposals

**Proposal.** Six TypeShade proposals, in the order of section 3. Each adds to 0078 to 0098 and to
C1 to C15 (section 4). The line counts are section 2's.

**P1 (row T7, rank 2): a record binding takes the flat typed array of its layout.** The construct is
a binding or a kernel function's array whose element is a struct of 4-byte scalars or an
`array<T, N>`, read as `pts[i].y` or `const [x, y, z] = pts[i]`. The evidence: 970 storage accesses
at `b * N + c`; 119 runs of record fields, 342 lines, that destructuring would cut by up to 223;
208 host lines in 20 modules that pack at a stride. Q58 and Q62 measure the gap: the kernel side
compiles, but the CPU tier refuses the `Float32Array` that TypeCV's hosts hold, and takes arrays of
objects or of arrays. The lowering: the call takes a typed array whose length is a whole number of
elements at `reflect()`'s stride; WebGPU binds its bytes; the CPU tier and the oracle read each
field at its offset in the same array; WebGL2 is unchanged. A struct with padding takes the padded
layout that `reflect()` reports. The author declares nothing, since the layout is the type's. The
proposal also gives a jagged `array<array<T>>` (Q49; five TypeCV row bounds by offsets) the
offsets-and-values form, as a lowering or as a refusal that names it. It owes surface §64, §65 and
§67 (the host value tables), the host view types, the site's kernel and host pages and the
editor's references. TypeCV records it in `compiler-changes.md` and converts family by family in
phase 10; the bytes stay the same.

**P2 (row T1, rank 3): a device-resident `Map` and `Set` with a capacity.** The construct is a
`Map<u32, V>` or a `Set<u32>` that a kernel function's loop fills, an add in a nested loop
included, and that later calls read on the device. Change 0080 took the form that the host
applies, and its decision 3 left a device-resident form open. §74 runs a nested add on the CPU
tier, and 0094 runs a map over a device range there too. The evidence: 188 code lines in 4 kernels
and 4 host modules, namely the voxel table, the TSDF block table with its growth loop and its host
rebuild, and the hash grids of the k-NN and the mesh. The lowering: open addressing over
`array<atomic<u32>>` with one reserved key word; an insert by compare-exchange, bounded by the table
size; `has` and `get` by linear probing; the insertion order as each key's least iteration
(`atomicMin`), compacted by 0078's scan only when the host reads the map; the last value per key by
the greatest iteration, as 0080's draft has it. A full table reports a size past its capacity, as
0078's count does, so that the host can retry with more room. GLSL ES 3.00 has no atomics, so
WebGL2 runs no such call; the oracle is JavaScript's `Map`. The author declares the capacity where
the host makes the map. The spelling (`dev.map(capacity)` or another) is the proposal's decision,
and so is a key of three cell coordinates: TypeCV hashes them today, and an object key compares by
identity. It owes §74, 0080's decisions 1 and 3 restated, the planner's group rule (§71), the
site's Map page and the editor's references. TypeCV converts the TSDF, voxel, mesh and k-NN tables
in phase 10, each held to its current output.

**P3 (row T4, rank 5): `sort` on a function-scope array.** The construct is
`xs.sort((a, b) => a - b)` on an `array<T, N>`, which ES2019 makes stable. The evidence: 108 kernel
lines in six sites (the two k-NN insertions, the MVS cost list, the P3P root sort, `selectAngle`
and `orbPickTop`) and 14 host lines that sort after a readback; Q6 refuses the method. The
lowering: an insertion sort per element type, length and comparator, with the comparator handed as
§14's functions are. WGSL and GLSL get one function per instance. The oracle runs the same IR, not
the engine's sort, so equal keys keep their order on every tier. A second part, a device sort of a
kernel function's array (§72's "Not yet"), would retire the host sorts. The author declares
nothing: N is the type's. It owes §63's table and its refusal list, the site's array-methods page
and the editor's references. TypeCV's sites follow in phases 9 and 10, each held to its current
output.

**P4 (row T6, rank 6): a bounded list in function scope.** The construct is an array that grows by
`push` up to a capacity and is read below its `length`, with `pop`, `includes`, `for…of` and P3's
`sort`. The evidence: 77 code lines in seven lists, among them P3P's `RootSet` with its counters,
its overflow flag and its dedupe (`p3p.shade.ts:10-14,38-103`), the four-pose append with its
duplicate test (`p3p.shade.ts:332-358`), the triangulation's matrix rows and SIFT's two
orientations per keypoint. Q10 and Q12 refuse the TypeScript forms. The lowering: a struct of an
`array<T, N>` and a `u32` length; `push` stores below N and always counts, as 0078's count may pass
the capacity, so `length > N` is the overflow test that the kernels write by hand; reads below
`min(length, N)`; one IR for WGSL, GLSL and the oracle. The author declares the capacity N. Its
spelling is the proposal's first decision: an ECMAScript form, or a §9.3 row under Rules 9.6 and
13.6. It owes §63, §18, a new surface section and the editor's references. TypeCV's P3P,
triangulation and recovery sites are sparse-scene kernels, so they wait for R03.

**P5 (row T17, rank 8): a generic over an array's length.** The construct is
`function solve<N extends number>(a: array<f64, N>, …)`, settled from the argument's type or
written `solve<36>(…)`. The evidence: `solve3` (21 code lines, unrolled) and `solve6` (33) are one
Cholesky at two sizes (`numerics.shade.ts:25-45,221-253`), and BA's 6- and 7-wide tiles sit inside
C1's copy. Q52 refuses both spellings, with two diagnostics for one call. The lowering: one instance
per value of N, as §30 makes one per type; WGSL and GLSL get one function per instance; the oracle
runs the same IR. The author declares nothing. A length that no argument settles stays refused,
with one diagnostic. It owes Rule 8.9's text, §30 and the editor's references. For TypeCV, one
generic `solve` changes `solve3`'s order of operations, so it is a behaviour change under Rule 2.2,
after R03.

**P6 (row T22, rank 16): `Infinity` and the finiteness tests.** The construct is ECMAScript's
`Infinity`, `Number.isFinite`, `Number.isNaN`, `isFinite` and `isNaN`, names of Rule 2.1's sources
that need no §9.3 row. The evidence: 17 lines spell the largest finite `f32` three ways, 15 lines
test the exponent bits in `float-matching.shade.ts`, and `1e30` stands for "no distance" in
`cloud-knn.shade.ts:40`; Q44, Q55 and Q61 refuse each name. The lowering: `Infinity` in an `f32`
position is the largest finite `f32`, and `-Infinity` its negation. That is the start that §65
gives `min` and `max`, stated as a difference from TypeScript (Rule 7.2), since WGSL lets a driver
assume no infinity. `Number.isFinite(x)` is `(bitcast<u32>(x) & 0x7f800000) != 0x7f800000`, and
`Number.isNaN(x)` adds a nonzero mantissa: integer tests that no finite-math assumption removes,
the same on WGSL and GLSL; the oracle tests the same bits. The author declares nothing. It owes the
ambient library, §28, §45 and the site. TypeCV's float-matching and SLIC sites follow, each held to
its current output.

- **Inference.** Two candidates are left out: narrow fields in storage (T25, 27 lines in 2 sites)
  and a fixed-point `f32` scatter (T24, 9 lines in 1 site). Each needs a declaration (a width, a
  scale) that no WGSL, ECMAScript or §9.3 name spells today.
- **Proposal.** Two compiler findings go to TypeShade as issues, not proposals: Q24 and Q49
  (section 8, item 1).

## 7. Kernel refactors that need no compiler change

**Proposal.** Each step uses today's surface and keeps the CPU-tier outputs, the gate of phase 9.
A step that changes the WGSL renews the kernel digests in its record. A step on a sparse-scene
kernel waits for R03 (`library-structure.md` section 4, rule 6). A body refactor is no conversion,
so phase 9 needs an amendment that adds it, as #257 proposes for the k-NN.

| Step | What changes | Sites | Evidence | Phase |
| --: | --- | --- | --- | --- |
| 1 | The k-NN body: one `insertSorted(@inout best, k, d2)` with a `while`, a `scanBucket` helper, and `break` for the `done` flag | `cloud-knn.shade.ts:79-158`: two insertions (20 code lines), one flag (7) | K1 to K5 (#257) | 9, by #257's amendment |
| 2 | One cell hash: `hash32` of `typeshade/random` for the `lowbias32` avalanche of `cellBucket`, `voxelSlot` and the tail of `keySlot`; one helper module for these and for `encodeNormal`, `decodeNormal` and `hash` (a murmur finaliser, not `hash32`) | 42 lines past the first copy (T23) | Q45 for `cellBucket`; §73's constants for the others (**Inference**); §68 | 9 |
| 3 | An exit flag becomes `break` | `klt.shade.ts:339-343`, `tsdf.shade.ts:430-432` | K3 (#257) | 9 |
| 4 | `select(f, t, c)` becomes `c ? t : f` | 66 calls: 63 in kernels that are not sparse-scene ones (`klt` 27, `hough` 19, `sift` 5, `features` 4, `slic` 4, `mvs` 2, `orb` 2), 3 in sparse ones | Q63: the same WGSL and GLSL at the pin. After a pin with 0097, a costly arm of `?:` runs only when taken | 9; the 3 sparse sites in 10 |
| 5 | `TWO_PI` becomes `TAU` | `sift.shade.ts:333`, `mvs.shade.ts:229` | §9.3; both round to the `f32` 6.2831855 | 9 |
| 6 | The shader constants that hosts copy are exported and imported | phase 9's list, plus a third copy of `EMPTY` in `mvs/gpu/tsdf-box-table.ts:3` | S4 | 9, as written |
| 7 | The best-and-second-best update becomes a class with one method | 8 blocks in `float-matching.shade.ts` (48 code lines), 2 in `features.shade.ts` | Q51 | 9 |
| 8 | Helpers equal but for their binding become one helper that takes an arrow | `features.shade.ts:108,113`, `klt.shade.ts:133,138,200,214`: 25 lines past the first copy | Q53 | 9 |
| 9 | Whole-array folds become `reduce`, `some` or `findIndex` | 43 one-line folds (T2) | Q11; a function-scope `reduce` folds left to right (§63) | 9 or 10, by area; a D1 gain only |
| 10 | Function-scope sentinels become `T \| undefined` | `library-structure.md`'s sentinel step | 0088 and 0093 at the pin | 10, after R03 (Rule 2.2) |
| 11 | `GpuP3pStatus` comes from an exported shader `enum` | `geometry/gpu/p3p.ts:5-13` against `p3p.shade.ts` | Q59, Q60 | 10, after R03; a public export, so the owner decides |

- **Inference.** The guarded `for` loops of T10 are no step: their counted bound is a safety cap,
  which a plain `while` would drop.

## 8. Open items and questions for the owner

Open items (**Fact**):

1. Four compiler findings are not filed as TypeShade issues. Q24: a kernel function's
   `T | undefined` result compiles with no diagnostic to a host view `never`. Q49:
   `storage<array<array<u32>>>` compiles with no diagnostic to WGSL that WGSL forbids (an
   **Inference**; Tint did not run). Q41 and Q52: two diagnostics for one mistake. Q61a: the text
   suggests `finite`, the helper that holds the call, where section 5 asks for the bounded form.
2. Not measured: a WebGPU, WebGL2 or Tint run of any probe; the WebGPU host value of Q62's binding;
   any speed. Each count of removed lines is an upper bound or an estimate, labelled where it is
   used.
3. Q45's editor half did not run (section 0.1).
4. The record states facts at `73c84dc`. `main` gained `87724f9` (#255, phase 2) and `94d28c8`
   (#256) after the branch was cut, 76 files in all. They change no kernel module and not
   `docs/README.md`. They change two host modules that section 2 cites,
   `src/features/gpu/features.ts` and `src/sfm/tracks/gpu/observation-expansion.ts`, so their
   counts in T5 and T7 may differ at `main`.
5. Row C15 and the k-NN step of section 7 are #257's proposals. They wait for its review.

Questions for the owner (each a decision):

1. Which of P1 to P6 to draft first. The ranking puts P1 first.
2. Whether P2 is an amendment of change 0080 (its decision 3) or a proposal of its own.
3. Whether a bounded list (P4) and narrow storage fields (T25) justify a §9.3 name (Rules 9.6 and
   13.6), or must find an ECMAScript form.
4. Whether the body refactors of section 7 enter phase 9 by one amendment, with #257's k-NN step,
   or by one amendment each.
5. Whether `GpuP3pStatus`, a public export of `typecv/gpu`, may become a re-export of a shader
   `enum` (section 7, step 11).
6. Whether the 33 clamp-and-mask entries of T11 wait for 0096's decision D4, or convert to kernel
   functions after a pin that carries 0096.
