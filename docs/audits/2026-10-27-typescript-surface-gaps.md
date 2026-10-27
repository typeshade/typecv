# Where TypeCV's GPU code had to leave TypeScript

Record: a read-only audit of 2026-10-27 by a Claude Code sub-agent at the coordinator's request. It
is the inventory that TypeShade proposals cite: for each construct a TypeScript author reaches for,
what TypeShade's surface says at TypeCV's pin, where TypeCV wrote a hand-made form instead, what that
form costs, and how a compiler could lower the construct. No file of either repository was changed,
except this record and the index line of [`../README.md`](../README.md). It states facts at the
commits of section 0.

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
     loop shapes and entries. Each count in this record names what it counts.
  2. Line counts: the code lines (not blank, not only a comment) of the cited ranges. Each count
     lists its ranges.
  3. 65 probe programs under 60 ids (section 0.1). Each compiles a "use typeshade" source of 5 to
     20 lines with `compile()` and `hostFace()`, and asks the language service
     (`createTypeshadeLanguageService().getDiagnostics`) for the editor's half. The compiler is
     bundled with esbuild, as `tests/typeshade/compiler-probes.test.mjs` does. Eight probes also
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
- **Fact.** Another agent measures `src/mvs/gpu/cloud-knn.shade.ts` (pointer parameters, `break`,
  `while`, early return, permutation writes) on branch `claude/compiler-probes-knn`. No pull
  request of it was open on 2026-10-27, so this record cites those shapes as "under measurement".

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

- **Fact.** The two halves give the same codes on every probe. The editor adds TypeScript's own
  `TS2339` beside `TS8099` on Q5 to Q9, and `TS7006` on Q7 and Q8. Q41 and Q52 give two
  diagnostics for one mistake in both halves.
- **Fact.** Q45's editor half was not run: the harness's file name could not resolve the package.
- **Inference.** WGSL requires an array's element type to have a creation-fixed footprint, so
  Tint refuses Q49's WGSL. Not run on Tint.

## 1. The answer in short

This section follows in the next commit of this pull request.

## 2. The inventory by construct

This section follows in the next commit of this pull request.

## 3. The ranking

This section follows in the next commit of this pull request.

## 4. The map to existing changes and rows

This section follows in the next commit of this pull request.

## 5. What cannot map, and the refusals the compiler should give

This section follows in the next commit of this pull request.

## 6. The recommended proposals

This section follows in the next commit of this pull request.

## 7. Kernel refactors that need no compiler change

This section follows in the next commit of this pull request.

## 8. Open items and questions for the owner

This section follows in the next commit of this pull request.
