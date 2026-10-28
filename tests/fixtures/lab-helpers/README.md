# Lab helper snapshots used by library benchmarks

`klt-join.ts` is an exact snapshot of `website/src/lib/klt-join.ts` from TypeCV revision
`715fd1b694508014315e1a505aa2c970051a7f3a` (blob `c1dc865`). The RGB structure-quality benchmark
uses its pure `mergeTrackerPairs` helper to evaluate the captured library reconstruction output.
The snapshot keeps this library experiment reproducible without importing website code.
