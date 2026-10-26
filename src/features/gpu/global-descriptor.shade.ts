'use typeshade';

// The word assignment of the global descriptor (`docs/multi-video.md`, section 3.2, step 2 of the
// recipe). One invocation takes one RootSIFT descriptor and finds the nearest of the vocabulary's
// words by squared L2 distance. The distance sums the 128 squared differences in order, in f32,
// and the first word of the least distance wins: a tie goes to the lower word index. The CPU
// reference (`assignWordsReference` in `global-descriptor.ts`) sums in the same order in f32, so
// the CPU tier gives its indices exactly.
//
// There is no barrier and no workgroup memory, so the CPU tier runs this kernel as it is. A
// chunk of a long pass is the range `[globalRange.first, globalRange.end)` of descriptors, and the
// result of a descriptor does not depend on the chunk it falls in.
//
// - `globalDescriptors`: f32 × `dimensions` per descriptor.
// - `globalVocabulary`: f32 × `dimensions` per word.
// - `globalWords`: u32 per descriptor, the index of its word.

class GlobalControl {
  words: u32;
  dimensions: u32;
}
class GlobalRange {
  first: u32;
  end: u32;
}
declare const globalControl: uniform<GlobalControl>;
declare const globalRange: uniform<GlobalRange>;
declare const globalDescriptors: storage<array<f32>>;
declare const globalVocabulary: storage<array<f32>>;
declare const globalWords: storage<array<u32>, 'read_write'>;

@compute([64])
export function globalAssign(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = globalRange.first + gid.x;
  if (i >= globalRange.end) return;
  const dimensions: u32 = globalControl.dimensions;
  let best: f32 = bitcast<f32>(u32(0x7f7fffff));
  let bestWord: u32 = 0;
  for (let w: u32 = 0; w < globalControl.words; w++) {
    let distance: f32 = 0;
    for (let d: u32 = 0; d < dimensions; d++) {
      const delta: f32 = globalDescriptors[i * dimensions + d] - globalVocabulary[w * dimensions + d];
      distance += delta * delta;
    }
    if (distance < best) {
      best = distance;
      bestWord = w;
    }
  }
  globalWords[i] = bestWord;
}
