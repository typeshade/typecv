import type { ImageFeatures, PairMatches, RegisteredView } from '../../src/reconstruction/types.ts';

export function orbitFixture(options?: {
  ratio?: number;
  start?: number;
  views?: number;
  points?: number;
}): {
  images: ImageFeatures[];
  pairs: PairMatches[];
  width: number;
  ratio: number;
  truth: { views: RegisteredView[]; points: Float64Array[] };
};
