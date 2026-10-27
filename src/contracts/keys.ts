/**
 * The string keys of the sparse pipeline's records, one spelling each (docs/library-structure.md,
 * section 5, phase 2). A set or map of one module is read by another with these keys, so every
 * site builds them here.
 */

/** The key of an observation: `viewId:featureId`. */
export const observationKey = (viewId: number, featureId: number): string =>
  `${viewId}:${featureId}`;

/** The key of an unordered image pair: `min:max` of the two view ids. */
export const pairKey = (first: number, second: number): string =>
  `${Math.min(first, second)}:${Math.max(first, second)}`;
