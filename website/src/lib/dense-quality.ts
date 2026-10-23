/** Explicit experiments for paired MVS comparisons; absent query parameters preserve defaults. */
export interface DenseQualityOptions {
  readonly boundaryAware?: boolean;
  readonly minFiniteGain?: number;
}

export function denseQualityOptions(search: string): DenseQualityOptions {
  const query = new URLSearchParams(search);
  const options: { boundaryAware?: boolean; minFiniteGain?: number } = {};
  const boundary = query.get('denseBoundary');
  if (boundary === '1' || boundary === '0') options.boundaryAware = boundary === '1';
  const gain = query.get('denseFiniteGain');
  if (gain !== null && gain.trim() !== '') {
    const value = Number(gain);
    if (Number.isFinite(value) && value >= 0 && value <= 2) options.minFiniteGain = value;
  }
  return options;
}
