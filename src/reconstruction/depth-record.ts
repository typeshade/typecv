import type { InverseDepthSnapshot } from './inverse-depth-tracker.ts';

/** Portable sidecar for reconstruction/report consumers. Does not materialize unknown depth. */
export function inverseDepthRecord(snapshot: InverseDepthSnapshot) {
  return {
    format: 'typecv-inverse-depth-v1' as const,
    units: 'current-monocular-map-gauge' as const,
    pointParameters: 'pinhole-a-b-inverse-Z' as const,
    rangeParameter: 'inverse-Euclidean-range-from-fixed-reference-chart' as const,
    reference: 'fixed-coordinate-chart-not-an-extra-pose-measurement' as const,
    revision: snapshot.revision,
    assumptions: { ...snapshot.assumptions },
    uncertainty: snapshot.uncertainty && { ...snapshot.uncertainty },
    diagnostics: { ...snapshot.diagnostics },
    tracks: snapshot.tracks.map((t) => ({
      id: t.id,
      status: t.status,
      reason: t.reason,
      referenceViewId: t.referenceViewId,
      bearing: Array.from(t.bearing),
      ...(t.bearingCovariance
        ? {
            bearingCovariance: Array.from(t.bearingCovariance),
            bearingCovarianceConditionalOn: t.bearingCovarianceConditionalOn,
          }
        : {}),
      observations: t.observations.map((o) => ({ ...o, pixel: [...o.pixel] })),
      ...(t.estimate
        ? {
            estimate: {
              parameters: Array.from(t.estimate.parameters),
              reference: {
                R: Array.from(t.estimate.reference.R),
                t: Array.from(t.estimate.reference.t),
              },
            },
          }
        : {}),
      ...(t.covariance ? { covariance: Array.from(t.covariance) } : {}),
      ...(t.inverseRange ? { inverseRange: { ...t.inverseRange } } : {}),
      ...(t.maxErrorPx !== undefined
        ? { maxErrorPx: Number.isFinite(t.maxErrorPx) ? t.maxErrorPx : null }
        : {}),
    })),
  };
}
