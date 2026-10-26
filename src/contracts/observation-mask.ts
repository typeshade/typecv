export interface CameraObservationMask {
  readonly viewId: number;
  /** 1 permits a background hypothesis; 0 separates the feature from camera/static-map estimation. */
  readonly values: Uint8Array;
  readonly source: 'manual' | 'geometry' | 'learned' | 'ground-truth';
}
