/** Source-video timestamps in seconds; the end is exclusive. */
export interface VideoRange {
  start: number;
  end: number;
}

export function videoRangeValid(value: unknown, duration = Infinity): value is VideoRange {
  if (!value || typeof value !== 'object') return false;
  const range = value as VideoRange;
  return (
    Number.isFinite(range.start) &&
    Number.isFinite(range.end) &&
    range.start >= 0 &&
    range.end > range.start &&
    range.end <= duration + 1e-6
  );
}

/** Intersect a requested selection with the actual track, without widening it to the whole video. */
export function videoRangeBounds(first: number, end: number, range?: VideoRange): VideoRange {
  if (range && !videoRangeValid(range)) throw new RangeError('Invalid analysis range.');
  const start = Math.max(first, range?.start ?? first);
  const stop = Math.min(end, range?.end ?? end);
  if (range && stop <= start) throw new RangeError('The analysis range contains no video frames.');
  return { start, end: stop };
}

export function videoRangeProgress(time: number, duration: number, range?: VideoRange): number {
  const start = range?.start ?? 0;
  const span = (range?.end ?? duration) - start;
  return span > 0 ? Math.max(0, Math.min(1, (time - start) / span)) : 0;
}

export function formatVideoTime(seconds: number): string {
  const hundredths = Math.round(Math.max(0, seconds) * 100);
  const minutes = Math.floor(hundredths / 6000);
  return `${minutes}:${((hundredths % 6000) / 100).toFixed(2).padStart(5, '0')}`;
}
