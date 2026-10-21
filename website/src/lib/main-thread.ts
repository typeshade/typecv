/**
 * What the page's main thread did while a run was in flight: long tasks (the Long Tasks API,
 * tasks over 50 ms) and frame gaps (`requestAnimationFrame` more than 100 ms apart). Both make
 * the page unresponsive to the visitor, and neither is the GPU's work: the SfM job runs in a
 * worker. The watch names the lab's stage at the longest of each, so a record says where the
 * page stalled on a device the developers do not have.
 */
import type { RunRecordMainThread } from './run-record';

/** A frame gap longer than this counts: six dropped frames at 60 Hz. */
export const FRAME_GAP_MS = 100;

/**
 * Start watching. `stage` returns the lab's current stage text; it is read when a long task or a
 * gap ends. `stop()` returns the summary and ends the watch.
 */
export function watchMainThread(stage: () => string): { stop(): RunRecordMainThread } {
  const summary = {
    longTasks: 0,
    longTaskMs: 0,
    longestTaskMs: 0,
    longestTaskStage: '',
    frameGaps: 0,
    frameGapMs: 0,
    longestGapMs: 0,
    longestGapStage: '',
    hiddenMs: 0,
  };
  const label = () => stage().slice(0, 128);
  let observer: PerformanceObserver | undefined;
  if (
    typeof PerformanceObserver !== 'undefined' &&
    PerformanceObserver.supportedEntryTypes?.includes('longtask')
  ) {
    observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        summary.longTasks++;
        summary.longTaskMs += entry.duration;
        if (entry.duration > summary.longestTaskMs) {
          summary.longestTaskMs = entry.duration;
          summary.longestTaskStage = label();
        }
      }
    });
    observer.observe({ entryTypes: ['longtask'] });
  }
  // A hidden tab paints no frames, so a gap that overlaps hidden time says nothing about the page.
  // The time is kept apart in `hiddenMs`, and the gap is not counted.
  let hiddenSince = typeof document !== 'undefined' && document.hidden ? performance.now() : -1;
  let hiddenInGap = hiddenSince >= 0;
  const onVisibility = () => {
    const now = performance.now();
    if (document.hidden) {
      hiddenSince = now;
      hiddenInGap = true;
    } else if (hiddenSince >= 0) {
      summary.hiddenMs += now - hiddenSince;
      hiddenSince = -1;
    }
  };
  if (typeof document !== 'undefined') document.addEventListener('visibilitychange', onVisibility);
  let last = performance.now();
  let frame = 0;
  let stopped = false;
  let result: RunRecordMainThread | undefined;
  const tick = () => {
    if (stopped) return;
    const now = performance.now();
    const gap = now - last;
    const hidden = hiddenInGap;
    hiddenInGap = hiddenSince >= 0;
    if (gap > FRAME_GAP_MS && !hidden) {
      summary.frameGaps++;
      summary.frameGapMs += gap;
      if (gap > summary.longestGapMs) {
        summary.longestGapMs = gap;
        summary.longestGapStage = label();
      }
    }
    last = now;
    frame = requestAnimationFrame(tick);
  };
  frame = requestAnimationFrame(tick);
  return {
    stop() {
      if (result) return result;
      stopped = true;
      cancelAnimationFrame(frame);
      observer?.disconnect();
      if (typeof document !== 'undefined')
        document.removeEventListener('visibilitychange', onVisibility);
      if (hiddenSince >= 0) summary.hiddenMs += performance.now() - hiddenSince;
      const round = (value: number) => Math.round(value * 10) / 10;
      return (result = {
        longTasks: summary.longTasks,
        longTaskMs: round(summary.longTaskMs),
        longestTaskMs: round(summary.longestTaskMs),
        longestTaskStage: summary.longestTaskStage,
        frameGaps: summary.frameGaps,
        frameGapMs: round(summary.frameGapMs),
        longestGapMs: round(summary.longestGapMs),
        longestGapStage: summary.longestGapStage,
        hiddenMs: round(summary.hiddenMs),
      });
    },
  };
}

/** One line for a record: `Main thread: 4 long tasks, longest 2851 ms (pnp · scorePnP); 3 frame gaps, longest 2854 ms`. */
export function mainThreadLine(summary: RunRecordMainThread | undefined): string | undefined {
  if (!summary) return undefined;
  const at = (stage: string) => (stage ? ' (' + stage + ')' : '');
  return (
    'Main thread: ' +
    summary.longTasks +
    (summary.longTasks === 1 ? ' long task' : ' long tasks') +
    (summary.longTasks
      ? ', longest ' + Math.round(summary.longestTaskMs) + ' ms' + at(summary.longestTaskStage)
      : '') +
    '; ' +
    summary.frameGaps +
    (summary.frameGaps === 1 ? ' frame gap' : ' frame gaps') +
    (summary.frameGaps
      ? ', longest ' + Math.round(summary.longestGapMs) + ' ms' + at(summary.longestGapStage)
      : '')
  );
}
