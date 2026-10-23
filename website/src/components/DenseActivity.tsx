import { useEffect, useState } from 'react';
import { denseActivityText, type DenseActivity as Activity } from '../lib/dense-activity';

/** Visible during the quiet tail of the MVS, including a GPU readback or synchronous cleanup. */
export default function DenseActivity({ activity, ko }: { activity: Activity; ko: boolean }) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(performance.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const elapsed = Math.floor((activity.elapsedMs + Math.max(0, now - activity.receivedAt)) / 1000);
  const label = denseActivityText(activity, ko);
  return (
    <div className="sfm-dense-activity" data-phase={activity.phase} role="status">
      <strong>{label}</strong>
      <progress
        aria-label={label}
        max={activity.total > 0 ? activity.total : 1}
        value={activity.total > 0 ? Math.min(activity.done, activity.total) : undefined}
      />
      <span>
        {ko ? `Dense 작업 ${elapsed}초` : `Dense work ${elapsed} s`}
        {activity.total > 0 &&
          ` · ${ko ? '현재 작업' : 'Current operation'} ${Math.floor((100 * activity.done) / activity.total)}%`}
        {(activity.points ?? 0) > 0 &&
          (ko
            ? ` · 추출 ${activity.points!.toLocaleString()}점`
            : ` · ${activity.points!.toLocaleString()} points extracted`)}
      </span>
    </div>
  );
}
