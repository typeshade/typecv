/** One stable track for measured work and waits; unknown work never gets a guessed percent. */
export default function WorkProgress({ label, percent }: { label: string; percent?: number }) {
  const value = percent === undefined ? undefined : Math.max(0, Math.min(100, percent));
  return (
    <div
      className="lab-work-progress"
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={value}
      aria-valuetext={value === undefined ? label : `${label} · ${value}%`}
      data-indeterminate={value === undefined || undefined}
    >
      <span style={{ width: value === undefined ? undefined : `${value}%` }} />
    </div>
  );
}
