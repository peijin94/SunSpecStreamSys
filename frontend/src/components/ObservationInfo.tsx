import * as React from 'react';

const ovroTime = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/Los_Angeles',
  year: 'numeric',
  month: 'short',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hourCycle: 'h23',
  timeZoneName: 'short',
});

export function ObservationInfo({ elevation }: { elevation: number | null }) {
  const [now, setNow] = React.useState(() => new Date());

  React.useEffect(() => {
    const interval = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(interval);
  }, []);

  const utc = now.toISOString();
  return (
    <dl aria-label="Observation time and Sun elevation" className="flex flex-wrap gap-x-6 gap-y-1 text-sm tabular-nums">
      <div className="flex gap-2">
        <dt className="text-slate-400">UTC</dt>
        <dd><time dateTime={utc}>{utc.slice(0, 19).replace('T', ' ')}</time></dd>
      </div>
      <div className="flex gap-2">
        <dt className="text-slate-400">OVRO local</dt>
        <dd><time dateTime={utc}>{ovroTime.format(now)}</time></dd>
      </div>
      <div className="flex gap-2" title="Solar elevation at OVRO, updated every minute">
        <dt className="text-slate-400">Sun elevation</dt>
        <dd>{elevation !== null && Number.isFinite(elevation) ? `${elevation.toFixed(1)}°` : '—'}</dd>
      </div>
    </dl>
  );
}
