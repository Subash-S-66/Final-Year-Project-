export function MetricCard({ label, value, ariaLabel }) {
  return (
    <div className="rounded-xl border border-slate-200/80 bg-white/70 p-4 dark:border-slate-700 dark:bg-slate-900/50">
      <p className="text-xs uppercase tracking-[0.15em] text-slate-500 dark:text-slate-300">{label}</p>
      <p className="mt-2 text-xl font-semibold" aria-label={ariaLabel}>
        {value}
      </p>
    </div>
  );
}
