import { getConfidenceTier } from "../utils/confidence";

export function PredictionCard({
  unstableToken,
  unstableConfidence,
  committedToken,
  committedConfidence,
  stabilizationProgress,
  isStabilizing,
}) {
  const unstableTier = getConfidenceTier(unstableConfidence);
  const committedTier = getConfidenceTier(committedConfidence);

  return (
    <section className="glass-panel p-5" aria-label="Prediction panel">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="font-display text-lg font-bold tracking-tight">Prediction</h2>
        <span
          className={`rounded-full px-3 py-1 text-xs font-semibold ${isStabilizing ? "bg-amber-500/20 text-amber-700 dark:text-amber-200" : "bg-emerald-500/20 text-emerald-700 dark:text-emerald-200"}`}
          aria-label={isStabilizing ? "Prediction stabilizing" : "Prediction committed"}
        >
          {isStabilizing ? "Unstable" : "Committed"}
        </span>
      </div>

      <div className="space-y-4">
        <div className="rounded-xl border border-slate-200/70 bg-white/70 p-4 dark:border-slate-700 dark:bg-slate-900/50">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-300">Live token</p>
          <p className="mt-2 text-2xl font-semibold">{unstableToken || "NO_SIGN"}</p>
          <div className="mt-3">
            <div className="mb-1 flex items-center justify-between text-sm">
              <span className="text-slate-600 dark:text-slate-300">Confidence</span>
              <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${unstableTier.chipClass}`}>
                {unstableTier.label}
              </span>
            </div>
            <div className="h-2 rounded-full bg-slate-200/80 dark:bg-slate-700/70" role="meter" aria-valuenow={Math.round(unstableConfidence * 100)} aria-valuemin={0} aria-valuemax={100} aria-label="Live confidence">
              <div className={`h-2 rounded-full transition-all duration-300 ${unstableTier.colorClass}`} style={{ width: `${Math.max(0, Math.min(100, unstableConfidence * 100))}%` }} />
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200/70 bg-white/70 p-4 dark:border-slate-700 dark:bg-slate-900/50">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-300">Committed token</p>
          <p className="mt-2 text-2xl font-semibold">{committedToken || "NO_SIGN"}</p>
          <div className="mt-3">
            <div className="mb-1 flex items-center justify-between text-sm">
              <span className="text-slate-600 dark:text-slate-300">Confidence</span>
              <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${committedTier.chipClass}`}>
                {committedTier.label}
              </span>
            </div>
            <div className="h-2 rounded-full bg-slate-200/80 dark:bg-slate-700/70" role="meter" aria-valuenow={Math.round(committedConfidence * 100)} aria-valuemin={0} aria-valuemax={100} aria-label="Committed confidence">
              <div className={`h-2 rounded-full transition-all duration-300 ${committedTier.colorClass}`} style={{ width: `${Math.max(0, Math.min(100, committedConfidence * 100))}%` }} />
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200/70 bg-white/70 p-4 dark:border-slate-700 dark:bg-slate-900/50">
          <div className="mb-1 flex items-center justify-between text-sm">
            <span className="text-slate-600 dark:text-slate-300">Debounce progress</span>
            <span className="font-semibold">{Math.round(stabilizationProgress * 100)}%</span>
          </div>
          <div className="h-2 rounded-full bg-slate-200/80 dark:bg-slate-700/70" aria-label="Stabilization progress">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${
                isStabilizing ? "bg-amber-400" : "bg-emerald-500"
              }`}
              style={{ width: `${Math.max(0, Math.min(100, stabilizationProgress * 100))}%` }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
