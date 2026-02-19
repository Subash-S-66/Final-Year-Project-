export function HistoryList({ history, onClear }) {
  return (
    <section className="glass-panel p-5" aria-label="Recognition history">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="font-display text-lg font-bold tracking-tight">History</h2>
        <button
          type="button"
          onClick={onClear}
          className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-100 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-800"
          aria-label="Clear recognition history"
        >
          Clear
        </button>
      </div>

      {history.length === 0 ? (
        <p className="rounded-xl border border-dashed border-slate-300/90 p-4 text-sm text-slate-600 dark:border-slate-700 dark:text-slate-300">
          No committed tokens yet.
        </p>
      ) : (
        <ul className="max-h-64 space-y-2 overflow-auto pr-1" aria-label="Committed token list">
          {history.map((item) => (
            <li
              key={item.id}
              className="rounded-xl border border-slate-200/80 bg-white/70 px-3 py-2 dark:border-slate-700 dark:bg-slate-900/50"
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold">{item.token}</span>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {(item.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {new Date(item.timestamp).toLocaleTimeString()}
              </p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
