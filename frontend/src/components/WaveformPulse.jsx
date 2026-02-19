export function WaveformPulse({ active, intensity = 0 }) {
  const barCount = 10;
  const clamped = Math.max(0, Math.min(1, intensity));

  return (
    <div className="relative rounded-xl border border-slate-200/80 bg-white/70 p-4 dark:border-slate-700 dark:bg-slate-900/50">
      <p className="mb-3 text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-300">Gesture activity</p>
      <div className="relative flex h-16 items-end justify-center gap-1" aria-label="Gesture activity waveform">
        {active && (
          <div className="pointer-events-none absolute inset-0 animate-pulseGlow rounded-lg bg-teal-400/15 dark:bg-teal-300/10" />
        )}
        {Array.from({ length: barCount }).map((_, index) => {
          const weight = 0.55 + Math.sin((index / (barCount - 1)) * Math.PI) * 0.45;
          const height = 12 + Math.round(42 * clamped * weight);
          return (
            <span
              // Slightly staggered animation gives a waveform feel.
              key={`bar-${index}`}
              className={`w-2 origin-bottom rounded-full ${
                active ? "animate-wave bg-teal-500 dark:bg-teal-300" : "bg-slate-300 dark:bg-slate-700"
              }`}
              style={{
                height: `${height}px`,
                animationDelay: `${index * 0.08}s`,
                animationDuration: `${Math.max(0.55, 1.15 - clamped * 0.45)}s`,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
