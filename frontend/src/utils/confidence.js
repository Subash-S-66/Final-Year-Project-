export function getConfidenceTier(confidence = 0) {
  if (confidence < 0.4) {
    return {
      label: "Low",
      colorClass: "bg-slate-500",
      chipClass: "bg-slate-500/20 text-slate-700 dark:text-slate-200",
    };
  }
  if (confidence < 0.75) {
    return {
      label: "Medium",
      colorClass: "bg-amber-400",
      chipClass: "bg-amber-400/25 text-amber-700 dark:text-amber-200",
    };
  }
  return {
    label: "High",
    colorClass: "bg-emerald-500",
    chipClass: "bg-emerald-500/20 text-emerald-700 dark:text-emerald-200",
  };
}
