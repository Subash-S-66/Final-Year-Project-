from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlanRow:
    label: str
    y: int | None
    count: int
    priority: str
    recommended_extra: int

    @property
    def target_total(self) -> int:
        return self.count + self.recommended_extra


PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _load_vocab(vocab_path: Path) -> list[str]:
    with vocab_path.open("r", encoding="utf-8") as f:
        tokens = json.load(f)
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
        raise ValueError(f"Invalid vocab JSON at {vocab_path}")
    return tokens


def _read_label_summary(summary_csv: Path) -> dict[str, dict[str, str]]:
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"label", "y", "count"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Summary CSV missing required columns {sorted(required)}. Got: {reader.fieldnames}"
            )
        rows: dict[str, dict[str, str]] = {}
        for r in reader:
            label = (r.get("label") or "").strip()
            if not label:
                continue
            rows[label] = r
        return rows


def _priority_and_extra(count: int) -> tuple[str, int]:
    if count <= 0:
        return "P0", 20
    if 1 <= count <= 2:
        return "P1", 15
    if 3 <= count <= 5:
        return "P2", 10
    return "P3", 0


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    s = value.strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _print_table(rows: list[PlanRow]) -> None:
    headers = [
        ("priority", 8),
        ("count", 5),
        ("extra", 5),
        ("target", 6),
        ("y", 3),
        ("label", 22),
    ]

    def fmt_cell(text: str, width: int) -> str:
        if len(text) <= width:
            return text.ljust(width)
        return (text[: width - 1] + "…") if width >= 2 else text[:width]

    line = " ".join(fmt_cell(h, w) for h, w in headers)
    sep = " ".join("-" * w for _, w in headers)
    print(line)
    print(sep)

    for r in rows:
        y_str = "" if r.y is None else str(r.y)
        parts = [
            fmt_cell(r.priority, 8),
            fmt_cell(str(r.count), 5),
            fmt_cell(str(r.recommended_extra), 5),
            fmt_cell(str(r.target_total), 6),
            fmt_cell(y_str, 3),
            fmt_cell(r.label, 22),
        ]
        print(" ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a custom data collection plan (priority buckets + extra sample targets) "
            "from an existing per-label summary."
        )
    )
    parser.add_argument("--samples", default="data/processed/cislr", help="(info) Dataset root")
    parser.add_argument("--vocab", default="configs/vocab_v1_51.json", help="Vocab JSON")
    parser.add_argument(
        "--summary",
        default="reports/cislr_label_summary.csv",
        help="Input per-label summary CSV",
    )
    parser.add_argument(
        "--out",
        default="reports/collection_plan_v1.csv",
        help="Output collection plan CSV",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    vocab_path = (repo_root / args.vocab).resolve()
    summary_path = (repo_root / args.summary).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Inputs are recorded for traceability; we do not rescan the dataset here by design.
    _ = (repo_root / args.samples).resolve()

    if not summary_path.exists():
        print(f"Missing summary CSV: {summary_path}")
        print("Run: python tools/dataset_label_summary.py ...")
        return 1

    vocab = _load_vocab(vocab_path)
    summary_rows = _read_label_summary(summary_path)

    plan_rows: list[PlanRow] = []
    dist: dict[str, int] = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}

    for idx, label in enumerate(vocab):
        r = summary_rows.get(label)
        count = _parse_int(r.get("count") if r else None) or 0
        y = _parse_int(r.get("y") if r else None)
        if y is None:
            y = idx

        priority, extra = _priority_and_extra(count)
        dist[priority] += 1
        plan_rows.append(
            PlanRow(label=label, y=y, count=count, priority=priority, recommended_extra=extra)
        )

    plan_rows.sort(key=lambda r: (PRIORITY_ORDER[r.priority], r.count, r.label))

    tokens_needing_data = sum(1 for r in plan_rows if r.recommended_extra > 0)
    total_new_samples_required = sum(r.recommended_extra for r in plan_rows)

    print("\nCollection plan (derived from summary counts)")
    print(f"- vocab: {vocab_path}")
    print(f"- summary: {summary_path}")
    print(f"- output: {out_path}\n")

    _print_table(plan_rows)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "y",
                "count",
                "priority",
                "recommended_extra",
                "target_total",
            ],
        )
        writer.writeheader()
        for r in plan_rows:
            writer.writerow(
                {
                    "label": r.label,
                    "y": r.y if r.y is not None else "",
                    "count": r.count,
                    "priority": r.priority,
                    "recommended_extra": r.recommended_extra,
                    "target_total": r.target_total,
                }
            )

    print("\nSummary totals")
    print(f"- tokens_needing_data: {tokens_needing_data}")
    print(f"- total_new_samples_required: {total_new_samples_required}")
    print("- distribution_by_priority:")
    for p in ["P0", "P1", "P2", "P3"]:
        print(f"  - {p}: {dist[p]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
