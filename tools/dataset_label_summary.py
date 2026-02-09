from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _load_vocab(vocab_path: Path) -> list[str]:
    with vocab_path.open("r", encoding="utf-8") as f:
        tokens = json.load(f)
    if not isinstance(tokens, list) or any(not isinstance(t, str) for t in tokens):
        raise ValueError(f"Invalid vocab JSON at {vocab_path}")
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create per-label dataset summary CSV and print low-sample warnings."
    )
    parser.add_argument("--samples", default="data/processed/cislr", help="Root samples folder")
    parser.add_argument(
        "--vocab",
        default="configs/vocab_v1_51.json",
        help="Vocab JSON (used to include zero-count labels)",
    )
    parser.add_argument(
        "--out",
        default="reports/cislr_label_summary.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Warn when a label has fewer than this many samples",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    samples_root = (repo_root / args.samples).resolve()
    vocab_path = (repo_root / args.vocab).resolve() if args.vocab else None
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not samples_root.exists():
        print(f"No samples folder found: {samples_root}")
        return 1

    files = list(samples_root.rglob("*.npz"))
    if not files:
        print("No .npz samples found.")
        return 0

    vocab: list[str] | None = None
    if vocab_path and vocab_path.exists():
        vocab = _load_vocab(vocab_path)

    label_counts: Counter[str] = Counter()
    label_to_y: dict[str, int] = {}

    # Count labels and sanity-check y/label mapping.
    for p in files:
        data = np.load(p, allow_pickle=False)
        label = str(data["label"])
        y = int(data["y"]) if "y" in data.files else -1
        label_counts[label] += 1

        if y >= 0:
            # First observed y wins, but we’ll warn on conflicts.
            if label in label_to_y and label_to_y[label] != y:
                print(f"WARN: label {label} has conflicting y values: {label_to_y[label]} vs {y}")
            else:
                label_to_y[label] = y

    # If vocab is present, include zero-count labels.
    all_labels: list[str]
    if vocab:
        all_labels = list(vocab)
    else:
        all_labels = sorted(label_counts.keys())

    total = sum(label_counts.values())

    rows = []
    for label in all_labels:
        count = int(label_counts.get(label, 0))
        y = label_to_y.get(label)
        if y is None and vocab:
            # y index is vocab index when known.
            try:
                y = vocab.index(label)
            except ValueError:
                y = None
        pct = (count / total * 100.0) if total > 0 else 0.0
        rows.append(
            {
                "label": label,
                "y": "" if y is None else int(y),
                "count": count,
                "percent": f"{pct:.2f}",
            }
        )

    # Sort by count desc, then label.
    rows.sort(key=lambda r: (-int(r["count"]), str(r["label"])) )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "y", "count", "percent"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {out_path}")
    print(f"Total samples: {total}")

    # Warnings
    low = [r for r in rows if int(r["count"]) > 0 and int(r["count"]) < args.min_count]
    zero = [r for r in rows if int(r["count"]) == 0]

    if low:
        print(f"\nLow-sample warnings (<{args.min_count}): {len(low)}")
        for r in low:
            print(f"- {r['label']}: {r['count']}")

    if vocab and zero:
        print(f"\nZero-sample vocab labels: {len(zero)}")
        for r in zero:
            print(f"- {r['label']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
