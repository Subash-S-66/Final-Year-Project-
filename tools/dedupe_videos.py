from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find and remove duplicate video files by exact content hash."
    )
    parser.add_argument("--root", default="data/raw/cislr/videos", help="Video root to scan")
    parser.add_argument("--glob", default="*.mp4", help="File pattern under root")
    parser.add_argument(
        "--action",
        choices=["report", "move", "delete"],
        default="report",
        help="report: detect only, move: move duplicates to quarantine, delete: permanently delete duplicates",
    )
    parser.add_argument(
        "--quarantine",
        default="data/raw/cislr/duplicates_quarantine",
        help="Target folder used when --action move",
    )
    parser.add_argument("--report-csv", default="reports/raw_cislr_duplicate_report.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root = (repo_root / args.root).resolve()
    quarantine = (repo_root / args.quarantine).resolve()
    report_csv = (repo_root / args.report_csv).resolve()
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"Missing root: {root}")
        return 1

    files = sorted([p for p in root.rglob(args.glob) if p.is_file()])
    if not files:
        print(f"No files found under {root} with glob {args.glob}")
        return 0

    by_sig: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for p in files:
        sig = (p.stat().st_size, _sha1(p))
        by_sig[sig].append(p)

    dup_groups = [g for g in by_sig.values() if len(g) > 1]
    rows: list[dict[str, str]] = []
    moved = 0
    deleted = 0

    for g in dup_groups:
        keep = g[0]
        for p in g[1:]:
            rows.append(
                {
                    "keep": str(keep),
                    "duplicate": str(p),
                    "size_bytes": str(p.stat().st_size),
                }
            )
            if args.action == "move":
                rel = p.relative_to(root)
                dst = quarantine / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                p.rename(dst)
                moved += 1
            elif args.action == "delete":
                p.unlink(missing_ok=True)
                deleted += 1

    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["keep", "duplicate", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)

    total_dups = sum(len(g) - 1 for g in dup_groups)
    print(f"Scanned files: {len(files)}")
    print(f"Duplicate groups: {len(dup_groups)}")
    print(f"Duplicate files: {total_dups}")
    if args.action == "move":
        print(f"Moved: {moved} -> {quarantine}")
    elif args.action == "delete":
        print(f"Deleted: {deleted}")
    print(f"Report: {report_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

