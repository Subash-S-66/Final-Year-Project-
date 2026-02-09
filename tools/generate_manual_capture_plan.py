from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenPlan:
    label: str
    y: int
    count: int
    priority: str
    recommended_extra: int


PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _read_collection_plan(path: Path) -> list[TokenPlan]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"label", "y", "count", "priority", "recommended_extra"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"collection plan CSV missing columns: {sorted(required)}")

        out: list[TokenPlan] = []
        for r in reader:
            label = (r.get("label") or "").strip()
            if not label:
                continue
            out.append(
                TokenPlan(
                    label=label,
                    y=int(r["y"]),
                    count=int(r["count"]),
                    priority=str(r["priority"]).strip(),
                    recommended_extra=int(r["recommended_extra"]),
                )
            )
        return out


def _session_split(total_extra: int, session: str) -> int:
    """Strategy C split across 3 sessions.

    - P0 total 20 -> 8/6/6 (S1/S2/S3)
    - P1 total 15 -> 5/5/5
    - P2 total 10 -> 4/3/3 (not used in Mode B2 here)
    """
    if total_extra == 20:
        return {"S1": 8, "S2": 6, "S3": 6}[session]
    if total_extra == 15:
        return {"S1": 5, "S2": 5, "S3": 5}[session]
    if total_extra == 10:
        return {"S1": 4, "S2": 3, "S3": 3}[session]
    # Fallback: even split (ceiling-ish first)
    base = total_extra // 3
    rem = total_extra % 3
    if session == "S1":
        return base + (1 if rem >= 1 else 0)
    if session == "S2":
        return base + (1 if rem >= 2 else 0)
    return base


def _fmt_table(rows: list[dict[str, str]]) -> str:
    cols = [
        ("priority", 8),
        ("count", 5),
        ("extra", 5),
        ("S1", 3),
        ("S2", 3),
        ("S3", 3),
        ("y", 3),
        ("label", 22),
    ]

    def cell(s: str, w: int) -> str:
        if len(s) <= w:
            return s.ljust(w)
        return (s[: w - 1] + "…") if w >= 2 else s[:w]

    lines = []
    lines.append(" ".join(cell(h, w) for h, w in cols))
    lines.append(" ".join("-" * w for _, w in cols))
    for r in rows:
        lines.append(
            " ".join(
                [
                    cell(r["priority"], 8),
                    cell(r["count"], 5),
                    cell(r["extra"], 5),
                    cell(r["S1"], 3),
                    cell(r["S2"], 3),
                    cell(r["S3"], 3),
                    cell(r["y"], 3),
                    cell(r["label"], 22),
                ]
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Phase-6B Mode B2 (webcam) capture checklist + PowerShell batch commands."
    )
    parser.add_argument(
        "--collection-plan",
        default="reports/collection_plan_v1.csv",
        help="Input collection plan CSV",
    )
    parser.add_argument(
        "--out-md",
        default="reports/phase6b_mode_b2_manual_v1.md",
        help="Output Markdown plan",
    )
    parser.add_argument(
        "--out-ps1",
        default="reports/capture_manual_v1_mode_b2.ps1",
        help="Output PowerShell batch script (NOT executed)",
    )
    parser.add_argument(
        "--out-root",
        default="data/processed/manual_v1",
        help="Output root folder passed to capture_samples.py",
    )
    parser.add_argument("--person", default="p0", help="Single signer/person id")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--fps", type=float, default=15.0, help="Capture fps")
    parser.add_argument("--T", type=int, default=30, help="Sequence length")
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds per sample")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    plan_path = (repo_root / args.collection_plan).resolve()
    out_md = (repo_root / args.out_md).resolve()
    out_ps1 = (repo_root / args.out_ps1).resolve()

    tokens = _read_collection_plan(plan_path)
    tokens = [t for t in tokens if t.priority in {"P0", "P1"}]
    tokens.sort(key=lambda t: (PRIORITY_ORDER[t.priority], t.count, t.label))

    table_rows: list[dict[str, str]] = []
    total_s = {"S1": 0, "S2": 0, "S3": 0}

    for t in tokens:
        s1 = _session_split(t.recommended_extra, "S1")
        s2 = _session_split(t.recommended_extra, "S2")
        s3 = _session_split(t.recommended_extra, "S3")
        total_s["S1"] += s1
        total_s["S2"] += s2
        total_s["S3"] += s3
        table_rows.append(
            {
                "priority": t.priority,
                "count": str(t.count),
                "extra": str(t.recommended_extra),
                "S1": str(s1),
                "S2": str(s2),
                "S3": str(s3),
                "y": str(t.y),
                "label": t.label,
            }
        )

    table = _fmt_table(table_rows)

    # Print checklist to stdout.
    print("\nPHASE-6B / Mode B2 (Webcam + Real-Time Extraction)")
    print(f"- input collection plan: {plan_path}")
    print(f"- output root: {args.out_root}")
    print(f"- person: {args.person}\n")
    print(table)
    print("\nSession totals (number of samples to capture):")
    print(f"- S1: {total_s['S1']}")
    print(f"- S2: {total_s['S2']}")
    print(f"- S3: {total_s['S3']}")
    print(f"- total: {sum(total_s.values())}")

    # Write Markdown plan.
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# PHASE-6B — Mode B2 (Webcam + Real-Time Extraction)\n")
    md.append("## Inputs\n")
    md.append(f"- Dataset: `{args.out_root}` (capture output)\n")
    md.append(f"- Collection plan: `{args.collection_plan}`\n")
    md.append("\n## Strategy\n")
    md.append("- Single signer, multi-session (C)\n")
    md.append("- 3 sessions: S1 baseline, S2 mild angle/light shift, S3 faster signing\n")
    md.append("- Included tokens: P0 + P1 only\n")
    md.append("\n## Token checklist (P0 + P1)\n")
    md.append("```text\n" + table + "\n```\n")
    md.append("\n## Session schedule\n")
    md.append("| Session | Goal | Setup | Notes |\n")
    md.append("|---|---|---|---|\n")
    md.append("| S1 | Baseline lighting + angle | Same spot, stable lighting | Prioritize clean, consistent form |\n")
    md.append("| S2 | Mild angle change + lighting shift | Slight left/right shift; slightly brighter/dimmer | Keep background similar |\n")
    md.append("| S3 | Slightly faster signing speed | Same as S1 or S2 | Increase tempo, keep clarity |\n")
    md.append("\n## Expected workload\n")
    md.append(f"- S1 samples: {total_s['S1']}\n")
    md.append(f"- S2 samples: {total_s['S2']}\n")
    md.append(f"- S3 samples: {total_s['S3']}\n")
    md.append(f"- Total new samples: {sum(total_s.values())}\n")
    md.append("\n## Capture (PowerShell)\n")
    md.append("Run the generated script:\n")
    md.append(f"- `{args.out_ps1}`\n")
    out_md.write_text("".join(md), encoding="utf-8")

    # Write PowerShell batch script.
    ps1_lines = []
    ps1_lines.append("# Auto-generated. Runs capture per token for 3 sessions.\n")
    ps1_lines.append("# This does NOT run automatically; you must execute this .ps1 yourself.\n\n")
    ps1_lines.append("$ErrorActionPreference = 'Stop'\n")
    ps1_lines.append("$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent\n")
    ps1_lines.append("$Py = Join-Path $RepoRoot '.venv\\Scripts\\python.exe'\n")
    ps1_lines.append("$Tool = Join-Path $RepoRoot 'tools\\capture_samples.py'\n")
    ps1_lines.append(f"$OutRoot = '{args.out_root}'\n")
    ps1_lines.append(f"$Person = '{args.person}'\n")
    ps1_lines.append(f"$Camera = {args.camera}\n")
    ps1_lines.append(f"$Fps = {args.fps}\n")
    ps1_lines.append(f"$T = {args.T}\n")
    ps1_lines.append(f"$Duration = {args.duration}\n\n")

    def emit_session(session: str) -> None:
        ps1_lines.append(f"Write-Host '\n=== {session} ===' -ForegroundColor Cyan\n")
        ps1_lines.append(f"Write-Host 'Output: ' $OutRoot ' Person: ' $Person\n")
        for t in tokens:
            n = _session_split(t.recommended_extra, session)
            if n <= 0:
                continue
            ps1_lines.append(
                f"& $Py $Tool --label {t.label} --person $Person --session {session} --out $OutRoot "
                f"--camera $Camera --fps $Fps --T $T --duration $Duration --count {n}\n"
            )

    emit_session("S1")
    emit_session("S2")
    emit_session("S3")

    out_ps1.write_text("".join(ps1_lines), encoding="utf-8")

    print(f"\nWrote Markdown plan: {out_md}")
    print(f"Wrote PowerShell batch script: {out_ps1}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
