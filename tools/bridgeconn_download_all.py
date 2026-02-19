from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cislr_common import normalize_gloss, repo_root


_SAFE_RE = re.compile(r"[^A-Z0-9_]+")


def _safe_label(label: str) -> str:
    s = normalize_gloss(label)
    s = _SAFE_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "UNKNOWN"


def _safe_name(name: str, default: str) -> str:
    n = (name or default).strip()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"[^a-zA-Z0-9._-]+", "_", n).strip("._-")
    return n[:120] if n else default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ALL bridgeconn/sign-dictionary-isl clips (no vocab filtering)."
    )
    parser.add_argument("--dataset", default="bridgeconn/sign-dictionary-isl")
    parser.add_argument("--config", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/Download/bridgeconn_all/videos")
    args = parser.parse_args()

    root = repo_root()
    out_root = (root / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit(
            "Missing dependency 'datasets'. Install: pip install -r training/requirements.txt"
        ) from e

    print(f"Loading {args.dataset} ({args.config}, {args.split})...")
    ds = load_dataset(args.dataset, args.config, split=args.split)

    downloaded = 0
    existed = 0
    skipped = 0
    failed = 0

    for idx, row in enumerate(tqdm(ds, desc="download-all", unit="clip")):
        meta = row.get("json")
        if not isinstance(meta, dict):
            skipped += 1
            continue

        transcript = meta.get("transcript")
        if not isinstance(transcript, dict):
            skipped += 1
            continue

        gloss_raw = transcript.get("text")
        if not gloss_raw:
            skipped += 1
            continue

        label = _safe_label(str(gloss_raw))
        mp4_bytes = row.get("mp4")
        if not isinstance(mp4_bytes, bytes):
            failed += 1
            continue

        uid = _safe_name(str(row.get("__key__", f"clip_{idx:06d}")), default=f"clip_{idx:06d}")
        dest = out_root / label / f"{uid}.mp4"

        if dest.exists() and dest.stat().st_size > 0:
            existed += 1
            continue

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(mp4_bytes)
            downloaded += 1
        except Exception:
            failed += 1

    print("\nDownload summary")
    print(f"- downloaded: {downloaded}")
    print(f"- existed:    {existed}")
    print(f"- skipped:    {skipped}")
    print(f"- failed:     {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

