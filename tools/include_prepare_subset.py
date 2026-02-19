from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}


def _safe_name(x: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", x)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s or "clip"


def _content_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract selected INCLUDE labels from Zenodo zips into token folders."
    )
    parser.add_argument("--zips", default="data/Download/include/zips")
    parser.add_argument("--out", default="data/raw/include30/videos")
    parser.add_argument("--label-map", default="configs/include30_label_map.json")
    parser.add_argument("--max-per-token", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--clear-out", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    zips_root = (root / args.zips).resolve()
    out_root = (root / args.out).resolve()
    label_map_path = (root / args.label_map).resolve()

    if not zips_root.exists():
        print(f"Missing zips folder: {zips_root}")
        return 1
    if not label_map_path.exists():
        print(f"Missing label map: {label_map_path}")
        return 1

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    if not isinstance(label_map, dict) or not label_map:
        print(f"Invalid label map JSON: {label_map_path}")
        return 1

    if args.clear_out and out_root.exists():
        for p in out_root.rglob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)
        for p in sorted(out_root.rglob("*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
    out_root.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(zips_root.glob("*.zip"))
    if not zip_files:
        print(f"No zip files found under: {zips_root}")
        return 1

    # Existing clip counts if continuing.
    per_token = Counter()
    for d in out_root.iterdir():
        if d.is_dir():
            per_token[d.name] = len([p for p in d.rglob("*") if p.is_file()])

    # De-duplicate by content hash across all zips.
    seen_hashes: set[str] = set()
    added = 0
    skipped_dup = 0
    skipped_label = 0
    skipped_cap = 0
    by_token_added = Counter()
    available_by_token = Counter()

    for zp in zip_files:
        try:
            with zipfile.ZipFile(zp) as zf:
                for name in zf.namelist():
                    parts = [p for p in name.split("/") if p]
                    if len(parts) < 3:
                        continue
                    ext = Path(parts[-1]).suffix.lower()
                    if ext not in VIDEO_EXTS:
                        continue

                    raw_label = re.sub(r"^\d+\.\s*", "", parts[1]).strip()
                    token = label_map.get(raw_label)
                    if not token:
                        skipped_label += 1
                        continue
                    available_by_token[token] += 1

                    if args.max_per_token and per_token[token] >= args.max_per_token:
                        skipped_cap += 1
                        continue

                    data = zf.read(name)
                    h = _content_hash(data)
                    if h in seen_hashes:
                        skipped_dup += 1
                        continue
                    seen_hashes.add(h)

                    stem = _safe_name(Path(parts[-1]).stem)
                    dst_name = f"{_safe_name(zp.stem)}__{stem}{ext}"
                    dst = out_root / token / dst_name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(data)

                    per_token[token] += 1
                    by_token_added[token] += 1
                    added += 1
        except Exception as e:
            print(f"WARN: failed zip {zp.name}: {e}")

    print("\nINCLUDE subset extraction summary")
    print(f"- zips_scanned:    {len(zip_files)}")
    print(f"- clips_added:     {added}")
    print(f"- skipped_label:   {skipped_label}")
    print(f"- skipped_dup:     {skipped_dup}")
    print(f"- skipped_cap:     {skipped_cap}")
    print(f"- output:          {out_root}")

    print("\nPer-token counts (available -> extracted)")
    for raw_label, token in sorted(label_map.items(), key=lambda kv: kv[1]):
        avail = int(available_by_token.get(token, 0))
        extd = int(per_token.get(token, 0))
        print(f"- {token}: {avail} -> {extd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
