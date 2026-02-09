from __future__ import annotations

import argparse
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import sys

# Allow running as: python tools/cislr_download.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from tqdm import tqdm

from tools.cislr_common import (
    build_alias_lookup,
    load_aliases,
    load_vocab,
    repo_root,
    require_hf_token,
    resolve_to_vocab_match,
    validate_hf_token,
)


def _hf_download_file(repo_id: str, *, filename: str, token: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise SystemExit(
            "Missing dependency 'huggingface-hub'. Install: pip install -r training/requirements.txt"
        ) from e

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            token=token,
        )
    )


def _find_video_member(zf: zipfile.ZipFile, uid: str) -> Optional[str]:
    """Try to locate a clip inside the CISLR videos zip.

    CISLR exposes uid like '<youtubeId>_1'. The zip may store either '<uid>.mp4' or '<youtubeId>.mp4'.
    """
    uid_base = uid.split("_")[0]
    candidates = [
        f"{uid}.mp4",
        f"{uid_base}.mp4",
        f"{uid}.MP4",
        f"{uid_base}.MP4",
    ]
    # Some zips might include nested folder(s). Try prefix match.
    names = set(zf.namelist())
    for cand in candidates:
        if cand in names:
            return cand
        # Try any path suffix match
        for n in names:
            if n.endswith("/" + cand):
                return n
    return None


def _guess_field(example: dict[str, Any], candidates: list[str]) -> Optional[str]:
    for k in candidates:
        if k in example:
            return k
    return None


_SAFE_RE = re.compile(r"[^A-Z0-9_]+")


def _safe_label_dir(label: str) -> str:
    # Normalize folder name for Windows paths.
    s = label.strip().upper()
    s = s.replace(" ", "_")
    s = _SAFE_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "UNKNOWN"


def _safe_uid(uid: str) -> str:
    u = uid.strip()
    u = re.sub(r"\s+", "_", u)
    u = re.sub(r"[^a-zA-Z0-9._-]+", "_", u)
    u = u.strip("._-")
    return u[:120] if u else "unknown"


def _extract_url(row: dict[str, Any], url_field: Optional[str]) -> Optional[str]:
    if url_field and url_field in row:
        v = row.get(url_field)
        if isinstance(v, str) and v.startswith("http"):
            return v
        if isinstance(v, dict):
            u = v.get("url")
            if isinstance(u, str) and u.startswith("http"):
                return u

    # Common fields
    for k in ["url", "video_url", "mp4_url", "video", "file_url", "download_url"]:
        if k in row:
            v = row.get(k)
            if isinstance(v, str) and v.startswith("http"):
                return v
            if isinstance(v, dict):
                u = v.get("url")
                if isinstance(u, str) and u.startswith("http"):
                    return u

    # Fallback scan
    for v in row.values():
        if isinstance(v, str) and v.startswith("http"):
            return v
        if isinstance(v, dict):
            u = v.get("url")
            if isinstance(u, str) and u.startswith("http"):
                return u
    return None


def _extract_uid(row: dict[str, Any], uid_field: Optional[str], default_uid: str) -> str:
    if uid_field and uid_field in row:
        return str(row.get(uid_field) or default_uid)
    for k in ["uid", "id", "video_id", "clip_id", "name"]:
        if k in row and row.get(k) is not None:
            return str(row.get(k))
    return default_uid


def _download_to(url: str, dest: Path, *, token: str, timeout_s: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CISLR via HuggingFace Datasets and store videos by gloss.")
    parser.add_argument("--dataset", default="Exploration-Lab/CISLR", help="HF dataset name")
    parser.add_argument("--config", default=None, help="Optional HF config name")
    parser.add_argument("--split", default="test", help="Split name (train/validation/test) or 'all'")
    parser.add_argument("--out", default="data/raw/cislr/videos", help="Output folder root")
    parser.add_argument("--vocab", default="configs/vocab_v1_51.json", help="Filter to vocab tokens")
    parser.add_argument("--aliases", default="configs/cislr_aliases.json", help="Optional alias table JSON")
    parser.add_argument("--full", action="store_true", help="Download full dataset (ignore vocab filter)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (or set env HF_TOKEN / login)")
    parser.add_argument("--gloss-field", default=None, help="Override gloss field name")
    parser.add_argument("--url-field", default=None, help="Override URL field name")
    parser.add_argument("--uid-field", default=None, help="Override UID field name")
    parser.add_argument(
        "--video-subdir",
        default="CISLR_v1.5-a_videos",
        help="Subdirectory in the HF dataset repo that contains MP4 clips",
    )
    parser.add_argument(
        "--zip-name",
        default="CISLR_v1.5-a_videos.zip",
        help="Zip filename within --video-subdir",
    )
    parser.add_argument(
        "--max-per-token",
        type=int,
        default=0,
        help="Maximum number of clips to keep per vocab token (0 = unlimited). Includes existing files.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick tests")
    args = parser.parse_args()

    root = repo_root()
    out_root = (root / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab((root / args.vocab).resolve())
    vocab_set = set(vocab)
    alias_lookup = build_alias_lookup(load_aliases((root / args.aliases).resolve() if args.aliases else None))

    # Token is required for gated dataset access; we require it by default for CISLR.
    token = require_hf_token(args.hf_token, dataset_id=args.dataset)
    validate_hf_token(token)

    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit(
            "Missing dependency 'datasets'. Install: pip install -r training/requirements.txt"
        ) from e

    splits = [args.split]
    if args.split == "all":
        # We'll try common split names; user can rerun with exact split names if needed.
        splits = ["train", "validation", "test"]

    seen_splits: set[str] = set()

    downloaded = 0
    existed = 0
    skipped = 0
    skipped_max = 0
    failed = 0
    fail_reasons: Counter[str] = Counter()
    unmatched_gloss: Counter[str] = Counter()
    match_exact = 0
    match_alias = 0

    # CISLR videos are packaged as a zip in the dataset repo.
    zip_repo_path = f"{args.video_subdir}/{args.zip_name}"
    print(f"Preparing CISLR videos zip: {zip_repo_path}")
    try:
        zip_path = _hf_download_file(args.dataset, filename=zip_repo_path, token=token)
    except Exception as e:
        raise SystemExit(
            "Failed to download CISLR videos zip from HuggingFace.\n"
            "- Ensure you accepted dataset access conditions\n"
            "- Ensure your token can access gated dataset files\n"
            f"- Missing file: {zip_repo_path}\n"
            f"Error: {str(e)[:200]}"
        )

    if not zip_path.exists():
        raise SystemExit(f"Zip not found after download: {zip_path}")

    try:
        zf = zipfile.ZipFile(zip_path)
    except Exception as e:
        raise SystemExit(f"Failed to open zip: {zip_path} ({e})")

    # Track how many clips we already have per token directory.
    per_token_count: dict[str, int] = {}
    if args.max_per_token and args.max_per_token > 0:
        for d in out_root.iterdir():
            if not d.is_dir():
                continue
            per_token_count[d.name] = len(list(d.glob("*.mp4")))

    for split in splits:
        try:
            ds = load_dataset(args.dataset, args.config, split=split, token=token)
        except Exception as e:
            # Some datasets expose only a single split (often 'test'). Fall back.
            try:
                from datasets import get_dataset_split_names

                avail = get_dataset_split_names(args.dataset, args.config, token=token)
            except Exception:
                avail = []
            if avail:
                fallback = avail[0]
                print(f"Split '{split}' not available; falling back to '{fallback}'.")
                ds = load_dataset(args.dataset, args.config, split=fallback, token=token)
                split = fallback
            else:
                print(f"Failed to load split '{split}': {e}")
                continue

        # If multiple requested splits fall back to the same actual split, don't redo work.
        if split in seen_splits:
            continue
        seen_splits.add(split)

        if len(ds) == 0:
            print(f"Split '{split}' is empty")
            continue

        # Field guessing based on first example
        ex0 = ds[0]
        gloss_field = args.gloss_field or _guess_field(ex0, ["gloss", "label", "word", "text", "annotation"])

        if not gloss_field:
            raise SystemExit(
                f"Could not infer gloss field. Provide --gloss-field. Keys: {list(ex0.keys())}"
            )

        uid_field = args.uid_field or _guess_field(ex0, ["uid", "id", "video_id", "clip_id", "name"])
        print(
            f"Using fields: gloss='{gloss_field}', uid='{uid_field or 'inferred'}' for split='{split}'"
        )

        it = ds
        if args.limit and args.limit > 0:
            it = ds.select(range(min(args.limit, len(ds))))

        for idx, row in enumerate(tqdm(it, desc=f"download:{split}", unit="clip")):
            gloss_val = row.get(gloss_field)
            if gloss_val is None:
                skipped += 1
                continue
            gloss = str(gloss_val)
            match = resolve_to_vocab_match(gloss, vocab_set=vocab_set, alias_lookup=alias_lookup)

            if match is None:
                if args.full:
                    label_dir = _safe_label_dir(gloss)
                else:
                    unmatched_gloss[gloss] += 1
                    skipped += 1
                    continue
            else:
                label_dir = _safe_label_dir(match.token)
                if match.match_type == "exact":
                    match_exact += 1
                else:
                    match_alias += 1

            if args.max_per_token and args.max_per_token > 0:
                cur = per_token_count.get(label_dir, 0)
                if cur >= args.max_per_token:
                    skipped_max += 1
                    continue

            url = _extract_url(row, args.url_field)
            uid_raw = _extract_uid(row, uid_field, default_uid=f"{split}_{idx:08d}")
            uid = _safe_uid(uid_raw)
            dest = out_root / label_dir / f"{uid}.mp4"

            if dest.exists() and dest.stat().st_size > 0:
                existed += 1
                if args.max_per_token and args.max_per_token > 0:
                    per_token_count[label_dir] = per_token_count.get(label_dir, 0) + 1
                continue

            try:
                if url:
                    _download_to(url, dest, token=token)
                else:
                    member = _find_video_member(zf, uid)
                    if not member:
                        failed += 1
                        fail_reasons["missing_in_zip"] += 1
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member, "r") as src, dest.open("wb") as out:
                        out.write(src.read())
                downloaded += 1
                if args.max_per_token and args.max_per_token > 0:
                    per_token_count[label_dir] = per_token_count.get(label_dir, 0) + 1
            except Exception as e:
                failed += 1
                reason = type(e).__name__
                fail_reasons[reason] += 1

    print("\nDownload summary")
    print(f"- downloaded: {downloaded}")
    print(f"- existed:    {existed}")
    print(f"- skipped:    {skipped}")
    if args.max_per_token and args.max_per_token > 0:
        print(f"- skipped_max_per_token: {skipped_max}")
    print(f"- failed:     {failed}")
    print(f"- matches:    exact={match_exact} alias={match_alias}")

    if fail_reasons:
        print("\nFailure reasons (top 10):")
        for r, c in fail_reasons.most_common(10):
            print(f"- {r}: {c}")

    if unmatched_gloss:
        print("\nTop unmatched gloss strings (top 25):")
        for g, c in unmatched_gloss.most_common(25):
            print(f"- {g}: {c}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
