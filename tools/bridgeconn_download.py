from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter

from tqdm import tqdm

from tools.cislr_common import (
    build_alias_lookup,
    load_aliases,
    load_vocab,
    normalize_gloss,
    repo_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download bridgeconn/sign-dictionary-isl and save mp4 clips by vocab token."
    )
    parser.add_argument("--dataset", default="bridgeconn/sign-dictionary-isl")
    parser.add_argument("--config", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/raw/bridgeconn/videos")
    parser.add_argument("--vocab", default="configs/vocab_v1_51.json")
    parser.add_argument("--aliases", default="configs/cislr_aliases.json")
    parser.add_argument("--max-per-token", type=int, default=0)
    args = parser.parse_args()

    root = repo_root()
    out_root = (root / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab((root / args.vocab).resolve())
    vocab_set = set(vocab)
    alias_lookup = build_alias_lookup(
        load_aliases((root / args.aliases).resolve() if args.aliases else None)
    )

    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit(
            "Missing dependency 'datasets'. Install: pip install -r training/requirements.txt"
        ) from e

    print(f"Loading {args.dataset} ({args.config}, {args.split})...")
    ds = load_dataset(args.dataset, args.config, split=args.split)

    # Track per-token counts for --max-per-token
    per_token_count: dict[str, int] = {}
    if args.max_per_token and args.max_per_token > 0:
        for d in out_root.iterdir():
            if not d.is_dir():
                continue
            per_token_count[d.name] = len(list(d.glob("*.mp4")))

    downloaded = 0
    existed = 0
    skipped = 0
    skipped_max = 0
    failed = 0
    match_exact = 0
    match_alias = 0
    unmatched: Counter[str] = Counter()

    for idx, row in enumerate(tqdm(ds, desc="download", unit="clip")):
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

        gloss = normalize_gloss(str(gloss_raw))

        # Check vocab match
        label_dir = None
        if gloss in vocab_set:
            label_dir = gloss
            match_exact += 1
        elif gloss in alias_lookup:
            canonical = alias_lookup[gloss]
            if canonical in vocab_set:
                label_dir = canonical
                match_alias += 1

        if not label_dir:
            unmatched[str(gloss_raw)] += 1
            skipped += 1
            continue

        # Check max-per-token
        if args.max_per_token and args.max_per_token > 0:
            cur = per_token_count.get(label_dir, 0)
            if cur >= args.max_per_token:
                skipped_max += 1
                continue

        # Extract mp4 bytes
        mp4_bytes = row.get("mp4")
        if not isinstance(mp4_bytes, bytes):
            failed += 1
            continue

        # Use dataset row key as UID
        uid = row.get("__key__", f"clip_{idx:06d}")
        # Sanitize uid for filename
        uid_safe = uid.replace("/", "_").replace("\\", "_").replace(" ", "_")

        dest = out_root / label_dir / f"{uid_safe}.mp4"

        if dest.exists() and dest.stat().st_size > 0:
            existed += 1
            if args.max_per_token and args.max_per_token > 0:
                per_token_count[label_dir] = per_token_count.get(label_dir, 0) + 1
            continue

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(mp4_bytes)
            downloaded += 1
            if args.max_per_token and args.max_per_token > 0:
                per_token_count[label_dir] = per_token_count.get(label_dir, 0) + 1
        except Exception as e:
            failed += 1
            print(f"Failed to save {dest}: {e}")

    print("\nDownload summary")
    print(f"- downloaded: {downloaded}")
    print(f"- existed:    {existed}")
    print(f"- skipped:    {skipped}")
    if args.max_per_token and args.max_per_token > 0:
        print(f"- skipped_max_per_token: {skipped_max}")
    print(f"- failed:     {failed}")
    print(f"- matches:    exact={match_exact} alias={match_alias}")

    if unmatched:
        print("\nTop unmatched glosses (top 25):")
        for g, c in unmatched.most_common(25):
            print(f"- {g}: {c}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
