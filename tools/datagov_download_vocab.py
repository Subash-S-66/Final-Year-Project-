from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

from tools.cislr_common import (
    build_alias_lookup,
    load_aliases,
    load_vocab,
    normalize_gloss,
    repo_root,
    resolve_to_vocab_match,
)


def _safe_name(name: str) -> str:
    n = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    n = re.sub(r"_+", "_", n).strip("._-")
    return n[:180] if n else "clip"


def _candidate_glosses_from_filename(path: str) -> list[str]:
    stem = Path(path).stem
    parts = re.split(r"[_()&,-]+", stem)
    candidates = [stem]
    candidates.extend([p for p in parts if p])
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download only vocab-matching clips from data.gov ISL HF mirror."
    )
    parser.add_argument(
        "--dataset",
        default="silentone0725/Indian_Sign_Language_Data.gov_Rencoded",
    )
    parser.add_argument("--out", default="data/Download/datagov_vocab/videos")
    parser.add_argument("--vocab", default="configs/vocab_v1_51.json")
    parser.add_argument("--aliases", default="configs/cislr_aliases.json")
    parser.add_argument(
        "--max-per-token",
        type=int,
        default=30,
        help="Limit clips per vocab token (0 = unlimited).",
    )
    args = parser.parse_args()

    root = repo_root()
    out_root = (root / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab((root / args.vocab).resolve())
    vocab_set = set(vocab)
    alias_lookup = build_alias_lookup(load_aliases((root / args.aliases).resolve()))

    print(f"Listing files from {args.dataset} ...")
    files = list_repo_files(args.dataset, repo_type="dataset")
    mp4s = [f for f in files if f.lower().endswith(".mp4")]
    print(f"Found {len(mp4s)} mp4 files")

    per_token: dict[str, int] = {}
    downloaded = 0
    existed = 0
    skipped = 0
    matched = 0

    for rel in tqdm(mp4s, desc="download-vocab", unit="clip"):
        matched_token = None

        for cand in _candidate_glosses_from_filename(rel):
            m = resolve_to_vocab_match(
                normalize_gloss(cand),
                vocab_set=vocab_set,
                alias_lookup=alias_lookup,
            )
            if m is not None:
                matched_token = m.token
                break

        if matched_token is None:
            skipped += 1
            continue
        matched += 1

        if args.max_per_token and args.max_per_token > 0:
            if per_token.get(matched_token, 0) >= args.max_per_token:
                skipped += 1
                continue

        src_name = _safe_name(Path(rel).name)
        parent_tag = _safe_name(Path(rel).parent.name)
        dst = out_root / matched_token / f"{parent_tag}__{src_name}"
        if dst.exists() and dst.stat().st_size > 0:
            existed += 1
            per_token[matched_token] = per_token.get(matched_token, 0) + 1
            continue

        try:
            local_path = hf_hub_download(
                repo_id=args.dataset,
                repo_type="dataset",
                filename=rel,
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            Path(dst).write_bytes(Path(local_path).read_bytes())
            downloaded += 1
            per_token[matched_token] = per_token.get(matched_token, 0) + 1
        except Exception:
            skipped += 1

    print("\nDownload summary")
    print(f"- matched files: {matched}")
    print(f"- downloaded:    {downloaded}")
    print(f"- existed:       {existed}")
    print(f"- skipped:       {skipped}")

    print("\nPer-token counts")
    for tok in vocab:
        c = per_token.get(tok, 0)
        if c > 0:
            print(f"- {tok}: {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

