from __future__ import annotations

import argparse
from collections import Counter

import sys
from pathlib import Path

# Allow running as: python tools/cislr_coverage.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cislr_common import (
    build_alias_lookup,
    load_aliases,
    load_vocab,
    require_hf_token,
    resolve_to_vocab_match,
    validate_hf_token,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute CISLR gloss coverage against configs/vocab_v1_51.json (with optional aliasing)."
    )
    parser.add_argument("--dataset", default="Exploration-Lab/CISLR", help="HF dataset name")
    parser.add_argument("--config", default=None, help="Optional HF config name")
    parser.add_argument("--split", default="test", help="Split name or 'all'")
    parser.add_argument("--vocab", default="configs/vocab_v1_51.json")
    parser.add_argument("--aliases", default="configs/cislr_aliases.json")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--gloss-field", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    vocab_set = set(vocab)
    alias_lookup = build_alias_lookup(load_aliases(args.aliases))
    token = require_hf_token(args.hf_token, dataset_id=args.dataset)
    validate_hf_token(token)

    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit("Missing dependency 'datasets'. Install: pip install -r training/requirements.txt") from e

    splits = [args.split]
    if args.split == "all":
        splits = ["train", "validation", "test"]

    exact: Counter[str] = Counter()
    alias: Counter[str] = Counter()
    unmatched: Counter[str] = Counter()

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
            else:
                print(f"Failed to load split '{split}': {e}")
                continue

        if len(ds) == 0:
            continue

        ex0 = ds[0]
        gloss_field = args.gloss_field
        if not gloss_field:
            for cand in ["gloss", "label", "word", "text", "annotation"]:
                if cand in ex0:
                    gloss_field = cand
                    break
        if not gloss_field:
            raise SystemExit(f"Cannot infer gloss field; keys: {list(ex0.keys())}")

        it = ds
        if args.limit and args.limit > 0:
            it = ds.select(range(min(args.limit, len(ds))))

        for row in it:
            g = row.get(gloss_field)
            if g is None:
                continue
            m = resolve_to_vocab_match(str(g), vocab_set=vocab_set, alias_lookup=alias_lookup)
            if m is None:
                unmatched[str(g)] += 1
            elif m.match_type == "exact":
                exact[m.token] += 1
            else:
                alias[m.token] += 1

    print("\nCoverage summary")
    total_vocab = len(vocab)
    covered_map = Counter()
    covered_map.update(exact)
    covered_map.update(alias)
    covered_tokens = len([t for t in vocab if covered_map.get(t, 0) > 0])
    print(f"- vocab tokens: {total_vocab}")
    print(f"- covered tokens: {covered_tokens}")
    print(f"- uncovered tokens: {total_vocab - covered_tokens}")

    print("\nMatch breakdown:")
    print(f"- exact matches: {sum(exact.values())}")
    print(f"- alias matches: {sum(alias.values())}")
    print(f"- unmatched:     {sum(unmatched.values())}")

    print("\nTop exact-matched tokens (top 25):")
    for t, c in exact.most_common(25):
        print(f"- {t}: {c}")

    print("\nTop alias-matched tokens (top 25):")
    for t, c in alias.most_common(25):
        print(f"- {t}: {c}")

    print("\nUncovered tokens:")
    for t in vocab:
        if covered_map.get(t, 0) == 0:
            print(f"- {t}")

    if unmatched:
        print("\nTop unmatched gloss strings (top 25):")
        for g, c in unmatched.most_common(25):
            print(f"- {g}: {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
