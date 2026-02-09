from __future__ import annotations

import argparse
from collections import OrderedDict


def main() -> int:
    parser = argparse.ArgumentParser(description="Search Hugging Face Hub for candidate datasets.")
    parser.add_argument(
        "--query",
        action="append",
        default=["indian sign language", "ISL", "sign language"],
        help="Search query (repeatable).",
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()

    # De-dup while preserving order
    hits: OrderedDict[str, dict] = OrderedDict()

    for q in args.query:
        for ds in api.list_datasets(search=q, limit=args.limit):
            ds_id = getattr(ds, "id", None) or getattr(ds, "datasetId", None)
            if not ds_id:
                continue
            if ds_id in hits:
                continue
            hits[ds_id] = {
                "id": ds_id,
                "sha": getattr(ds, "sha", None),
                "lastModified": getattr(ds, "lastModified", None),
            }

    print("=== HuggingFace dataset search results ===")
    print(f"queries: {args.query}")
    print(f"unique results: {len(hits)}")
    for i, ds_id in enumerate(hits.keys(), start=1):
        print(f"{i:02d}. {ds_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
