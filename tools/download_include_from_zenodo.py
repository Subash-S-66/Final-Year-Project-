from __future__ import annotations

import argparse
from pathlib import Path

import requests


DEFAULT_CATEGORIES = [
    "Greetings",
    "Pronouns",
    "Days",
    "Home",
    "People",
    "Places",
    "Society",
    "Adjectives",
    "Jobs",
]

CATEGORY_ALIASES = {
    # Backward-compatible alias used in earlier commands/docs.
    "Days_and_Time": "Days",
}


def _download_with_resume(url: str, dest: Path, *, timeout_s: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    current = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if current > 0:
        headers["Range"] = f"bytes={current}-"

    with requests.get(url, stream=True, timeout=timeout_s, headers=headers) as r:
        if r.status_code == 416:
            return
        if r.status_code == 200 and current > 0:
            # Server ignored Range; restart file.
            current = 0
            with requests.get(url, stream=True, timeout=timeout_s) as r2:
                r2.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r2.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return

        r.raise_for_status()
        mode = "ab" if (current > 0 and r.status_code == 206) else "wb"
        with dest.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download selected INCLUDE zip parts from Zenodo.")
    parser.add_argument("--record", default="4010759", help="Zenodo record ID")
    parser.add_argument("--out", default="data/Download/include/zips", help="Output directory for zip files")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=DEFAULT_CATEGORIES,
        help="Category prefixes to download (e.g., Greetings Pronouns Home).",
    )
    parser.add_argument(
        "--extra-files",
        nargs="*",
        default=["README.md"],
        help="Additional files to download by exact filename.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    api_url = f"https://zenodo.org/api/records/{args.record}"
    resp = requests.get(api_url, timeout=60)
    resp.raise_for_status()
    rec = resp.json()
    files = rec.get("files", [])
    if not files:
        print("No files found in record.")
        return 1

    cats_in = set(args.categories or [])
    cats = {CATEGORY_ALIASES.get(c, c) for c in cats_in}
    extra = set(args.extra_files or [])

    wanted = []
    for f in files:
        key = str(f.get("key", ""))
        link = str((f.get("links") or {}).get("self", ""))
        if not key or not link:
            continue
        if key in extra:
            wanted.append((key, link, int(f.get("size") or 0)))
            continue
        if key.endswith(".zip"):
            prefix = key.split("_", 1)[0]
            if prefix in cats:
                wanted.append((key, link, int(f.get("size") or 0)))

    wanted.sort(key=lambda x: x[0])
    if not wanted:
        print("No matching files for selected categories.")
        return 1

    total_bytes = sum(s for _, _, s in wanted)
    print(f"Will download {len(wanted)} files (~{total_bytes / (1024**3):.2f} GB)")
    for key, _link, size in wanted:
        print(f"- {key} ({size / (1024**2):.1f} MB)")

    ok = 0
    fail = 0
    for key, link, _size in wanted:
        dest = out_dir / key
        try:
            _download_with_resume(link, dest)
            ok += 1
            print(f"OK: {key}")
        except Exception as e:
            fail += 1
            print(f"FAIL: {key} -> {str(e)[:200]}")

    print(f"\nDone. ok={ok} fail={fail} out={out_dir}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
