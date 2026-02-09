from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load_vocab(vocab_path: Path) -> list[str]:
    with vocab_path.open("r", encoding="utf-8") as f:
        tokens = json.load(f)
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("Invalid vocab JSON")
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate train/val/test split manifests (prefer split by person_id).")
    parser.add_argument("--samples", default="data/processed/samples", help="Samples root")
    parser.add_argument("--splits", default="data/processed/splits", help="Output splits folder")
    parser.add_argument("--vocab", default="configs/vocab_v1_51.json", help="Vocab JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_people", type=int, default=1)
    parser.add_argument("--test_people", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    samples_root = (repo_root / args.samples).resolve()
    splits_root = (repo_root / args.splits).resolve()
    vocab_path = (repo_root / args.vocab).resolve()

    _ = _load_vocab(vocab_path)
    splits_root.mkdir(parents=True, exist_ok=True)

    # Collect all sample files
    files = list(samples_root.rglob("*.npz"))
    if not files:
        print("No samples found.")
        return 1

    # Read person_id from npz if present
    person_to_files: dict[str, list[Path]] = {}
    unknown: list[Path] = []

    import numpy as np

    for p in files:
        data = np.load(p, allow_pickle=False)
        if "person_id" in data.files:
            pid = str(data["person_id"])
            person_to_files.setdefault(pid, []).append(p)
        else:
            unknown.append(p)

    people = sorted(person_to_files.keys())
    random.Random(args.seed).shuffle(people)

    val_people = max(0, min(args.val_people, len(people)))
    test_people = max(0, min(args.test_people, len(people) - val_people))

    val_set = set(people[:val_people])
    test_set = set(people[val_people : val_people + test_people])
    train_set = set(people[val_people + test_people :])

    train_files = [p for pid in train_set for p in person_to_files[pid]]
    val_files = [p for pid in val_set for p in person_to_files[pid]]
    test_files = [p for pid in test_set for p in person_to_files[pid]]

    # If we have legacy files without person_id, put them in train by default.
    # (We warn because this weakens person-generalization evaluation.)
    if unknown:
        print(f"Warning: {len(unknown)} samples are missing person_id. Assigning them to TRAIN.")
        train_files.extend(unknown)

    def write_list(name: str, plist: list[Path]) -> None:
        out = splits_root / f"{name}.txt"
        rel_lines = [str(p.relative_to(repo_root)).replace('\\', '/') for p in sorted(plist)]
        out.write_text("\n".join(rel_lines) + "\n", encoding="utf-8")

    write_list("train", train_files)
    write_list("val", val_files)
    write_list("test", test_files)

    print("Wrote splits:")
    print(f"- train: {len(train_files)}")
    print(f"- val:   {len(val_files)}")
    print(f"- test:  {len(test_files)}")
    print(f"People: train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
