from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _load_vocab(vocab_path: Path) -> list[str]:
    with vocab_path.open("r", encoding="utf-8") as f:
        tokens = json.load(f)
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
        raise ValueError("Invalid vocab JSON")
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity-check processed dataset samples.")
    parser.add_argument("--samples", default="data/processed/cislr", help="Root samples folder")
    parser.add_argument("--vocab", default="configs/vocab_v1_51.json", help="Vocab JSON")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    samples_root = (repo_root / args.samples).resolve()
    vocab_path = (repo_root / args.vocab).resolve()

    tokens = _load_vocab(vocab_path)
    if not samples_root.exists():
        print(f"No samples folder found: {samples_root}")
        return 1

    files = list(samples_root.rglob("*.npz"))
    if not files:
        print("No .npz samples found.")
        return 0

    label_counts: Counter[str] = Counter()
    bad_shapes = 0
    missing_pose_frames = 0
    missing_right_hand_frames = 0
    total_frames = 0
    durations = []
    src_fps_values = []
    target_fps_values = []
    person_counts: Counter[str] = Counter()

    for p in files:
        data = np.load(p, allow_pickle=False)
        X = data["X"].astype(np.float32)
        y = int(data["y"])
        label = str(data["label"])

        if "person_id" in data.files:
            person_counts[str(data["person_id"])] += 1

        if y < 0 or y >= len(tokens):
            print(f"Invalid y at {p}: {y}")
            continue
        if tokens[y] != label:
            print(f"Label mismatch at {p}: y maps to {tokens[y]} but file label is {label}")

        if X.ndim != 2:
            bad_shapes += 1
            continue

        T, F = X.shape
        label_counts[label] += 1

        if "duration_s" in data.files:
            durations.append(float(data["duration_s"]))
        if "src_fps" in data.files:
            src_fps_values.append(float(data["src_fps"]))
        if "target_fps" in data.files:
            target_fps_values.append(float(data["target_fps"]))

        # Using v1 feature positions from backend/app/landmarks.py
        # left_present at 0, right_present at 65, pose_present at 130
        if F >= 131:
            pose_present = X[:, 130]
            right_present = X[:, 65]
            missing_pose_frames += int(np.sum(pose_present < 0.5))
            missing_right_hand_frames += int(np.sum(right_present < 0.5))
            total_frames += int(T)

    print(f"Samples: {len(files)}")
    if bad_shapes:
        print(f"Bad shape files: {bad_shapes}")

    print("\nCounts per label:")
    for label, c in label_counts.most_common():
        print(f"- {label}: {c}")

    if total_frames > 0:
        print("\nMissing landmark rates (rough):")
        print(f"- pose missing: {missing_pose_frames/total_frames:.2%}")
        print(f"- right hand missing: {missing_right_hand_frames/total_frames:.2%}")

    if durations:
        arr = np.array(durations, dtype=np.float32)
        print("\nDuration stats (seconds):")
        print(f"- avg: {arr.mean():.3f}")
        print(f"- p50: {np.quantile(arr, 0.50):.3f}")
        print(f"- p90: {np.quantile(arr, 0.90):.3f}")

    if src_fps_values:
        arr = np.array(src_fps_values, dtype=np.float32)
        print("\nSource FPS stats:")
        print(f"- avg: {arr.mean():.2f}")
        print(f"- min/max: {arr.min():.2f}/{arr.max():.2f}")

    if target_fps_values:
        arr = np.array(target_fps_values, dtype=np.float32)
        print("\nTarget FPS stats:")
        print(f"- avg: {arr.mean():.2f}")

    if person_counts:
        print("\nPerson variability (top 10):")
        for pid, c in person_counts.most_common(10):
            print(f"- {pid}: {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
