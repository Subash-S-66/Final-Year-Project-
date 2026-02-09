from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np


def _load_vocab(vocab_path: Path) -> list[str]:
    with vocab_path.open("r", encoding="utf-8") as f:
        tokens = json.load(f)
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
        raise ValueError("Invalid vocab JSON: expected a non-empty list of strings")
    return tokens


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pad_or_truncate(sequence: np.ndarray, T: int) -> np.ndarray:
    # sequence: (N, F)
    N, F = sequence.shape
    out = np.zeros((T, F), dtype=np.float32)
    if N <= 0:
        return out
    if N >= T:
        return sequence[:T].astype(np.float32)
    out[:N] = sequence.astype(np.float32)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture landmark sequences and save as .npz samples.")
    parser.add_argument("--label", required=True, help="Label token, must exist in vocab JSON")
    parser.add_argument(
        "--person",
        default="p0",
        help="Person/subject id (used for train/val/test split by person). Example: p1, p2...",
    )
    parser.add_argument(
        "--session",
        default="s0",
        help="Session id for multi-session capture (e.g., S1, S2, S3). Stored in the sample .npz.",
    )
    parser.add_argument("--vocab", default="configs/vocab_v1_51.json", help="Path to vocab JSON")
    parser.add_argument("--out", default="data/processed/samples", help="Output root folder")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--fps", type=float, default=15.0, help="Capture FPS")
    parser.add_argument("--T", type=int, default=30, help="Window length (frames)")
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds to record per sample")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to record")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    vocab_path = (repo_root / args.vocab).resolve()
    out_root = (repo_root / args.out).resolve()

    tokens = _load_vocab(vocab_path)
    if args.label not in tokens:
        raise SystemExit(f"Label '{args.label}' not found in vocab: {vocab_path}")
    y = int(tokens.index(args.label))

    # Make backend/ importable as a root (same idea as `uvicorn ... --app-dir backend`).
    sys.path.insert(0, str(repo_root / "backend"))
    from app.landmarks import FEATURE_DIM, LandmarkExtractor  # noqa: E402

    extractor = LandmarkExtractor(enable_canonical_mirroring=True)

    # Prefer DirectShow on Windows (usually faster camera open).
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Failed to open camera")

    sample_dir = out_root / args.label
    _ensure_dir(sample_dir)

    frame_interval = 1.0 / max(args.fps, 1e-6)
    frames_per_sample = max(1, int(round(args.duration * args.fps)))
    print(f"Recording label={args.label} (y={y}) | samples={args.count} | frames/sample≈{frames_per_sample}")
    print("Instructions: press SPACE to start each sample, ESC to quit.")

    try:
        for i in range(args.count):
            # Preview until SPACE
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit("Camera read failed")

                overlay = frame.copy()
                cv2.putText(
                    overlay,
                    f"Label: {args.label} | sample {i+1}/{args.count} | SPACE=start | ESC=quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("ISL Capture", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    return 0
                if key == 32:
                    break

            # Record
            seq = np.zeros((frames_per_sample, FEATURE_DIM), dtype=np.float32)
            got = 0
            next_t = time.perf_counter()
            while got < frames_per_sample:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(min(next_t - now, 0.005))
                    continue
                next_t += frame_interval

                ok, frame = cap.read()
                if not ok:
                    break

                features, dbg = extractor.extract(frame)
                seq[got] = features
                got += 1

                overlay = frame.copy()
                cv2.putText(
                    overlay,
                    f"REC {got}/{frames_per_sample} | pose={dbg.pose_present} L={dbg.left_hand_present} R={dbg.right_hand_present} mir={dbg.canonical_mirrored}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("ISL Capture", overlay)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    return 0

            if got < 1:
                print("Warning: recorded 0 frames, skipping sample")
                continue

            X = _pad_or_truncate(seq[:got], args.T)
            sample_id = uuid.uuid4().hex
            out_path = sample_dir / f"{sample_id}.npz"
            np.savez_compressed(
                out_path,
                X=X,
                y=np.int64(y),
                label=np.array(args.label),
                person_id=np.array(args.person),
                session_id=np.array(args.session),
            )
            print(f"Saved {out_path.relative_to(repo_root)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
