from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from tqdm import tqdm

from tools.cislr_common import load_vocab, repo_root


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _select_frames_at_fps(video_path: Path, target_fps: float) -> tuple[list[np.ndarray], float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if src_fps <= 1e-3:
        src_fps = target_fps

    duration_s = frame_count / src_fps if frame_count > 0 else 0.0
    step = max(1, int(round(src_fps / target_fps)))

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames, src_fps, duration_s


def _pad_or_truncate(sequence: np.ndarray, T: int) -> np.ndarray:
    N, F = sequence.shape
    out = np.zeros((T, F), dtype=np.float32)
    if N <= 0:
        return out
    if N >= T:
        return sequence[:T].astype(np.float32)
    out[:N] = sequence.astype(np.float32)
    return out


def _process_one(
    video_path: str,
    token: str,
    y: int,
    uid: str,
    dataset_name: str,
    out_root: str,
    T: int,
    target_fps: float,
    cache: bool,
    delete_video: bool,
) -> tuple[bool, str]:
    video_p = Path(video_path)
    out_root_p = Path(out_root)

    root = repo_root()
    sys.path.insert(0, str(root / "backend"))
    from app.landmarks import FEATURE_DIM, LandmarkExtractor  # noqa: E402

    stat = video_p.stat()
    cache_key = _sha1(f"{video_p}|{stat.st_mtime_ns}|{stat.st_size}|T={T}|fps={target_fps}")
    out_dir = out_root_p / token
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uid}_{cache_key}.npz"

    if cache and out_path.exists():
        if delete_video:
            try:
                video_p.unlink(missing_ok=True)
            except Exception:
                pass
        return True, f"cached:{out_path.name}"

    frames, src_fps, duration_s = _select_frames_at_fps(video_p, target_fps=target_fps)

    extractor = LandmarkExtractor(enable_canonical_mirroring=True)
    try:
        seq = np.zeros((len(frames), FEATURE_DIM), dtype=np.float32)
        for i, frame in enumerate(frames):
            feat, _dbg = extractor.extract(frame)
            seq[i] = feat
    finally:
        extractor.close()

    X = _pad_or_truncate(seq, T)

    np.savez_compressed(
        out_path,
        X=X,
        y=np.int64(y),
        label=np.array(token),
        person_id=np.array(dataset_name),
        dataset=np.array(dataset_name),
        source_video=np.array(str(video_p).replace("\\", "/")),
        src_fps=np.float32(src_fps),
        target_fps=np.float32(target_fps),
        duration_s=np.float32(duration_s),
        frames_extracted=np.int64(len(frames)),
        T=np.int64(T),
    )
    if delete_video:
        try:
            video_p.unlink(missing_ok=True)
        except Exception:
            pass
    return True, f"ok:{out_path.name}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from bridgeconn videos into standardized .npz samples."
    )
    parser.add_argument("--videos", default="data/raw/bridgeconn/videos")
    parser.add_argument("--out", default="data/processed/bridgeconn")
    parser.add_argument("--vocab", default="configs/vocab_v1_51.json")
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--delete-videos-after-extract", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    videos_root = (root / args.videos).resolve()
    out_root = (root / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab((root / args.vocab).resolve())
    vocab_to_id = {t: i for i, t in enumerate(vocab)}

    if not videos_root.exists():
        print(f"Missing videos folder: {videos_root}")
        return 1

    jobs: list[tuple[str, str, int, str]] = []  # (video_path, token, y, uid)
    for label_dir in videos_root.iterdir():
        if not label_dir.is_dir():
            continue
        token_name = label_dir.name
        if token_name not in vocab_to_id:
            continue

        clips = sorted(label_dir.glob("*.mp4"))
        for vp in clips:
            y = int(vocab_to_id[token_name])
            uid = vp.stem
            jobs.append((str(vp), token_name, y, uid))

    if not jobs:
        print("No videos found to process.")
        return 1

    cache = not args.no_cache
    ok = 0
    fail = 0

    if args.workers <= 1:
        for video_path, token_name, y, uid in tqdm(jobs, desc="extract", unit="clip"):
            try:
                _process_one(
                    video_path=video_path,
                    token=token_name,
                    y=y,
                    uid=uid,
                    dataset_name="bridgeconn",
                    out_root=str(out_root),
                    T=args.T,
                    target_fps=args.fps,
                    cache=cache,
                    delete_video=bool(args.delete_videos_after_extract),
                )
                ok += 1
            except Exception as e:
                fail += 1
                print(f"FAIL {video_path}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = []
            for video_path, token_name, y, uid in jobs:
                futs.append(
                    ex.submit(
                        _process_one,
                        video_path,
                        token_name,
                        y,
                        uid,
                        "bridgeconn",
                        str(out_root),
                        args.T,
                        args.fps,
                        cache,
                        bool(args.delete_videos_after_extract),
                    )
                )
            for fut in tqdm(as_completed(futs), total=len(futs), desc="extract", unit="clip"):
                try:
                    fut.result()
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"FAIL: {e}")

    print(f"\nExtraction done: ok={ok} fail={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
