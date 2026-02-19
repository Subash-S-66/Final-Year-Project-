from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


@dataclass
class ClipQuality:
    path: str
    label: str
    width: int
    height: int
    fps: float
    frames: int
    duration_s: float
    blur: float
    contrast: float
    brightness: float
    motion: float
    score: float
    passed: bool
    fail_reason: str
    sha1: str


def _sample_indices(total_frames: int, sample_count: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=np.int64)
    n = min(total_frames, sample_count)
    return np.linspace(0, total_frames - 1, n, dtype=np.int64)


def _analyze_video(
    path: Path,
    *,
    sample_frames: int,
    min_width: int,
    min_height: int,
    min_frames: int,
    min_duration_s: float,
    max_duration_s: float,
    min_blur: float,
    min_contrast: float,
    min_motion: float,
) -> ClipQuality:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return ClipQuality(
            path=str(path),
            label=path.parent.name,
            width=0,
            height=0,
            fps=0.0,
            frames=0,
            duration_s=0.0,
            blur=0.0,
            contrast=0.0,
            brightness=0.0,
            motion=0.0,
            score=0.0,
            passed=False,
            fail_reason="open_failed",
            sha1="",
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1e-3:
        fps = 15.0
    duration_s = frames / fps if frames > 0 else 0.0

    blur_vals: list[float] = []
    contrast_vals: list[float] = []
    bright_vals: list[float] = []
    motion_vals: list[float] = []
    prev = None

    for idx in _sample_indices(frames, sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        contrast_vals.append(float(gray.std()))
        bright_vals.append(float(gray.mean()))

        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            motion_vals.append(float(np.mean(diff)))
        prev = gray

    cap.release()

    blur = float(np.median(blur_vals)) if blur_vals else 0.0
    contrast = float(np.median(contrast_vals)) if contrast_vals else 0.0
    brightness = float(np.median(bright_vals)) if bright_vals else 0.0
    motion = float(np.median(motion_vals)) if motion_vals else 0.0

    # Hard quality gates (strict).
    fail = []
    if width < min_width or height < min_height:
        fail.append("low_resolution")
    if frames < min_frames:
        fail.append("too_few_frames")
    if duration_s < min_duration_s:
        fail.append("too_short")
    if duration_s > max_duration_s:
        fail.append("too_long")
    if blur < min_blur:
        fail.append("too_blurry")
    if contrast < min_contrast:
        fail.append("low_contrast")
    if motion < min_motion:
        fail.append("low_motion")
    if brightness < 20.0 or brightness > 235.0:
        fail.append("bad_lighting")

    sharp_norm = _clamp01((blur - min_blur) / 250.0)
    contrast_norm = _clamp01((contrast - min_contrast) / 50.0)
    motion_norm = _clamp01((motion - min_motion) / 18.0)
    res_norm = _clamp01(min(width, height) / 480.0)
    duration_norm = _clamp01(1.0 - abs(duration_s - 2.0) / 2.5)
    light_norm = _clamp01(1.0 - abs(brightness - 128.0) / 128.0)

    score = (
        0.30 * sharp_norm
        + 0.18 * contrast_norm
        + 0.18 * motion_norm
        + 0.14 * res_norm
        + 0.12 * duration_norm
        + 0.08 * light_norm
    )
    passed = len(fail) == 0

    return ClipQuality(
        path=str(path),
        label=path.parent.name,
        width=width,
        height=height,
        fps=fps,
        frames=frames,
        duration_s=duration_s,
        blur=blur,
        contrast=contrast,
        brightness=brightness,
        motion=motion,
        score=score,
        passed=passed,
        fail_reason=",".join(fail),
        sha1=_sha1(path),
    )


def _copy_or_move(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
        return
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep only high-quality CISLR videos (strict filtering + per-label top-k)."
    )
    parser.add_argument("--src", default="data/raw/cislr/videos_unfiltered", help="Input root: <LABEL>/*.mp4")
    parser.add_argument("--dst", default="data/raw/cislr/videos", help="Curated output root")
    parser.add_argument("--report-dir", default="reports/cislr_quality")
    parser.add_argument("--sample-frames", type=int, default=24)
    parser.add_argument("--min-width", type=int, default=320)
    parser.add_argument("--min-height", type=int, default=240)
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--min-duration-s", type=float, default=0.7)
    parser.add_argument("--max-duration-s", type=float, default=7.0)
    parser.add_argument("--min-blur", type=float, default=35.0)
    parser.add_argument("--min-contrast", type=float, default=20.0)
    parser.add_argument("--min-motion", type=float, default=1.8)
    parser.add_argument("--keep-per-label", type=int, default=20)
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")
    parser.add_argument("--clear-dst", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_root = (repo_root / args.src).resolve()
    dst_root = (repo_root / args.dst).resolve()
    report_dir = (repo_root / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        print(f"Missing src: {src_root}")
        return 1

    if args.clear_dst and dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    videos = sorted([p for p in src_root.rglob("*.mp4") if p.is_file()])
    if not videos:
        print(f"No .mp4 files in {src_root}")
        return 1

    assessed: list[ClipQuality] = []
    for p in tqdm(videos, desc="quality-scan", unit="clip"):
        assessed.append(
            _analyze_video(
                p,
                sample_frames=args.sample_frames,
                min_width=args.min_width,
                min_height=args.min_height,
                min_frames=args.min_frames,
                min_duration_s=args.min_duration_s,
                max_duration_s=args.max_duration_s,
                min_blur=args.min_blur,
                min_contrast=args.min_contrast,
                min_motion=args.min_motion,
            )
        )

    # Deduplicate exact content by sha1: keep highest score per hash.
    best_by_hash: dict[str, ClipQuality] = {}
    for q in assessed:
        if not q.sha1:
            continue
        keep = best_by_hash.get(q.sha1)
        if keep is None or q.score > keep.score:
            best_by_hash[q.sha1] = q
    dup_hashes = {q.sha1 for q in assessed if q.sha1 and q.sha1 in best_by_hash and best_by_hash[q.sha1].path != q.path}

    by_label: dict[str, list[ClipQuality]] = {}
    rejected = 0
    for q in assessed:
        if q.sha1 in dup_hashes and best_by_hash[q.sha1].path != q.path:
            rejected += 1
            continue
        if not q.passed:
            rejected += 1
            continue
        by_label.setdefault(q.label, []).append(q)

    kept = 0
    for label, items in by_label.items():
        items.sort(key=lambda x: x.score, reverse=True)
        selected = items if args.keep_per_label <= 0 else items[: args.keep_per_label]
        for q in selected:
            src = Path(q.path)
            dst = dst_root / label / src.name
            _copy_or_move(src, dst, args.mode)
            kept += 1
        rejected += max(0, len(items) - len(selected))

    # Reports
    csv_path = report_dir / "quality_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "width",
                "height",
                "fps",
                "frames",
                "duration_s",
                "blur",
                "contrast",
                "brightness",
                "motion",
                "score",
                "passed",
                "fail_reason",
                "sha1",
            ],
        )
        w.writeheader()
        for q in assessed:
            w.writerow(asdict(q))

    label_stats = {
        label: {
            "kept": len(items if args.keep_per_label <= 0 else items[: args.keep_per_label]),
            "mean_score": float(np.mean([x.score for x in items])) if items else 0.0,
        }
        for label, items in sorted(by_label.items())
    }
    stats = {
        "src": str(src_root),
        "dst": str(dst_root),
        "total_scanned": len(assessed),
        "kept": kept,
        "rejected": rejected,
        "labels_with_kept_samples": len(label_stats),
        "duplicate_hash_groups": len(dup_hashes),
        "thresholds": {
            "min_width": args.min_width,
            "min_height": args.min_height,
            "min_frames": args.min_frames,
            "min_duration_s": args.min_duration_s,
            "max_duration_s": args.max_duration_s,
            "min_blur": args.min_blur,
            "min_contrast": args.min_contrast,
            "min_motion": args.min_motion,
            "keep_per_label": args.keep_per_label,
        },
        "label_stats": label_stats,
    }
    stats_path = report_dir / "quality_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\nCuration summary")
    print(f"- scanned:  {len(assessed)}")
    print(f"- kept:     {kept}")
    print(f"- rejected: {rejected}")
    print(f"- labels:   {len(label_stats)}")
    print(f"- report:   {csv_path}")
    print(f"- stats:    {stats_path}")
    print(f"- output:   {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
