from __future__ import annotations

import json
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.landmarks import FEATURE_DIM, LandmarkExtractor
from app.onnx_infer import OnnxLstmRunner
from app.protocol import FrameMeta, PredictionMessage
from app.settings import settings


app = FastAPI(title=settings.api_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MediaPipe initialization is expensive; keep one extractor per process.
extractor = LandmarkExtractor(enable_canonical_mirroring=True)
onnx_init_error: str = ""


def _try_create_onnx_runner() -> OnnxLstmRunner | None:
    global onnx_init_error
    if not settings.use_onnx:
        onnx_init_error = "ONNX disabled by settings"
        return None
    try:
        onnx_init_error = ""
        return OnnxLstmRunner(
            onnx_path=settings.onnx_model_path,
            vocab_path=settings.vocab_path,
            meta_path=settings.onnx_meta_path,
        )
    except Exception as e:
        # Keep API running even if model isn't present, but preserve error for diagnostics.
        onnx_init_error = str(e)
        return None


onnx_runner = _try_create_onnx_runner()


LEFT_PRESENT_IDX = 0
RIGHT_PRESENT_IDX = 65
LEFT_COORD_SLICE = slice(2, 65)
RIGHT_COORD_SLICE = slice(67, 130)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "onnx_enabled": settings.use_onnx,
        "onnx_loaded": onnx_runner is not None,
        "onnx_model_path": settings.onnx_model_path,
        "onnx_meta_path": settings.onnx_meta_path,
        "vocab_path": settings.vocab_path,
        "onnx_init_error": onnx_init_error,
        "acceptance_global_conf": (float(onnx_runner.global_conf_threshold) if onnx_runner is not None else 0.0),
        "acceptance_global_margin": (float(onnx_runner.global_margin_threshold) if onnx_runner is not None else 0.0),
    }


@dataclass
class PendingFrame:
    meta: FrameMeta


class SlidingWindow:
    """Maintains a sliding window of recent landmark features."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = int(window_size)
        self.buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

    def add_frame(self, features: np.ndarray) -> None:
        if features.shape != (FEATURE_DIM,):
            raise ValueError(f"Expected feature vector shape ({FEATURE_DIM},), got {features.shape}")
        self.buffer.append(features.astype(np.float32, copy=False))

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.window_size

    def get_window(self) -> np.ndarray:
        if not self.is_ready():
            raise ValueError("Window not ready")
        return np.stack(list(self.buffer), axis=0)


@dataclass
class StableDecision:
    token: str
    confidence: float
    is_committed: bool
    live_token: str
    live_confidence: float
    vote_ratio: float


class PredictionStabilizer:
    """Aggregates window predictions and emits debounced, hysteresis-controlled tokens."""

    def __init__(
        self,
        *,
        vocab: list[str],
        no_sign_token: str,
        vote_window: int,
        min_conf_enter: float,
        min_conf_exit: float,
        min_vote_ratio: float,
        min_margin: float,
        debounce_frames: int,
        release_frames: int,
    ) -> None:
        self.vocab = vocab
        self.no_sign_token = no_sign_token
        self.no_sign_idx = self.vocab.index(no_sign_token) if no_sign_token in self.vocab else -1

        self.vote_window = max(1, int(vote_window))
        self.min_conf_enter = float(min_conf_enter)
        self.min_conf_exit = float(min_conf_exit)
        self.min_vote_ratio = float(min_vote_ratio)
        self.min_margin = float(max(0.0, min_margin))
        self.debounce_frames = max(1, int(debounce_frames))
        self.release_frames = max(1, int(release_frames))

        self.history: deque[np.ndarray] = deque(maxlen=self.vote_window)
        self.emitted_idx = self.no_sign_idx
        self.candidate_idx = -1
        self.candidate_count = 0
        self.release_count = 0

    def reset_to_no_sign(self) -> None:
        self.history.clear()
        self.emitted_idx = self.no_sign_idx
        self.release_count = 0
        self._clear_candidate()

    def _token(self, idx: int) -> str:
        if 0 <= idx < len(self.vocab):
            return self.vocab[idx]
        return self.no_sign_token

    def _update_candidate(self, idx: int) -> None:
        if idx == self.candidate_idx:
            self.candidate_count += 1
        else:
            self.candidate_idx = idx
            self.candidate_count = 1

    def _clear_candidate(self) -> None:
        self.candidate_idx = -1
        self.candidate_count = 0

    def update(self, probs: np.ndarray) -> StableDecision:
        if probs.ndim != 1:
            raise ValueError(f"Expected probs shape (C,), got {probs.shape}")
        self.history.append(probs.astype(np.float32, copy=False))

        hist = np.stack(list(self.history), axis=0)
        mean_probs = hist.mean(axis=0)
        live_idx = int(np.argmax(mean_probs))
        live_conf = float(mean_probs[live_idx])
        if mean_probs.shape[0] > 1:
            top2 = np.partition(mean_probs, -2)[-2]
        else:
            top2 = 0.0
        margin = float(max(0.0, live_conf - top2))

        votes = np.argmax(hist, axis=1)
        vote_ratio = float(np.mean(votes == live_idx))

        live_is_sign = live_idx != self.no_sign_idx
        strong_candidate = (
            live_is_sign
            and live_conf >= self.min_conf_enter
            and vote_ratio >= self.min_vote_ratio
            and margin >= self.min_margin
        )

        if self.emitted_idx == self.no_sign_idx:
            if strong_candidate:
                self._update_candidate(live_idx)
                if self.candidate_count >= self.debounce_frames:
                    self.emitted_idx = live_idx
                    self.release_count = 0
                    self._clear_candidate()
            else:
                self._clear_candidate()
        else:
            if live_idx == self.emitted_idx and live_conf >= self.min_conf_exit:
                self.release_count = 0
                self._clear_candidate()
            else:
                self.release_count += 1
                if strong_candidate:
                    self._update_candidate(live_idx)
                    if self.candidate_count >= self.debounce_frames:
                        self.emitted_idx = live_idx
                        self.release_count = 0
                        self._clear_candidate()
                if self.release_count >= self.release_frames:
                    self.emitted_idx = self.no_sign_idx
                    self.release_count = 0
                    self._clear_candidate()

        emitted_is_sign = self.emitted_idx != self.no_sign_idx
        if emitted_is_sign:
            token = self._token(self.emitted_idx)
            confidence = float(mean_probs[self.emitted_idx])
            is_committed = True
        else:
            token = self.no_sign_token
            confidence = float(max(0.5, 1.0 - live_conf))
            is_committed = False

        return StableDecision(
            token=token,
            confidence=confidence,
            is_committed=is_committed,
            live_token=self._token(live_idx),
            live_confidence=live_conf,
            vote_ratio=vote_ratio,
        )


def _decode_image(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def _sample_every_n_frames(*, src_fps: float, sample_fps: float) -> int:
    if sample_fps <= 0:
        return 1
    if src_fps <= 1e-3:
        src_fps = sample_fps
    return max(1, int(round(src_fps / sample_fps)))


def _window_hand_presence_ratio(raw_window: np.ndarray) -> float:
    if raw_window.ndim != 2 or raw_window.shape[0] <= 0:
        return 0.0
    if raw_window.shape[1] <= RIGHT_PRESENT_IDX:
        return 0.0
    has_hands = (raw_window[:, LEFT_PRESENT_IDX] >= 0.5) | (raw_window[:, RIGHT_PRESENT_IDX] >= 0.5)
    return float(np.mean(has_hands.astype(np.float32)))


def _window_hand_motion_score(raw_window: np.ndarray) -> float:
    if raw_window.ndim != 2 or raw_window.shape[0] <= 1:
        return 0.0
    if raw_window.shape[1] < RIGHT_COORD_SLICE.stop:
        return 0.0

    left = raw_window[:, LEFT_COORD_SLICE]
    right = raw_window[:, RIGHT_COORD_SLICE]
    d_left = np.abs(np.diff(left, axis=0))
    d_right = np.abs(np.diff(right, axis=0))

    left_mask = (raw_window[1:, LEFT_PRESENT_IDX] >= 0.5) & (raw_window[:-1, LEFT_PRESENT_IDX] >= 0.5)
    right_mask = (raw_window[1:, RIGHT_PRESENT_IDX] >= 0.5) & (raw_window[:-1, RIGHT_PRESENT_IDX] >= 0.5)

    vals = []
    if left_mask.any():
        vals.append(float(d_left[left_mask].mean()))
    if right_mask.any():
        vals.append(float(d_right[right_mask].mean()))
    return float(np.mean(vals)) if vals else 0.0


def _window_activity_stats(raw_window: np.ndarray, *, has_no_sign_vocab: bool) -> tuple[float, float, bool]:
    hand_ratio = _window_hand_presence_ratio(raw_window)
    motion_score = _window_hand_motion_score(raw_window)
    motion_thr = settings.min_motion_score if has_no_sign_vocab else max(settings.min_motion_score, settings.min_motion_score_no_no_sign)
    hand_ok = hand_ratio >= settings.min_hand_presence_ratio
    # Held signs can have low motion but sustained hand presence.
    held_sign_ok = hand_ratio >= (0.80 if has_no_sign_vocab else 0.70)
    has_activity = bool(hand_ok and (motion_score >= motion_thr or held_sign_ok))
    return hand_ratio, motion_score, has_activity


def _init_state() -> tuple[SlidingWindow, PredictionStabilizer]:
    vocab = onnx_runner.vocab if onnx_runner is not None else [settings.no_sign_token]
    has_no_sign = settings.no_sign_token in vocab
    window = SlidingWindow(window_size=settings.window_size)
    stabilizer = PredictionStabilizer(
        vocab=vocab,
        no_sign_token=settings.no_sign_token,
        vote_window=settings.vote_window,
        min_conf_enter=(
            settings.min_confidence_enter
            if has_no_sign
            else max(settings.min_confidence_enter, settings.min_confidence_enter_no_no_sign)
        ),
        min_conf_exit=(
            settings.min_confidence_exit
            if has_no_sign
            else max(settings.min_confidence_exit, settings.min_confidence_exit_no_no_sign)
        ),
        min_vote_ratio=(
            settings.min_vote_ratio
            if has_no_sign
            else max(settings.min_vote_ratio, settings.min_vote_ratio_no_no_sign)
        ),
        min_margin=(0.0 if has_no_sign else settings.min_margin_no_no_sign),
        debounce_frames=settings.debounce_frames,
        release_frames=settings.release_frames,
    )
    return window, stabilizer


def _zero_pad_or_truncate(sequence: np.ndarray, target_t: int) -> np.ndarray:
    if sequence.ndim != 2:
        raise ValueError(f"Expected (N,F) sequence, got shape={sequence.shape}")
    n, f = sequence.shape
    out = np.zeros((target_t, f), dtype=np.float32)
    if n <= 0:
        return out
    if n >= target_t:
        return sequence[:target_t].astype(np.float32)
    out[:n] = sequence.astype(np.float32)
    return out


def _infer_one_frame(
    payload: bytes,
    *,
    window: SlidingWindow,
    stabilizer: PredictionStabilizer,
) -> tuple[str, float, bool, dict]:
    img = _decode_image(payload)
    return _infer_bgr_frame(img, window=window, stabilizer=stabilizer)


def _infer_bgr_frame(
    img: np.ndarray,
    *,
    window: SlidingWindow,
    stabilizer: PredictionStabilizer,
    landmark_extractor: LandmarkExtractor | None = None,
) -> tuple[str, float, bool, dict]:
    active_extractor = landmark_extractor if landmark_extractor is not None else extractor
    features, ex_debug = active_extractor.extract(img)
    window.add_frame(features)

    token = settings.no_sign_token
    confidence = 1.0
    is_committed = False
    live_token = settings.no_sign_token
    live_confidence = 0.0
    vote_ratio = 0.0
    live_margin = 0.0
    accepted_by_thresholds = False
    hand_presence_ratio = 0.0
    motion_score = 0.0
    has_activity = False
    conf_thr = 0.0
    margin_thr = 0.0
    relaxed_conf_thr = 0.0
    relaxed_margin_thr = 0.0
    live_gate_ok = False
    gating_mode = "strict"
    has_hands = bool(ex_debug.left_hand_present or ex_debug.right_hand_present)

    if window.is_ready() and onnx_runner is not None:
        raw_window = window.get_window()
        has_no_sign = settings.no_sign_token in onnx_runner.vocab
        hand_presence_ratio, motion_score, has_activity = _window_activity_stats(raw_window, has_no_sign_vocab=has_no_sign)

        if settings.require_hand_for_sign and not has_hands:
            stabilizer.reset_to_no_sign()
        else:
            pred = onnx_runner.predict(raw_window)
            live_margin = float(pred.margin)
            conf_thr, margin_thr = onnx_runner.acceptance_thresholds_for(pred.y)
            accepted_by_thresholds = onnx_runner.accepts(pred.y, pred.confidence, pred.margin)
            strong_static = bool(pred.confidence >= max(0.85, conf_thr) and pred.margin >= max(0.30, margin_thr))
            activity_ok = bool(has_activity or strong_static)
            if not has_no_sign:
                # For vocab-only models (no explicit NO_SIGN class), class-wise calibrated
                # thresholds can be too strict for noisy webcam frames. Use a relaxed live gate
                # plus activity checks, then let stabilizer handle debounce/hysteresis.
                gating_mode = "relaxed_no_no_sign"
                relaxed_conf_thr = max(0.20, settings.min_confidence_enter_no_no_sign * 0.55)
                relaxed_margin_thr = max(0.04, settings.min_margin_no_no_sign * 0.50)
                live_gate_ok = bool(pred.confidence >= relaxed_conf_thr and pred.margin >= relaxed_margin_thr)
                if not activity_ok or not live_gate_ok:
                    stabilizer.reset_to_no_sign()
                else:
                    decision = stabilizer.update(pred.probs)
                    token = decision.token
                    confidence = decision.confidence
                    is_committed = decision.is_committed
                    live_token = decision.live_token
                    live_confidence = decision.live_confidence
                    vote_ratio = decision.vote_ratio
            else:
                if not accepted_by_thresholds and not activity_ok:
                    stabilizer.reset_to_no_sign()
                else:
                    decision = stabilizer.update(pred.probs)
                    token = decision.token
                    confidence = decision.confidence
                    is_committed = decision.is_committed
                    live_token = decision.live_token
                    live_confidence = decision.live_confidence
                    vote_ratio = decision.vote_ratio

    debug = {
        "pose_detected": ex_debug.pose_present,
        "left_hand_detected": ex_debug.left_hand_present,
        "right_hand_detected": ex_debug.right_hand_present,
        "has_hands": has_hands,
        "canonical_mirrored": ex_debug.canonical_mirrored,
        "feature_dim": int(features.shape[0]),
        "window_ready": window.is_ready(),
        "window_fill": len(window.buffer),
        "onnx_enabled": settings.use_onnx,
        "onnx_loaded": onnx_runner is not None,
        "live_token": live_token,
        "live_confidence": float(live_confidence),
        "live_margin": float(live_margin),
        "vote_ratio": float(vote_ratio),
        "accepted_by_thresholds": bool(accepted_by_thresholds),
        "accept_conf_threshold": float(conf_thr),
        "accept_margin_threshold": float(margin_thr),
        "relaxed_conf_threshold": float(relaxed_conf_thr),
        "relaxed_margin_threshold": float(relaxed_margin_thr),
        "live_gate_ok": bool(live_gate_ok),
        "gating_mode": gating_mode,
        "window_hand_presence_ratio": float(hand_presence_ratio),
        "window_motion_score": float(motion_score),
        "window_has_activity": bool(has_activity),
        "stabilizer_history": len(stabilizer.history),
    }
    return token, float(confidence), bool(is_committed), debug


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    sample_fps: float = 15.0,
    max_frames: int = 600,
):
    """Offline test endpoint: upload one video file and get stabilized prediction summary."""
    if onnx_runner is None:
        raise HTTPException(status_code=503, detail=f"ONNX model is not loaded: {onnx_init_error}")
    if sample_fps <= 0:
        raise HTTPException(status_code=400, detail="sample_fps must be > 0")
    if max_frames <= 0:
        raise HTTPException(status_code=400, detail="max_frames must be > 0")

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    tmp_path = ""
    local_extractor: LandmarkExtractor | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open uploaded video")

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = _sample_every_n_frames(src_fps=src_fps, sample_fps=sample_fps)

        window, stabilizer = _init_state()
        local_extractor = LandmarkExtractor(enable_canonical_mirroring=True)
        sampled = 0
        inferred = 0
        idx = 0
        committed = 0
        any_hands_detected = False
        by_live: dict[str, int] = {}
        by_committed: dict[str, int] = {}
        by_committed_conf_sum: dict[str, float] = {}
        by_window_pred: dict[str, int] = {}
        by_window_conf_sum: dict[str, float] = {}
        by_window_relaxed: dict[str, int] = {}
        by_window_relaxed_conf_sum: dict[str, float] = {}
        by_window_relaxed_margin_sum: dict[str, float] = {}
        timeline: list[dict[str, float | int | str | bool]] = []
        no_sign = settings.no_sign_token
        has_no_sign_vocab = settings.no_sign_token in onnx_runner.vocab

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue

            sampled += 1
            token, conf, is_committed, debug = _infer_bgr_frame(
                frame,
                window=window,
                stabilizer=stabilizer,
                landmark_extractor=local_extractor,
            )
            any_hands_detected = any_hands_detected or bool(debug.get("has_hands", False))
            live_token = str(debug.get("live_token", no_sign))
            live_conf = float(debug.get("live_confidence", 0.0))
            vote_ratio = float(debug.get("vote_ratio", 0.0))
            live_margin = float(debug.get("live_margin", 0.0))
            accepted = bool(debug.get("accepted_by_thresholds", False))
            if debug.get("window_ready"):
                inferred += 1
                by_live[live_token] = by_live.get(live_token, 0) + 1
                if is_committed and token != no_sign:
                    committed += 1
                    by_committed[token] = by_committed.get(token, 0) + 1
                    by_committed_conf_sum[token] = by_committed_conf_sum.get(token, 0.0) + float(conf)

                # Upload path should not depend on real-time debounce; aggregate raw window predictions.
                raw_window = window.get_window()
                hand_ratio, motion_score, good_activity = _window_activity_stats(raw_window, has_no_sign_vocab=has_no_sign_vocab)
                if onnx_runner is not None:
                    pred_raw = onnx_runner.predict(raw_window)
                    accepted_raw = onnx_runner.accepts(pred_raw.y, pred_raw.confidence, pred_raw.margin)
                    strong_static = bool(pred_raw.confidence >= 0.85 and pred_raw.margin >= 0.30)
                    activity_ok = bool(good_activity or strong_static)
                    if settings.require_hand_for_sign and hand_ratio < settings.min_hand_presence_ratio:
                        activity_ok = False
                    if activity_ok:
                        by_window_relaxed[pred_raw.token] = by_window_relaxed.get(pred_raw.token, 0) + 1
                        by_window_relaxed_conf_sum[pred_raw.token] = (
                            by_window_relaxed_conf_sum.get(pred_raw.token, 0.0) + float(pred_raw.confidence)
                        )
                        by_window_relaxed_margin_sum[pred_raw.token] = (
                            by_window_relaxed_margin_sum.get(pred_raw.token, 0.0) + float(pred_raw.margin)
                        )
                    if accepted_raw and activity_ok:
                        if activity_ok:
                            by_window_pred[pred_raw.token] = by_window_pred.get(pred_raw.token, 0) + 1
                            by_window_conf_sum[pred_raw.token] = by_window_conf_sum.get(pred_raw.token, 0.0) + float(pred_raw.confidence)
                            timeline.append(
                                {
                                    "frame_idx": int(idx),
                                    "live_token": pred_raw.token,
                                    "live_confidence": float(pred_raw.confidence),
                                    "live_margin": float(pred_raw.margin),
                                    "vote_ratio": float(vote_ratio),
                                    "token": pred_raw.token,
                                    "confidence": float(pred_raw.confidence),
                                    "is_committed": bool(True),
                                }
                            )

                if is_committed or (accepted and live_conf >= 0.60 and live_margin >= 0.20):
                    timeline.append(
                        {
                            "frame_idx": int(idx),
                            "live_token": live_token,
                            "live_confidence": float(live_conf),
                            "live_margin": float(live_margin),
                            "vote_ratio": float(vote_ratio),
                            "token": token,
                            "confidence": float(conf),
                            "is_committed": bool(is_committed),
                        }
                    )
            idx += 1
            if sampled >= max_frames:
                break

        cap.release()

        if sampled == 0:
            raise HTTPException(status_code=400, detail="No frames sampled from uploaded video")
        if inferred == 0:
            # Short clip fallback: pad to window size and do one direct ONNX pass.
            fallback_token = no_sign
            fallback_conf = 0.0
            fallback_inferred = 0
            if len(window.buffer) > 0 and onnx_runner is not None:
                raw_seq = np.stack(list(window.buffer), axis=0)  # (n, F)
                raw_seq_n = int(raw_seq.shape[0])
                raw_seq = _zero_pad_or_truncate(raw_seq, settings.window_size)

                pred = onnx_runner.predict(raw_seq)
                hand_ratio, motion_score, good_activity = _window_activity_stats(raw_seq, has_no_sign_vocab=has_no_sign_vocab)
                accepted = onnx_runner.accepts(pred.y, pred.confidence, pred.margin)
                strong_static = bool(pred.confidence >= 0.85 and pred.margin >= 0.30)
                activity_ok = bool(good_activity or strong_static)
                if settings.require_hand_for_sign and hand_ratio < settings.min_hand_presence_ratio:
                    activity_ok = False

                if (not settings.require_hand_for_sign or (any_hands_detected and activity_ok)) and accepted:
                    fallback_token = pred.token
                    fallback_conf = float(pred.confidence)
                elif (
                    raw_seq_n <= settings.window_size
                    and (not settings.require_hand_for_sign or (any_hands_detected and hand_ratio >= 0.75))
                    and pred.confidence >= 0.55
                    and pred.margin >= 0.35
                ):
                    # Short-clip rescue for clear but under-threshold single-window signs.
                    fallback_token = pred.token
                    fallback_conf = float(pred.confidence)
                elif (
                    not has_no_sign_vocab
                    and (not settings.require_hand_for_sign or (any_hands_detected and activity_ok))
                    and pred.confidence >= 0.30
                    and pred.margin >= 0.12
                ):
                    # No explicit NO_SIGN class: allow moderate-confidence short clips.
                    fallback_token = pred.token
                    fallback_conf = float(pred.confidence)
                fallback_inferred = 1

            return {
                "status": "ok",
                "note": "Not enough sampled frames to fill inference window",
                "sampled_frames": sampled,
                "total_frames": total_frames,
                "sample_fps": sample_fps,
                "window_size": settings.window_size,
                "inferred_windows": fallback_inferred,
                "any_hands_detected": bool(any_hands_detected),
                "final_token": fallback_token,
                "final_confidence": float(fallback_conf),
                "committed_token": fallback_token if fallback_token != no_sign else no_sign,
                "committed_confidence": float(fallback_conf) if fallback_token != no_sign else 0.0,
                "top_live_tokens": [],
                "top_committed_tokens": [],
                "timeline": [],
            }

        # Final token from latest timeline event if present, otherwise default.
        final_token = no_sign
        final_conf = 0.0
        if timeline:
            final_token = str(timeline[-1]["token"])
            final_conf = float(timeline[-1]["confidence"])
        if final_token != no_sign:
            min_final_conf = (
                settings.min_confidence_enter
                if settings.no_sign_token in onnx_runner.vocab
                else max(0.40, settings.min_confidence_enter_no_no_sign)
            )
            if final_conf < min_final_conf:
                final_token = no_sign
                final_conf = 0.0

        if by_window_pred:
            committed_token, committed_count = max(by_window_pred.items(), key=lambda kv: kv[1])
            committed_conf = float(by_window_conf_sum.get(committed_token, 0.0) / max(1, committed_count))
            committed_ratio = float(committed_count / max(1, inferred))
            # Adaptive thresholds: allow short uploads, still reject OOD spikes.
            if inferred <= 2:
                min_commit_count = 1
                min_commit_ratio = 0.34
            else:
                min_commit_count = 2
                min_commit_ratio = 0.45
            min_commit_conf = 0.55 if not has_no_sign_vocab else settings.min_confidence_enter
            if (
                committed_count < min_commit_count
                or committed_ratio < min_commit_ratio
                or committed_conf < min_commit_conf
            ):
                # Strong confident segment override.
                if not (committed_count >= 3 and committed_conf >= 0.85):
                    committed_token = no_sign
                    committed_conf = 0.0
        elif by_committed:
            committed_token, committed_count = max(by_committed.items(), key=lambda kv: kv[1])
            committed_conf = float(by_committed_conf_sum.get(committed_token, 0.0) / max(1, committed_count))
            committed_ratio = float(committed_count / max(1, inferred))
            if (
                committed_count < 2
                or committed_ratio < 0.45
                or committed_conf < 0.55
            ):
                # If a strong confident segment appears for multiple windows,
                # keep it even if it occupies a smaller fraction of the clip.
                if not (committed_count >= 3 and committed_conf >= 0.85):
                    committed_token = no_sign
                    committed_conf = 0.0
        else:
            committed_token = no_sign
            committed_conf = 0.0

        # Targeted rescue for very short clips on vocab-only models.
        # This prevents NO_SIGN on in-domain short uploads where strict acceptance is too harsh.
        if (
            committed_token == no_sign
            and not has_no_sign_vocab
            and inferred <= 2
            and by_window_relaxed
        ):
            relaxed_token, relaxed_count = max(by_window_relaxed.items(), key=lambda kv: kv[1])
            relaxed_ratio = float(relaxed_count / max(1, inferred))
            relaxed_mean_conf = float(by_window_relaxed_conf_sum.get(relaxed_token, 0.0) / max(1, relaxed_count))
            relaxed_mean_margin = float(by_window_relaxed_margin_sum.get(relaxed_token, 0.0) / max(1, relaxed_count))
            ratio_ok = relaxed_ratio >= 0.50
            conf_ok = relaxed_mean_conf >= 0.19
            margin_ok = relaxed_mean_margin >= 0.01
            if ratio_ok and conf_ok and margin_ok:
                committed_token = relaxed_token
                committed_conf = float(relaxed_mean_conf)

        # Keep upload output consistent: use committed decision as final.
        if committed_token != no_sign:
            final_token = committed_token
            final_conf = float(committed_conf)
        else:
            final_token = no_sign
            final_conf = 0.0

        if final_token == no_sign:
            final_conf = 0.0

        top_live = sorted(by_live.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_committed = sorted(by_committed.items(), key=lambda kv: kv[1], reverse=True)[:5]

        return {
            "status": "ok",
            "filename": file.filename,
            "sampled_frames": sampled,
            "total_frames": total_frames,
            "sample_fps": sample_fps,
            "window_size": settings.window_size,
            "inferred_windows": inferred,
            "any_hands_detected": bool(any_hands_detected),
            "final_token": final_token,
            "final_confidence": float(final_conf),
            "committed_token": committed_token,
            "committed_confidence": float(committed_conf),
            "top_live_tokens": [{"token": t, "count": int(c)} for t, c in top_live],
            "top_committed_tokens": [{"token": t, "count": int(c)} for t, c in top_committed],
            "committed_windows": int(committed),
            "timeline": timeline[-120:],
        }
    finally:
        if local_extractor is not None:
            try:
                local_extractor.close()
            except Exception:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    await ws.accept()

    pending: Optional[PendingFrame] = None
    window, stabilizer = _init_state()

    try:
        while True:
            msg = await ws.receive()

            if msg.get("text") is not None:
                try:
                    meta_raw = json.loads(msg["text"])
                    meta = FrameMeta.model_validate(meta_raw)
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "error", "message": f"Invalid frame_meta: {str(e)}"}))
                    continue
                pending = PendingFrame(meta=meta)
                continue

            if msg.get("bytes") is not None:
                if pending is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "Missing prior frame_meta"}))
                    continue

                payload: bytes = msg["bytes"]
                if len(payload) > settings.max_frame_bytes:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "frame_id": pending.meta.frame_id,
                                "message": "Frame too large",
                            }
                        )
                    )
                    pending = None
                    continue

                start = time.perf_counter()
                try:
                    token, confidence, is_committed, debug = _infer_one_frame(
                        payload,
                        window=window,
                        stabilizer=stabilizer,
                    )
                except Exception as e:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "frame_id": pending.meta.frame_id,
                                "message": f"Inference failed: {str(e)}",
                            }
                        )
                    )
                    pending = None
                    continue

                latency_ms = int((time.perf_counter() - start) * 1000)
                out = PredictionMessage(
                    frame_id=pending.meta.frame_id,
                    token=token,
                    confidence=float(confidence),
                    is_committed=is_committed,
                    latency_ms=latency_ms,
                    debug=debug,
                )
                await ws.send_text(out.model_dump_json())
                pending = None
                continue

            await ws.send_text(json.dumps({"type": "error", "message": "Unknown message"}))

    except WebSocketDisconnect:
        return


@app.websocket("/ws")
async def ws_simple(ws: WebSocket):
    """Simple binary-frames-only endpoint for browser testing."""

    await ws.accept()

    window, stabilizer = _init_state()
    frame_counter = count(1)

    try:
        while True:
            msg = await ws.receive()

            if msg.get("bytes") is None:
                await ws.send_text(json.dumps({"type": "error", "message": "Expected binary JPEG bytes"}))
                continue

            payload: bytes = msg["bytes"]
            if len(payload) > settings.max_frame_bytes:
                await ws.send_text(json.dumps({"type": "error", "message": "Frame too large"}))
                continue

            start = time.perf_counter()
            frame_id = f"f{next(frame_counter)}"

            try:
                token, confidence, is_committed, debug = _infer_one_frame(
                    payload,
                    window=window,
                    stabilizer=stabilizer,
                )
                latency_ms = int((time.perf_counter() - start) * 1000)

                out = PredictionMessage(
                    frame_id=frame_id,
                    token=token,
                    confidence=float(confidence),
                    is_committed=is_committed,
                    latency_ms=latency_ms,
                    debug=debug,
                )
                await ws.send_text(out.model_dump_json())
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "frame_id": frame_id, "message": str(e)}))

    except WebSocketDisconnect:
        return
