
from __future__ import annotations

# Allow running as both a script and as a module
import sys
import os
if __name__ == "__main__" and ("app" not in sys.modules and not __package__):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.landmarks import FEATURE_DIM, LandmarkExtractor
from app.onnx_infer import OnnxLstmRunner
from app.protocol import FrameMeta, PredictionMessage
from app.settings import settings


app = FastAPI(title=settings.api_name)


# MediaPipe is relatively expensive to initialize; keep one extractor per process.
extractor = LandmarkExtractor(enable_canonical_mirroring=True)


def _try_create_onnx_runner() -> OnnxLstmRunner | None:
    if not settings.use_onnx:
        return None
    try:
        return OnnxLstmRunner(onnx_path=settings.onnx_model_path, vocab_path=settings.vocab_path)
    except Exception:
        # Keep API running even if model isn't present.
        return None


onnx_runner = _try_create_onnx_runner()


@app.get("/health")
def health():
    return {"status": "ok"}


@dataclass
class PendingFrame:
    meta: FrameMeta


class SlidingWindow:
    """Maintains a sliding window of recent landmark features for windowed inference."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self.last_prediction: str = "NO_SIGN"
        self.last_confidence: float = 0.0
        self.stable_count: int = 0

    def add_frame(self, features: np.ndarray) -> None:
        self.buffer.append(features)

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.window_size

    def get_window(self) -> np.ndarray:
        """Returns (T, F) array of recent frames."""
        if not self.is_ready():
            raise ValueError("Window not ready")
        return np.stack(list(self.buffer), axis=0)

    def update_smoothing(self, token: str, confidence: float, stability_threshold: int = 3) -> tuple[str, float, bool]:
        """Simple smoothing: commit a token only if it's stable for N consecutive windows.

        Returns: (token, confidence, is_committed)
        """
        if token == self.last_prediction:
            self.stable_count += 1
        else:
            self.stable_count = 1
            self.last_prediction = token
            self.last_confidence = confidence

        is_committed = self.stable_count >= stability_threshold
        return token, confidence, is_committed


def _decode_image(payload: bytes) -> np.ndarray:
    # Decode JPEG/WebP bytes into an image ndarray (BGR)
    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    await ws.accept()

    pending: Optional[PendingFrame] = None
    window = SlidingWindow(window_size=settings.window_size)

    try:
        while True:
            msg = await ws.receive()

            # Text message = metadata
            if msg.get("text") is not None:
                try:
                    meta_raw = json.loads(msg["text"])
                    meta = FrameMeta.model_validate(meta_raw)
                except Exception as e:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": f"Invalid frame_meta: {str(e)}",
                            }
                        )
                    )
                    continue

                pending = PendingFrame(meta=meta)
                continue

            # Bytes message = image payload
            if msg.get("bytes") is not None:
                if pending is None:
                    await ws.send_text(
                        json.dumps(
                            {"type": "error", "message": "Missing prior frame_meta"}
                        )
                    )
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

                # Phase 4: decode + extract landmarks + sliding window + inference hook.
                # Next phase: ONNX model inference replaces the stub below.
                try:
                    img = _decode_image(payload)
                    features, ex_debug = extractor.extract(img)

                    # Add to sliding window
                    window.add_frame(features)

                    # Inference
                    token = "NO_SIGN"
                    confidence = 1.0
                    is_committed = False

                    if window.is_ready():
                        if onnx_runner is not None:
                            pred = onnx_runner.predict(window.get_window())
                            token, confidence, is_committed = window.update_smoothing(
                                pred.token,
                                pred.confidence,
                            )
                        else:
                            # Fallback: stub inference always returns NO_SIGN
                            token, confidence, is_committed = window.update_smoothing("NO_SIGN", 0.99)

                    debug = {
                        "pose_detected": ex_debug.pose_present,
                        "left_hand_detected": ex_debug.left_hand_present,
                        "right_hand_detected": ex_debug.right_hand_present,
                        "canonical_mirrored": ex_debug.canonical_mirrored,
                        "feature_dim": int(features.shape[0]),
                        "window_ready": window.is_ready(),
                        "window_fill": len(window.buffer),
                        "onnx_enabled": settings.use_onnx,
                        "onnx_loaded": onnx_runner is not None,
                    }
                except Exception as e:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "frame_id": pending.meta.frame_id,
                                "message": f"Decode failed: {str(e)}",
                            }
                        )
                    )
                    pending = None
                    continue

                latency_ms = int((time.perf_counter() - start) * 1000)

                out = PredictionMessage(
                    frame_id=pending.meta.frame_id,
                    token=token,
                    confidence=confidence,
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
    """Minimal WS endpoint for testing.

    Protocol:
      - Client sends: binary JPEG frames only
      - Server replies: PredictionMessage JSON per received frame

    This endpoint exists to make browser testing simple (no separate frame_meta).
    """

    await ws.accept()

    window = SlidingWindow(window_size=settings.window_size)
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
                img = _decode_image(payload)
                features, ex_debug = extractor.extract(img)
                window.add_frame(features)

                token = "NO_SIGN"
                confidence = 1.0
                is_committed = False

                if window.is_ready():
                    if onnx_runner is not None:
                        pred = onnx_runner.predict(window.get_window())
                        if __name__ == "__main__":
                            import uvicorn
                            uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
                        token, confidence, is_committed = window.update_smoothing(
                            pred.token,
                            pred.confidence,
                        )
                    else:
                        token, confidence, is_committed = window.update_smoothing("NO_SIGN", 0.99)

                latency_ms = int((time.perf_counter() - start) * 1000)

                debug = {
                    "pose_detected": ex_debug.pose_present,
                    "left_hand_detected": ex_debug.left_hand_present,
                    "right_hand_detected": ex_debug.right_hand_present,
                    "canonical_mirrored": ex_debug.canonical_mirrored,
                    "feature_dim": int(features.shape[0]),
                    "window_ready": window.is_ready(),
                    "window_fill": len(window.buffer),
                    "onnx_enabled": settings.use_onnx,
                    "onnx_loaded": onnx_runner is not None,
                }

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
