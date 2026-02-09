from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    # backend/app -> backend -> repo
    return Path(__file__).resolve().parents[2]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


@dataclass(frozen=True)
class OnnxPrediction:
    token: str
    confidence: float
    logits: np.ndarray  # (C,)


class OnnxLstmRunner:
    """Minimal ONNXRuntime inference wrapper.

    Contract:
      input:  float32 (1, T=30, F=263)
      output: float32 (1, C=51) logits
    """

    def __init__(self, *, onnx_path: str, vocab_path: str) -> None:
        self.onnx_path = (Path(onnx_path) if Path(onnx_path).is_absolute() else _repo_root() / onnx_path).resolve()
        self.vocab_path = (Path(vocab_path) if Path(vocab_path).is_absolute() else _repo_root() / vocab_path).resolve()

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocab not found: {self.vocab_path}")

        self.vocab: list[str] = json.loads(self.vocab_path.read_text(encoding="utf-8"))

        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )

        # Keep names explicit to match export.
        self.input_name = "input"
        self.output_name = "logits"

    def predict(self, window: np.ndarray) -> OnnxPrediction:
        """Run inference.

        Args:
            window: (T, F) float32/float64

        Returns:
            token/confidence/logits
        """
        if window.ndim != 2:
            raise ValueError(f"Expected window shape (T, F), got {window.shape}")

        x = window.astype(np.float32, copy=False)[None, ...]  # (1, T, F)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]

        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

        logits1 = logits[0]
        if logits1.shape[0] != len(self.vocab):
            raise RuntimeError(
                f"Logits/vocab mismatch: logits={logits1.shape[0]} vocab={len(self.vocab)}"
            )

        prob = _softmax(logits1, axis=-1)
        idx = int(prob.argmax())
        return OnnxPrediction(token=self.vocab[idx], confidence=float(prob[idx]), logits=logits1)
