from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

def _repo_root() -> Path:
    # backend/app -> backend -> repo
    return Path(__file__).resolve().parents[2]


try:
    from shared.sequence_preprocess import (
        BASE_FEATURE_DIM,
        SequencePreprocessConfig,
        SequencePreprocessor,
        infer_delta_from_input_dim,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(_repo_root()))
    from shared.sequence_preprocess import (
        BASE_FEATURE_DIM,
        SequencePreprocessConfig,
        SequencePreprocessor,
        infer_delta_from_input_dim,
    )


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


@dataclass(frozen=True)
class OnnxPrediction:
    y: int
    token: str
    confidence: float
    second_confidence: float
    margin: float
    logits: np.ndarray  # (C,)
    probs: np.ndarray  # (C,)


class OnnxLstmRunner:
    """Minimal ONNXRuntime inference wrapper.

    Contract:
      input:  float32 (1, T=30, F=263|526)
      output: float32 (1, C=51) logits
    """

    def __init__(self, *, onnx_path: str, vocab_path: str, meta_path: str | None = None) -> None:
        self.onnx_path = (Path(onnx_path) if Path(onnx_path).is_absolute() else _repo_root() / onnx_path).resolve()
        self.vocab_path = (Path(vocab_path) if Path(vocab_path).is_absolute() else _repo_root() / vocab_path).resolve()
        self.meta_path = (
            (Path(meta_path) if Path(meta_path).is_absolute() else _repo_root() / meta_path).resolve()
            if meta_path
            else None
        )

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocab not found: {self.vocab_path}")

        self.vocab: list[str] = json.loads(self.vocab_path.read_text(encoding="utf-8"))
        self.meta: dict = {}
        if self.meta_path and self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )

        # Keep names explicit to match export.
        self.input_name = "input"
        self.output_name = "logits"

        input_shape = self.session.get_inputs()[0].shape
        self.expected_t = int(input_shape[1]) if isinstance(input_shape[1], int) else int(self.meta.get("T", 30))
        self.expected_f = int(input_shape[2]) if isinstance(input_shape[2], int) else int(self.meta.get("F", BASE_FEATURE_DIM))

        enable_delta = infer_delta_from_input_dim(self.expected_f)
        pre_cfg = SequencePreprocessConfig.from_dict(self.meta.get("preprocess", {}))
        pre_cfg = SequencePreprocessConfig(
            ema_alpha=pre_cfg.ema_alpha,
            enable_ema=pre_cfg.enable_ema,
            enable_delta=enable_delta,
            per_frame_rms_norm=pre_cfg.per_frame_rms_norm,
            per_sequence_zscore=pre_cfg.per_sequence_zscore,
            clip_value=pre_cfg.clip_value,
            eps=pre_cfg.eps,
        )
        self.preprocessor = SequencePreprocessor(pre_cfg)
        self.temperature = float(max(1e-2, float(self.meta.get("temperature", 1.0))))
        acceptance = self.meta.get("acceptance", {}) if isinstance(self.meta.get("acceptance", {}), dict) else {}
        self.global_conf_threshold = float(max(0.0, acceptance.get("global_conf", 0.0)))
        self.global_margin_threshold = float(max(0.0, acceptance.get("global_margin", 0.0)))
        class_conf = acceptance.get("class_conf", [])
        class_margin = acceptance.get("class_margin", [])
        self.class_conf_thresholds = (
            np.array(class_conf, dtype=np.float32)
            if isinstance(class_conf, list) and len(class_conf) == len(self.vocab)
            else None
        )
        self.class_margin_thresholds = (
            np.array(class_margin, dtype=np.float32)
            if isinstance(class_margin, list) and len(class_margin) == len(self.vocab)
            else None
        )

    def acceptance_thresholds_for(self, class_idx: int) -> tuple[float, float]:
        conf_thr = self.global_conf_threshold
        margin_thr = self.global_margin_threshold
        if self.class_conf_thresholds is not None and 0 <= class_idx < self.class_conf_thresholds.shape[0]:
            conf_thr = float(max(conf_thr, self.class_conf_thresholds[class_idx]))
        if self.class_margin_thresholds is not None and 0 <= class_idx < self.class_margin_thresholds.shape[0]:
            margin_thr = float(max(margin_thr, self.class_margin_thresholds[class_idx]))
        return conf_thr, margin_thr

    def accepts(self, class_idx: int, confidence: float, margin: float) -> bool:
        conf_thr, margin_thr = self.acceptance_thresholds_for(class_idx)
        if confidence < conf_thr:
            return False
        if margin < margin_thr:
            return False
        return True

    def predict(self, window: np.ndarray) -> OnnxPrediction:
        """Run inference.

        Args:
            window: (T, F) float32/float64

        Returns:
            token/confidence/logits
        """
        if window.ndim != 2:
            raise ValueError(f"Expected window shape (T, F), got {window.shape}")
        if window.shape[0] != self.expected_t:
            raise ValueError(f"Expected T={self.expected_t}, got {window.shape[0]}")
        if window.shape[1] != BASE_FEATURE_DIM:
            raise ValueError(f"Expected raw F={BASE_FEATURE_DIM}, got {window.shape[1]}")

        processed = self.preprocessor.transform_window(window, force_enable_delta=(self.expected_f == BASE_FEATURE_DIM * 2))
        if processed.shape[1] != self.expected_f:
            raise RuntimeError(f"Preprocess feature mismatch: got {processed.shape[1]} expected {self.expected_f}")

        x = processed.astype(np.float32, copy=False)[None, ...]  # (1, T, F)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]

        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

        logits1 = logits[0]
        if logits1.shape[0] != len(self.vocab):
            raise RuntimeError(
                f"Logits/vocab mismatch: logits={logits1.shape[0]} vocab={len(self.vocab)}"
            )

        prob = _softmax(logits1 / self.temperature, axis=-1)
        idx = int(prob.argmax())
        second = float(np.partition(prob, -2)[-2]) if prob.shape[0] > 1 else 0.0
        conf = float(prob[idx])
        margin = float(max(0.0, conf - second))
        return OnnxPrediction(
            y=idx,
            token=self.vocab[idx],
            confidence=conf,
            second_confidence=second,
            margin=margin,
            logits=logits1,
            probs=prob,
        )
