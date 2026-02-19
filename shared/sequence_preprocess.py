from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


BASE_FEATURE_DIM = 263

LEFT_PRESENT_IDX = 0
LEFT_HANDEDNESS_IDX = 1
LEFT_COORD_SLICE = slice(2, 65)

RIGHT_PRESENT_IDX = 65
RIGHT_HANDEDNESS_IDX = 66
RIGHT_COORD_SLICE = slice(67, 130)

POSE_PRESENT_IDX = 130
POSE_COORD_SLICE = slice(131, 263)


def infer_delta_from_input_dim(input_dim: int) -> bool:
    if input_dim == BASE_FEATURE_DIM:
        return False
    if input_dim == BASE_FEATURE_DIM * 2:
        return True
    raise ValueError(
        f"Unsupported input feature dimension={input_dim}. "
        f"Expected {BASE_FEATURE_DIM} or {BASE_FEATURE_DIM * 2}."
    )


@dataclass(frozen=True)
class SequencePreprocessConfig:
    # EMA over time helps suppress frame-to-frame landmark jitter.
    ema_alpha: float = 0.65
    enable_ema: bool = True

    # Velocity features: delta[t] = x[t] - x[t-1]
    enable_delta: bool = True

    # Per-frame scale normalization and per-sequence z-score.
    per_frame_rms_norm: bool = True
    per_sequence_zscore: bool = True

    # Clamp extreme outliers after normalization.
    clip_value: float = 6.0
    eps: float = 1e-5

    @classmethod
    def from_dict(cls, raw: dict | None) -> "SequencePreprocessConfig":
        if not raw:
            return cls()
        defaults = cls()
        data = {
            "ema_alpha": float(raw.get("ema_alpha", defaults.ema_alpha)),
            "enable_ema": bool(raw.get("enable_ema", defaults.enable_ema)),
            "enable_delta": bool(raw.get("enable_delta", defaults.enable_delta)),
            "per_frame_rms_norm": bool(raw.get("per_frame_rms_norm", defaults.per_frame_rms_norm)),
            "per_sequence_zscore": bool(raw.get("per_sequence_zscore", defaults.per_sequence_zscore)),
            "clip_value": float(raw.get("clip_value", defaults.clip_value)),
            "eps": float(raw.get("eps", defaults.eps)),
        }
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


class SequencePreprocessor:
    def __init__(self, cfg: SequencePreprocessConfig) -> None:
        self.cfg = cfg
        self._coord_idx = np.r_[
            np.arange(LEFT_COORD_SLICE.start, LEFT_COORD_SLICE.stop),
            np.arange(RIGHT_COORD_SLICE.start, RIGHT_COORD_SLICE.stop),
            np.arange(POSE_COORD_SLICE.start, POSE_COORD_SLICE.stop),
        ]

    @property
    def output_dim(self) -> int:
        return BASE_FEATURE_DIM * (2 if self.cfg.enable_delta else 1)

    def transform_window(self, X: np.ndarray, *, force_enable_delta: bool | None = None) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != BASE_FEATURE_DIM:
            raise ValueError(f"Expected window shape (T,{BASE_FEATURE_DIM}), got {tuple(X.shape)}")

        use_delta = self.cfg.enable_delta if force_enable_delta is None else bool(force_enable_delta)
        x = X.astype(np.float32, copy=True)

        self._apply_presence_mask(x)

        if self.cfg.per_frame_rms_norm:
            self._per_frame_rms_normalize(x)

        if self.cfg.enable_ema:
            x = self._ema_smooth(x)
            self._apply_presence_mask(x)

        valid_mask = self._valid_mask(x)
        if self.cfg.per_sequence_zscore:
            self._zscore_with_mask_inplace(x, valid_mask)

        self._clip_coords_inplace(x)

        if not use_delta:
            return x

        delta = self._delta_features(x, valid_mask)
        if self.cfg.per_sequence_zscore:
            delta_mask = self._delta_mask(valid_mask)
            self._zscore_with_mask_inplace(delta, delta_mask)
        self._clip_coords_inplace(delta)
        return np.concatenate([x, delta], axis=1)

    def _apply_presence_mask(self, x: np.ndarray) -> None:
        left_present = (x[:, LEFT_PRESENT_IDX] >= 0.5).astype(np.float32)
        right_present = (x[:, RIGHT_PRESENT_IDX] >= 0.5).astype(np.float32)
        pose_present = (x[:, POSE_PRESENT_IDX] >= 0.5).astype(np.float32)

        x[:, LEFT_HANDEDNESS_IDX] *= left_present
        x[:, RIGHT_HANDEDNESS_IDX] *= right_present

        x[:, LEFT_COORD_SLICE] *= left_present[:, None]
        x[:, RIGHT_COORD_SLICE] *= right_present[:, None]
        x[:, POSE_COORD_SLICE] *= pose_present[:, None]

    def _valid_mask(self, x: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(x, dtype=bool)
        left_present = x[:, LEFT_PRESENT_IDX] >= 0.5
        right_present = x[:, RIGHT_PRESENT_IDX] >= 0.5
        pose_present = x[:, POSE_PRESENT_IDX] >= 0.5
        mask[:, LEFT_COORD_SLICE] = left_present[:, None]
        mask[:, RIGHT_COORD_SLICE] = right_present[:, None]
        mask[:, POSE_COORD_SLICE] = pose_present[:, None]
        return mask

    def _per_frame_rms_normalize(self, x: np.ndarray) -> None:
        valid = self._valid_mask(x)[:, self._coord_idx]
        coords = x[:, self._coord_idx]

        denom = np.maximum(valid.sum(axis=1, keepdims=True), 1)
        rms = np.sqrt((np.square(coords) * valid).sum(axis=1, keepdims=True) / denom)
        scale = np.where(rms > self.cfg.eps, rms, 1.0)
        coords = np.where(valid, coords / scale, 0.0).astype(np.float32)
        x[:, self._coord_idx] = coords

    def _ema_smooth(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        a = float(np.clip(self.cfg.ema_alpha, 0.0, 1.0))
        if out.shape[0] <= 1 or a <= 0.0:
            return out
        for t in range(1, out.shape[0]):
            out[t, self._coord_idx] = a * out[t, self._coord_idx] + (1.0 - a) * out[t - 1, self._coord_idx]
        return out

    def _delta_features(self, x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        delta = np.zeros_like(x, dtype=np.float32)
        delta[1:, :] = x[1:, :] - x[:-1, :]
        m = self._delta_mask(valid_mask)
        delta = np.where(m, delta, 0.0).astype(np.float32)
        return delta

    def _delta_mask(self, valid_mask: np.ndarray) -> np.ndarray:
        m = np.zeros_like(valid_mask, dtype=bool)
        m[1:, self._coord_idx] = valid_mask[1:, self._coord_idx] & valid_mask[:-1, self._coord_idx]
        return m

    def _zscore_with_mask_inplace(self, x: np.ndarray, valid_mask: np.ndarray) -> None:
        coords = x[:, self._coord_idx]
        mask = valid_mask[:, self._coord_idx]

        denom = np.maximum(mask.sum(axis=0, keepdims=True), 1)
        mean = (coords * mask).sum(axis=0, keepdims=True) / denom
        var = (np.square(coords - mean) * mask).sum(axis=0, keepdims=True) / denom
        std = np.sqrt(var + self.cfg.eps)

        norm = np.where(mask, (coords - mean) / std, 0.0).astype(np.float32)
        x[:, self._coord_idx] = norm

    def _clip_coords_inplace(self, x: np.ndarray) -> None:
        c = float(self.cfg.clip_value)
        if c <= 0:
            return
        x[:, self._coord_idx] = np.clip(x[:, self._coord_idx], -c, c)

