from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from shared.sequence_preprocess import (
    LEFT_COORD_SLICE,
    POSE_COORD_SLICE,
    RIGHT_COORD_SLICE,
    SequencePreprocessor,
)
from training.dataset import SampleItem, load_manifest


class SequenceAugmentor:
    """Lightweight train-time augmentation on raw landmark windows."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        jitter_std: float = 0.008,
        jitter_prob: float = 0.8,
        temporal_shift_max: int = 2,
        temporal_shift_prob: float = 0.5,
        speed_perturb_prob: float = 0.35,
        speed_perturb_min: float = 0.9,
        speed_perturb_max: float = 1.15,
        frame_drop_prob: float = 0.25,
        frame_drop_max: int = 2,
        coord_dropout_prob: float = 0.30,
        coord_dropout_ratio: float = 0.08,
    ) -> None:
        self.enabled = bool(enabled)
        self.jitter_std = float(max(0.0, jitter_std))
        self.jitter_prob = float(np.clip(jitter_prob, 0.0, 1.0))
        self.temporal_shift_max = int(max(0, temporal_shift_max))
        self.temporal_shift_prob = float(np.clip(temporal_shift_prob, 0.0, 1.0))
        self.speed_perturb_prob = float(np.clip(speed_perturb_prob, 0.0, 1.0))
        self.speed_perturb_min = float(max(0.5, speed_perturb_min))
        self.speed_perturb_max = float(max(self.speed_perturb_min, speed_perturb_max))
        self.frame_drop_prob = float(np.clip(frame_drop_prob, 0.0, 1.0))
        self.frame_drop_max = int(max(0, frame_drop_max))
        self.coord_dropout_prob = float(np.clip(coord_dropout_prob, 0.0, 1.0))
        self.coord_dropout_ratio = float(np.clip(coord_dropout_ratio, 0.0, 1.0))

        self._coord_idx = np.r_[
            np.arange(LEFT_COORD_SLICE.start, LEFT_COORD_SLICE.stop),
            np.arange(RIGHT_COORD_SLICE.start, RIGHT_COORD_SLICE.stop),
            np.arange(POSE_COORD_SLICE.start, POSE_COORD_SLICE.stop),
        ]

    @classmethod
    def from_dict(cls, raw: dict | None) -> "SequenceAugmentor":
        if not raw:
            return cls(enabled=False)
        return cls(
            enabled=bool(raw.get("enabled", False)),
            jitter_std=float(raw.get("jitter_std", 0.008)),
            jitter_prob=float(raw.get("jitter_prob", 0.8)),
            temporal_shift_max=int(raw.get("temporal_shift_max", 2)),
            temporal_shift_prob=float(raw.get("temporal_shift_prob", 0.5)),
            speed_perturb_prob=float(raw.get("speed_perturb_prob", 0.35)),
            speed_perturb_min=float(raw.get("speed_perturb_min", 0.9)),
            speed_perturb_max=float(raw.get("speed_perturb_max", 1.15)),
            frame_drop_prob=float(raw.get("frame_drop_prob", 0.25)),
            frame_drop_max=int(raw.get("frame_drop_max", 2)),
            coord_dropout_prob=float(raw.get("coord_dropout_prob", 0.30)),
            coord_dropout_ratio=float(raw.get("coord_dropout_ratio", 0.08)),
        )

    def apply(self, X: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return X
        x = X.astype(np.float32, copy=True)
        T = x.shape[0]

        # Small temporal shift simulates timing variance in real-time input.
        if self.temporal_shift_max > 0 and np.random.rand() < self.temporal_shift_prob:
            shift = int(np.random.randint(-self.temporal_shift_max, self.temporal_shift_max + 1))
            if shift != 0:
                x = np.roll(x, shift=shift, axis=0)
                if shift > 0:
                    x[:shift] = x[shift]
                else:
                    x[shift:] = x[shift - 1]

        # Slight speed perturbation via temporal interpolation.
        if T >= 4 and np.random.rand() < self.speed_perturb_prob:
            speed = float(np.random.uniform(self.speed_perturb_min, self.speed_perturb_max))
            old_t = np.arange(T, dtype=np.float32)
            center = (T - 1) / 2.0
            new_t = np.clip((old_t - center) * speed + center, 0.0, T - 1.0)
            lo = np.floor(new_t).astype(np.int32)
            hi = np.minimum(lo + 1, T - 1)
            a = (new_t - lo).astype(np.float32)[:, None]
            x = ((1.0 - a) * x[lo] + a * x[hi]).astype(np.float32)

        # Coordinate jitter improves robustness to detector noise.
        if self.jitter_std > 0 and np.random.rand() < self.jitter_prob:
            noise = np.random.normal(0.0, self.jitter_std, size=(T, len(self._coord_idx))).astype(np.float32)
            x[:, self._coord_idx] += noise

        # Randomly drop a small subset of coordinates (occlusion robustness).
        if self.coord_dropout_ratio > 0 and np.random.rand() < self.coord_dropout_prob:
            k = max(1, int(round(len(self._coord_idx) * self.coord_dropout_ratio)))
            cols = np.random.choice(self._coord_idx, size=min(k, len(self._coord_idx)), replace=False)
            x[:, cols] = 0.0

        # Randomly blank a few frames to mimic intermittent hand-tracking dropouts.
        if self.frame_drop_max > 0 and np.random.rand() < self.frame_drop_prob:
            n_drop = int(np.random.randint(1, self.frame_drop_max + 1))
            drop_idx = np.random.choice(T, size=min(n_drop, T), replace=False)
            x[np.ix_(drop_idx, self._coord_idx)] = 0.0

        return x


class NpzSequenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        *,
        preprocessor: SequencePreprocessor | None = None,
        augmentor: SequenceAugmentor | None = None,
    ):
        self.items: list[SampleItem] = load_manifest(manifest_path)
        self.preprocessor = preprocessor
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.items)

    @property
    def labels(self) -> list[int]:
        return [int(it.y) for it in self.items]

    def __getitem__(self, idx: int):
        it = self.items[idx]
        data = np.load(it.path, allow_pickle=False)
        X = data["X"].astype(np.float32)  # (T, F)
        y = int(data["y"])

        if self.augmentor is not None:
            X = self.augmentor.apply(X)

        if self.preprocessor is not None:
            X = self.preprocessor.transform_window(X)

        # Optional runtime shape checks (useful for baseline training expectations)
        exp_t = int(os.environ.get("ISL_EXPECT_T", "0") or 0)
        exp_f = int(os.environ.get("ISL_EXPECT_F", "0") or 0)
        if exp_t and X.shape[0] != exp_t:
            raise ValueError(f"Unexpected T in {it.path}: got {X.shape[0]} expected {exp_t}")
        if exp_f and X.shape[1] != exp_f:
            raise ValueError(f"Unexpected F in {it.path}: got {X.shape[1]} expected {exp_f}")

        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)
