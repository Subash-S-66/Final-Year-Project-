from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from training.dataset import SampleItem, load_manifest


class NpzSequenceDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.items: list[SampleItem] = load_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        data = np.load(it.path, allow_pickle=False)
        X = data["X"].astype(np.float32)  # (T, F)
        y = int(data["y"])

        # Optional runtime shape checks (useful for baseline training expectations)
        exp_t = int(os.environ.get("ISL_EXPECT_T", "0") or 0)
        exp_f = int(os.environ.get("ISL_EXPECT_F", "0") or 0)
        if exp_t and X.shape[0] != exp_t:
            raise ValueError(f"Unexpected T in {it.path}: got {X.shape[0]} expected {exp_t}")
        if exp_f and X.shape[1] != exp_f:
            raise ValueError(f"Unexpected F in {it.path}: got {X.shape[1]} expected {exp_f}")

        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)
