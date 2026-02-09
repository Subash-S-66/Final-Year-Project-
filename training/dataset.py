from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SampleItem:
    path: Path
    label: str
    y: int
    person_id: str


def read_npz_sample(path: Path) -> SampleItem:
    data = np.load(path, allow_pickle=False)
    y = int(data["y"])
    label = str(data["label"])

    # person_id was introduced after initial samples; fallback keeps compatibility.
    if "person_id" in data.files:
        person_id = str(data["person_id"])
    else:
        person_id = "unknown"

    return SampleItem(path=path, label=label, y=y, person_id=person_id)


def load_manifest(manifest_path: Path) -> list[SampleItem]:
    items: list[SampleItem] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if not p.is_absolute():
            p = (manifest_path.parent / p).resolve()
        items.append(read_npz_sample(p))
    return items


def load_Xy(items: list[SampleItem]) -> tuple[np.ndarray, np.ndarray]:
    Xs = []
    ys = []
    for it in items:
        data = np.load(it.path, allow_pickle=False)
        X = data["X"].astype(np.float32)  # (T, F)
        Xs.append(X)
        ys.append(np.int64(it.y))
    return np.stack(Xs, axis=0), np.array(ys, dtype=np.int64)


def aggregate_static_features(X_seq: np.ndarray, method: str = "mean") -> np.ndarray:
    """Convert (N,T,F) sequences into (N,F) static features.

    - mean: average over time
    - last: last frame
    """
    if method == "mean":
        return X_seq.mean(axis=1)
    if method == "last":
        return X_seq[:, -1, :]
    raise ValueError(f"Unknown method: {method}")


def infer_TF_from_any_sample(samples_root: Path) -> Optional[tuple[int, int]]:
    for p in samples_root.rglob("*.npz"):
        data = np.load(p, allow_pickle=False)
        X = data["X"]
        if X.ndim == 2:
            return int(X.shape[0]), int(X.shape[1])
    return None


def scan_samples_root(samples_root: Path) -> list[SampleItem]:
    """Scan <root>/<LABEL>/*.npz and return SampleItems."""
    items: list[SampleItem] = []
    if not samples_root.exists():
        return items

    for p in samples_root.rglob("*.npz"):
        try:
            items.append(read_npz_sample(p))
        except Exception:
            # Skip corrupt or unexpected files.
            continue

    return items


def split_train_val_by_clip(
    items: list[SampleItem],
    *,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[SampleItem], list[SampleItem]]:
    """Clip-level split (by .npz file). Not frame-level.

    Uses a label-aware strategy that does not fail when some labels have only 1 sample.
    - If a label has 1 sample, it stays in train.
    - If a label has >=2 samples, allocate ~val_ratio (at least 1) to val.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[SampleItem]] = {}
    for it in items:
        by_label.setdefault(it.label, []).append(it)

    train: list[SampleItem] = []
    val: list[SampleItem] = []

    labels = sorted(by_label.keys())
    for label in labels:
        group = list(by_label[label])
        rng.shuffle(group)
        n = len(group)
        if n <= 1:
            train.extend(group)
            continue
        n_val = int(round(n * val_ratio))
        n_val = max(1, min(n - 1, n_val))
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
