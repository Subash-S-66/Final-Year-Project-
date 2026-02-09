from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch might not be installed yet
        pass


def load_vocab(vocab_path: str | Path) -> list[str]:
    p = Path(vocab_path)
    with p.open("r", encoding="utf-8") as f:
        tokens = json.load(f)
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
        raise ValueError("Invalid vocab JSON")
    return tokens


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
