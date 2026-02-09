from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import confusion_matrix as _confusion_matrix
    from sklearn.metrics import f1_score as _f1_score
except Exception:  # pragma: no cover
    _confusion_matrix = None
    _f1_score = None


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean()) if y.size else 0.0


def topk_accuracy(logits: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    if y.size == 0:
        return 0.0
    topk = np.argsort(-logits, axis=1)[:, :k]
    hits = 0
    for i in range(y.shape[0]):
        if int(y[i]) in set(topk[i].tolist()):
            hits += 1
    return float(hits / y.shape[0])


def macro_f1(pred: np.ndarray, y: np.ndarray, *, num_classes: int) -> float:
    if y.size == 0:
        return 0.0
    if _f1_score is None:
        raise RuntimeError("scikit-learn is required for F1. Install scikit-learn.")
    labels = list(range(int(num_classes)))
    return float(_f1_score(y, pred, average="macro", labels=labels, zero_division=0))


def confusion(pred: np.ndarray, y: np.ndarray, *, num_classes: int) -> np.ndarray:
    if y.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    if _confusion_matrix is None:
        raise RuntimeError("scikit-learn is required for confusion matrix. Install scikit-learn.")
    labels = list(range(int(num_classes)))
    return _confusion_matrix(y, pred, labels=labels).astype(np.int64)
