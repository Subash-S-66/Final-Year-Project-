from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training.dataset import aggregate_static_features, load_manifest, load_Xy
from training.utils import ensure_dir, load_vocab, repo_root, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a static baseline classifier (scikit-learn).")
    parser.add_argument("--vocab", default="configs/vocab_v1_51.json")
    parser.add_argument("--splits", default="data/processed/splits")
    parser.add_argument("--static", choices=["mean", "last"], default="mean")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    root = repo_root()
    vocab = load_vocab(root / args.vocab)
    splits = Path(root / args.splits)

    train_items = load_manifest(splits / "train.txt")
    val_items = load_manifest(splits / "val.txt") if (splits / "val.txt").exists() else []

    X_train_seq, y_train = load_Xy(train_items)
    X_train = aggregate_static_features(X_train_seq, method=args.static)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=None,
                    verbose=0,
                    multi_class="auto",
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    out_dir = ensure_dir(root / args.out)
    joblib_path = out_dir / "static_baseline.joblib"
    joblib.dump(clf, joblib_path)

    save_json(out_dir / "label_map.json", {"tokens": vocab})
    print(f"Saved model: {joblib_path}")

    if val_items:
        X_val_seq, y_val = load_Xy(val_items)
        X_val = aggregate_static_features(X_val_seq, method=args.static)
        pred = clf.predict(X_val)
        print("\nValidation report:")
        print(classification_report(y_val, pred, target_names=vocab, zero_division=0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
