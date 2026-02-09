from __future__ import annotations

import argparse
from pathlib import Path

import csv
import json
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from training.dataset import scan_samples_root, split_train_val_by_clip
from training.metrics import accuracy, confusion, macro_f1, topk_accuracy
from training.models import LSTMClassifier
from training.torch_dataset import NpzSequenceDataset
from training.utils import ensure_dir, load_vocab, repo_root, save_json, set_seed


def _run_epoch(model, loader, *, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()

    all_logits = []
    all_y = []
    losses = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 1), dtype=np.float32)
    y_np = np.concatenate(all_y, axis=0) if all_y else np.zeros((0,), dtype=np.int64)
    pred = logits_np.argmax(axis=1) if logits_np.size else np.zeros((0,), dtype=np.int64)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": accuracy(pred, y_np),
        "top5": topk_accuracy(logits_np, y_np, k=min(5, logits_np.shape[1])) if logits_np.size else 0.0,
        "pred": pred,
        "y": y_np,
    }


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _plot_curves(history: list[dict[str, Any]], out_png: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]

    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    tr_acc = [h["train_acc"] for h in history]
    va_acc = [h["val_acc"] for h in history]
    tr_f1 = [h["train_f1"] for h in history]
    va_f1 = [h["val_f1"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, tr_loss, label="train")
    axes[0].plot(epochs, va_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, tr_acc, label="train acc")
    axes[1].plot(epochs, va_acc, label="val acc")
    axes[1].plot(epochs, tr_f1, label="train f1")
    axes[1].plot(epochs, va_f1, label="val f1")
    axes[1].set_title("Accuracy / Macro-F1")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_confusion(cm: np.ndarray, tokens: list[str], out_png: Path, *, max_labels: int = 51) -> None:
    # For 51 labels this is fine; keep it readable.
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")

    labels = tokens[:max_labels]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _write_epoch_log_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)


def main() -> int:
    parser = argparse.ArgumentParser(description="PHASE 7: Train baseline LSTM model (PyTorch) from processed .npz clips.")
    parser.add_argument("--config", default="train_lstm.yaml", help="YAML config path")
    args = parser.parse_args()

    root = repo_root()
    cfg_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Missing config: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Config
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # Support both single root and multiple roots
    data_cfg = cfg["data"]
    if "roots" in data_cfg:
        data_roots = [root / r for r in data_cfg["roots"]]
    else:
        data_roots = [(root / data_cfg["root"]).resolve()]

    vocab_path = (root / data_cfg["vocab"]).resolve()
    val_ratio = float(data_cfg["split"].get("val_ratio", 0.2))
    split_seed = int(data_cfg["split"].get("seed", seed))

    hidden_dim = int(cfg["model"].get("hidden_dim", 256))
    num_layers = int(cfg["model"].get("num_layers", 2))
    dropout = float(cfg["model"].get("dropout", 0.2))
    bidirectional = bool(cfg["model"].get("bidirectional", False))

    epochs = int(cfg["train"].get("epochs", 50))
    batch_size = int(cfg["train"].get("batch_size", 16))
    lr = float(cfg["train"].get("lr", 3e-4))
    weight_decay = float(cfg["train"].get("weight_decay", 0.01))
    patience = int(cfg["train"].get("patience", 10))
    num_workers = int(cfg["train"].get("num_workers", 0))

    device = _resolve_device(str(cfg.get("runtime", {}).get("device", "auto")))

    outputs = cfg.get("outputs", {})
    checkpoints_dir = ensure_dir(root / outputs.get("checkpoints_dir", "reports/checkpoints"))
    run_dir = ensure_dir(root / outputs.get("run_dir", "reports/train_lstm_run"))
    log_csv = (root / outputs.get("log_csv", "reports/train_lstm_log.csv")).resolve()
    curves_png = (root / outputs.get("curves_png", "reports/train_lstm_curves.png")).resolve()
    confusion_png = (root / outputs.get("confusion_png", "reports/train_lstm_confusion.png")).resolve()
    metrics_json = (root / outputs.get("metrics_json", "reports/train_lstm_metrics.json")).resolve()

    vocab = load_vocab(vocab_path)
    num_classes = len(vocab)

    # Scan multiple dataset roots & merge
    items = []
    for dr in data_roots:
        items.extend(scan_samples_root(dr))

    if not items:
        raise SystemExit(f"No .npz samples found under: {data_roots}")

    train_items, val_items = split_train_val_by_clip(items, val_ratio=val_ratio, seed=split_seed)
    if not train_items or not val_items:
        raise SystemExit(f"Split resulted in train={len(train_items)} val={len(val_items)}. Need more clips.")

    # Write manifests for reproducibility
    train_manifest = run_dir / "train.txt"
    val_manifest = run_dir / "val.txt"
    train_manifest.write_text("\n".join(str(it.path) for it in train_items) + "\n", encoding="utf-8")
    val_manifest.write_text("\n".join(str(it.path) for it in val_items) + "\n", encoding="utf-8")

    train_ds = NpzSequenceDataset(train_manifest)
    val_ds = NpzSequenceDataset(val_manifest)

    # Infer T/F, and validate against expected T=30 F=263
    X0, _y0 = train_ds[0]
    T, F = int(X0.shape[0]), int(X0.shape[1])
    if T != 30 or F != 263:
        print(f"WARN: expected T=30,F=263 but got T={T},F={F}. Training will proceed.")

    print(f"Device: {device}")
    print(f"Data roots: {[str(d) for d in data_roots]}")
    print(f"Data: T={T} F={F} classes={num_classes} train={len(train_ds)} val={len(val_ds)}")

    model = LSTMClassifier(
        input_dim=F,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    # If bidirectional requested, we keep baseline model unidirectional for now.
    # (Requirement mentions baseline; we preserve existing model behavior.)
    if bidirectional:
        print("WARN: bidirectional=true requested, but current LSTMClassifier is unidirectional. Using unidirectional baseline.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    save_json(checkpoints_dir / "label_map.json", {"tokens": vocab, "T": T, "F": F})
    save_json(run_dir / "train_config_resolved.json", {"config_path": str(cfg_path), "config": cfg})

    best_score = -1.0
    best_epoch = 0
    no_improve = 0

    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        tr = _run_epoch(model, train_loader, device=device, optimizer=optim)
        va = _run_epoch(model, val_loader, device=device, optimizer=None)

        tr_pred, tr_y = tr.pop("pred"), tr.pop("y")
        va_pred, va_y = va.pop("pred"), va.pop("y")

        tr_f1 = macro_f1(tr_pred, tr_y, num_classes=num_classes)
        va_f1 = macro_f1(va_pred, va_y, num_classes=num_classes)
        va_cm = confusion(va_pred, va_y, num_classes=num_classes)

        row = {
            "epoch": epoch,
            "train_loss": float(tr["loss"]),
            "train_acc": float(tr["acc"]),
            "train_top5": float(tr["top5"]),
            "train_f1": float(tr_f1),
            "val_loss": float(va["loss"]),
            "val_acc": float(va["acc"]),
            "val_top5": float(va["top5"]),
            "val_f1": float(va_f1),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss={row['train_loss']:.4f} acc={row['train_acc']:.3f} f1={row['train_f1']:.3f} | "
            f"val loss={row['val_loss']:.4f} acc={row['val_acc']:.3f} f1={row['val_f1']:.3f}"
        )

        # Always save last
        last_path = checkpoints_dir / "lstm_last.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "T": T,
                "F": F,
                "vocab": vocab,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_f1": best_score,
                "config": cfg,
            },
            last_path,
        )

        # Best checkpoint by val macro-F1
        score = float(va_f1)
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            best_path = checkpoints_dir / "lstm_best.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "T": T,
                    "F": F,
                    "vocab": vocab,
                    "epoch": epoch,
                    "best_val_f1": best_score,
                    "config": cfg,
                },
                best_path,
            )
            # Save confusion matrix snapshot for best epoch
            np.save(run_dir / "confusion_best.npy", va_cm)
            _plot_confusion(va_cm, vocab, confusion_png)
            print(f"  saved best: {best_path} (val_f1={best_score:.3f})")
        else:
            no_improve += 1

        # Write logs/plots incrementally
        _write_epoch_log_csv(log_csv, history)
        _plot_curves(history, curves_png)

        if no_improve >= patience:
            print(f"Early stopping: no improvement for {patience} epochs. Best epoch={best_epoch} val_f1={best_score:.3f}")
            break

    # Final metrics dump (best confusion already saved)
    final = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "T": T,
        "F": F,
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "best_val_f1": best_score,
        "history": history,
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"Wrote metrics: {metrics_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
