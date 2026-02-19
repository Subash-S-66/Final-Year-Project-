from __future__ import annotations

import argparse
from pathlib import Path

import csv
import json
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from shared.sequence_preprocess import SequencePreprocessConfig, SequencePreprocessor
from training.dataset import scan_samples_root, split_train_val_by_clip
from training.metrics import accuracy, confusion, macro_f1, topk_accuracy
from training.models import LSTMClassifier
from training.torch_dataset import NpzSequenceDataset, SequenceAugmentor
from training.utils import ensure_dir, load_vocab, repo_root, save_json, set_seed


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(np.sum(e, axis=axis, keepdims=True), 1e-8)


def _compute_class_weights(
    labels: list[int],
    *,
    num_classes: int,
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((num_classes,), dtype=np.int64)
    for y in labels:
        if 0 <= int(y) < num_classes:
            counts[int(y)] += 1

    total = int(np.sum(counts))
    weights = np.zeros((num_classes,), dtype=np.float32)
    present = counts > 0
    if total <= 0 or not present.any():
        return counts, np.ones((num_classes,), dtype=np.float32)

    # Exact formula:
    #   w_c = (N / (K * n_c))^power, for n_c > 0; w_c = 0 for n_c = 0
    # then normalize mean weight over seen classes to 1.0.
    weights[present] = np.power((total / (num_classes * counts[present].astype(np.float32))), power)
    weights[present] /= np.maximum(np.mean(weights[present]), 1e-8)
    weights[~present] = 0.0
    return counts, weights


def _prediction_collapse_analysis(pred: np.ndarray, *, num_classes: int) -> dict[str, float | int | bool]:
    if pred.size == 0:
        return {
            "dominant_ratio": 0.0,
            "dominant_class": -1,
            "predicted_label_count": 0,
            "normalized_entropy": 0.0,
            "is_collapsed": False,
        }
    counts = np.bincount(pred.astype(np.int64), minlength=num_classes).astype(np.float64)
    p = counts / np.maximum(counts.sum(), 1.0)
    nz = p[p > 0]
    entropy = float(-(nz * np.log(np.maximum(nz, 1e-12))).sum())
    norm_entropy = float(entropy / np.log(max(num_classes, 2)))
    dominant_class = int(np.argmax(counts))
    dominant_ratio = float(np.max(p))
    predicted_label_count = int(np.sum(counts > 0))
    return {
        "dominant_ratio": dominant_ratio,
        "dominant_class": dominant_class,
        "predicted_label_count": predicted_label_count,
        "normalized_entropy": norm_entropy,
        "is_collapsed": bool(dominant_ratio >= 0.35 or predicted_label_count <= max(2, num_classes // 10)),
    }


def _per_label_confidence_distributions(
    probs: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    *,
    vocab: list[str],
) -> dict[str, dict[str, float | int]]:
    if probs.size == 0 or y.size == 0:
        return {}

    n = y.shape[0]
    true_conf = probs[np.arange(n), y]
    pred_conf = probs[np.arange(n), pred]

    out: dict[str, dict[str, float | int]] = {}
    for idx, token in enumerate(vocab):
        m_true = y == idx
        m_pred = pred == idx
        stats: dict[str, float | int] = {
            "true_count": int(np.sum(m_true)),
            "pred_count": int(np.sum(m_pred)),
        }
        if m_true.any():
            tc = true_conf[m_true]
            stats["true_conf_mean"] = float(np.mean(tc))
            stats["true_conf_p90"] = float(np.percentile(tc, 90))
        if m_pred.any():
            pc = pred_conf[m_pred]
            stats["pred_conf_mean"] = float(np.mean(pc))
            stats["pred_conf_p90"] = float(np.percentile(pc, 90))
        out[token] = stats
    return out


def _fit_temperature(logits: np.ndarray, y: np.ndarray, *, device: torch.device, max_iter: int = 100) -> dict[str, float]:
    if logits.size == 0 or y.size == 0:
        return {"temperature": 1.0, "nll_before": 0.0, "nll_after": 0.0}

    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        nll_before = float(criterion(logits_t, y_t).item())

    temperature = torch.ones(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        t = torch.clamp(temperature, min=1e-2, max=100.0)
        loss = criterion(logits_t / t, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    t_val = float(torch.clamp(temperature.detach(), min=1e-2, max=100.0).item())

    with torch.no_grad():
        nll_after = float(criterion(logits_t / t_val, y_t).item())

    return {"temperature": t_val, "nll_before": nll_before, "nll_after": nll_after}


def _compute_acceptance_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, Any]:
    if probs.size == 0 or y.size == 0:
        return {
            "global_conf": 0.55,
            "global_margin": 0.10,
            "class_conf": [0.55] * num_classes,
            "class_margin": [0.10] * num_classes,
            "tp_total": 0,
            "fp_total": 0,
        }

    conf = probs[np.arange(probs.shape[0]), pred]
    top2 = np.partition(probs, -2, axis=1)[:, -2] if probs.shape[1] > 1 else np.zeros_like(conf)
    margin = np.maximum(conf - top2, 0.0)

    tp = pred == y
    fp = ~tp
    tp_conf = conf[tp]
    tp_margin = margin[tp]
    fp_conf = conf[fp]
    fp_margin = margin[fp]

    if tp_conf.size > 0:
        tp_q10 = float(np.percentile(tp_conf, 10))
        tp_m_q10 = float(np.percentile(tp_margin, 10))
        tp_q50 = float(np.percentile(tp_conf, 50))
    else:
        tp_q10 = 0.55
        tp_m_q10 = 0.10
        tp_q50 = 0.70

    if fp_conf.size > 0:
        fp_q90 = float(np.percentile(fp_conf, 90))
        fp_m_q90 = float(np.percentile(fp_margin, 90))
    else:
        fp_q90 = 0.0
        fp_m_q90 = 0.0

    conf_from_tp = max(0.50, tp_q10)
    conf_from_fp = min(0.70, fp_q90 + 0.03) if fp_conf.size > 0 else 0.0
    global_conf = float(np.clip(max(conf_from_tp, conf_from_fp), 0.50, 0.70))

    margin_from_tp = max(0.10, tp_m_q10)
    margin_from_fp = min(0.25, fp_m_q90 + 0.01) if fp_margin.size > 0 else 0.0
    global_margin = float(np.clip(max(margin_from_tp, margin_from_fp), 0.10, 0.25))

    class_conf = np.full((num_classes,), global_conf, dtype=np.float32)
    class_margin = np.full((num_classes,), global_margin, dtype=np.float32)

    for c in range(num_classes):
        mask = tp & (y == c)
        if int(mask.sum()) <= 0:
            continue
        c_conf = conf[mask]
        c_margin = margin[mask]
        if c_conf.size >= 3:
            class_conf[c] = float(np.clip(np.percentile(c_conf, 10), global_conf * 0.90, min(0.80, global_conf + 0.15)))
            class_margin[c] = float(np.clip(np.percentile(c_margin, 10), global_margin * 0.85, min(0.35, global_margin + 0.12)))
        else:
            class_conf[c] = float(global_conf)
            class_margin[c] = float(global_margin)

    return {
        "global_conf": float(global_conf),
        "global_margin": float(global_margin),
        "class_conf": [float(x) for x in class_conf.tolist()],
        "class_margin": [float(x) for x in class_margin.tolist()],
        "tp_total": int(tp.sum()),
        "fp_total": int(fp.sum()),
    }


def _run_epoch(model, loader, *, device, loss_fn, optimizer=None, grad_clip_norm: float = 0.0):
    train = optimizer is not None
    model.train(train)

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
            if grad_clip_norm and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 1), dtype=np.float32)
    y_np = np.concatenate(all_y, axis=0) if all_y else np.zeros((0,), dtype=np.int64)
    pred = logits_np.argmax(axis=1) if logits_np.size else np.zeros((0,), dtype=np.int64)
    probs = _softmax_np(logits_np, axis=1) if logits_np.size else np.zeros_like(logits_np, dtype=np.float32)
    conf = probs.max(axis=1) if probs.size else np.zeros((0,), dtype=np.float32)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": accuracy(pred, y_np),
        "top3": topk_accuracy(logits_np, y_np, k=min(3, logits_np.shape[1])) if logits_np.size else 0.0,
        "pred": pred,
        "y": y_np,
        "logits": logits_np,
        "probs": probs,
        "conf": conf,
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
    min_train_per_label = int(data_cfg["split"].get("min_train_per_label", 2))
    min_val_per_label = int(data_cfg["split"].get("min_val_per_label", 1))

    hidden_dim = int(cfg["model"].get("hidden_dim", 256))
    num_layers = int(cfg["model"].get("num_layers", 2))
    dropout = float(cfg["model"].get("dropout", 0.2))
    bidirectional = bool(cfg["model"].get("bidirectional", False))

    epochs = int(cfg["train"].get("epochs", 50))
    batch_size = int(cfg["train"].get("batch_size", 16))
    lr = float(cfg["train"].get("lr", 3e-4))
    weight_decay = float(cfg["train"].get("weight_decay", 0.01))
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.05))
    class_weight_power = float(cfg["train"].get("class_weight_power", 0.7))
    grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 1.0))
    patience = int(cfg["train"].get("patience", 10))
    num_workers = int(cfg["train"].get("num_workers", 0))
    temperature_max_iter = int(cfg["train"].get("temperature_max_iter", 100))
    use_weighted_sampler = bool(cfg["train"].get("use_weighted_sampler", False))

    pre_cfg = SequencePreprocessConfig.from_dict(cfg.get("preprocess", {}))
    preprocessor = SequencePreprocessor(pre_cfg)
    augmentor = SequenceAugmentor.from_dict(cfg.get("augment", {}))

    device = _resolve_device(str(cfg.get("runtime", {}).get("device", "auto")))

    outputs = cfg.get("outputs", {})
    checkpoints_dir = ensure_dir(root / outputs.get("checkpoints_dir", "reports/checkpoints"))
    run_dir = ensure_dir(root / outputs.get("run_dir", "reports/train_lstm_run"))
    log_csv = (root / outputs.get("log_csv", "reports/train_lstm_log.csv")).resolve()
    curves_png = (root / outputs.get("curves_png", "reports/train_lstm_curves.png")).resolve()
    confusion_png = (root / outputs.get("confusion_png", "reports/train_lstm_confusion.png")).resolve()
    metrics_json = (root / outputs.get("metrics_json", "reports/train_lstm_metrics.json")).resolve()
    analysis_json = (root / outputs.get("analysis_json", "reports/train_lstm_analysis.json")).resolve()

    vocab = load_vocab(vocab_path)
    num_classes = len(vocab)

    # Scan multiple dataset roots & merge
    items = []
    for dr in data_roots:
        items.extend(scan_samples_root(dr))

    if not items:
        raise SystemExit(f"No .npz samples found under: {data_roots}")

    train_items, val_items = split_train_val_by_clip(
        items,
        val_ratio=val_ratio,
        min_train_per_label=min_train_per_label,
        min_val_per_label=min_val_per_label,
        seed=split_seed,
    )
    if not train_items or not val_items:
        raise SystemExit(f"Split resulted in train={len(train_items)} val={len(val_items)}. Need more clips.")

    # Write manifests for reproducibility
    train_manifest = run_dir / "train.txt"
    val_manifest = run_dir / "val.txt"
    train_manifest.write_text("\n".join(str(it.path) for it in train_items) + "\n", encoding="utf-8")
    val_manifest.write_text("\n".join(str(it.path) for it in val_items) + "\n", encoding="utf-8")

    train_ds = NpzSequenceDataset(train_manifest, preprocessor=preprocessor, augmentor=augmentor)
    val_ds = NpzSequenceDataset(val_manifest, preprocessor=preprocessor)

    # Infer T/F, and validate against expected T=30 F=263
    X0, _y0 = train_ds[0]
    T, F = int(X0.shape[0]), int(X0.shape[1])
    if T != 30 or F not in (263, 526):
        print(f"WARN: expected T=30 and F in [263,526] but got T={T},F={F}. Training will proceed.")

    print(f"Device: {device}")
    print(f"Data roots: {[str(d) for d in data_roots]}")
    print(f"Data: T={T} F={F} classes={num_classes} train={len(train_ds)} val={len(val_ds)}")
    print(f"Preprocess: {pre_cfg.to_dict()}")
    print(f"Augment: enabled={augmentor.enabled}")

    class_counts, class_weights = _compute_class_weights(train_ds.labels, num_classes=num_classes, power=class_weight_power)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=label_smoothing,
    )
    nonzero_counts = [int(c) for c in class_counts.tolist() if int(c) > 0]
    if nonzero_counts:
        print(
            f"Class counts: seen={len(nonzero_counts)}/{num_classes} "
            f"min={min(nonzero_counts)} max={max(nonzero_counts)} median={int(np.median(nonzero_counts))}"
        )

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

    train_sampler = None
    if use_weighted_sampler and len(train_ds) > 0:
        sample_w = np.array([class_weights[int(y)] for y in train_ds.labels], dtype=np.float64)
        if float(sample_w.sum()) > 0:
            train_sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_w, dtype=torch.double),
                num_samples=len(train_ds),
                replacement=True,
            )
            print("Sampler: weighted random sampler enabled")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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

    save_json(
        checkpoints_dir / "label_map.json",
        {"tokens": vocab, "T": T, "F": F, "base_F": 263, "preprocess": pre_cfg.to_dict()},
    )
    save_json(run_dir / "train_config_resolved.json", {"config_path": str(cfg_path), "config": cfg})

    best_score = -1.0
    best_epoch = 0
    no_improve = 0

    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        tr = _run_epoch(
            model,
            train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optim,
            grad_clip_norm=grad_clip_norm,
        )
        va = _run_epoch(model, val_loader, device=device, loss_fn=loss_fn, optimizer=None)

        tr_pred, tr_y = tr["pred"], tr["y"]
        va_pred, va_y = va["pred"], va["y"]

        tr_f1 = macro_f1(tr_pred, tr_y, num_classes=num_classes)
        va_f1 = macro_f1(va_pred, va_y, num_classes=num_classes)
        va_cm = confusion(va_pred, va_y, num_classes=num_classes)
        va_collapse = _prediction_collapse_analysis(va_pred, num_classes=num_classes)

        row = {
            "epoch": epoch,
            "train_loss": float(tr["loss"]),
            "train_acc": float(tr["acc"]),
            "train_top3": float(tr["top3"]),
            "train_f1": float(tr_f1),
            "val_loss": float(va["loss"]),
            "val_acc": float(va["acc"]),
            "val_top3": float(va["top3"]),
            "val_f1": float(va_f1),
            "val_dom_ratio": float(va_collapse["dominant_ratio"]),
            "val_pred_label_count": int(va_collapse["predicted_label_count"]),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss={row['train_loss']:.4f} acc={row['train_acc']:.3f} f1={row['train_f1']:.3f} | "
            f"val loss={row['val_loss']:.4f} acc={row['val_acc']:.3f} f1={row['val_f1']:.3f} "
            f"top3={row['val_top3']:.3f}"
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
                "preprocess": pre_cfg.to_dict(),
                "class_counts": class_counts.tolist(),
                "class_weights": class_weights.tolist(),
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
                    "preprocess": pre_cfg.to_dict(),
                    "class_counts": class_counts.tolist(),
                    "class_weights": class_weights.tolist(),
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

    best_path = checkpoints_dir / "lstm_best.pt"
    if not best_path.exists():
        raise SystemExit(f"Missing best checkpoint: {best_path}")

    # Post-hoc confidence calibration (temperature scaling) on validation logits.
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    model.to(device)
    model.eval()
    va_best = _run_epoch(model, val_loader, device=device, loss_fn=loss_fn, optimizer=None)

    cal = _fit_temperature(va_best["logits"], va_best["y"], device=device, max_iter=temperature_max_iter)
    temperature = float(cal["temperature"])

    logits_cal = va_best["logits"] / max(temperature, 1e-4)
    probs_cal = _softmax_np(logits_cal, axis=1)
    pred_cal = logits_cal.argmax(axis=1) if logits_cal.size else np.zeros((0,), dtype=np.int64)

    val_f1_cal = macro_f1(pred_cal, va_best["y"], num_classes=num_classes) if va_best["y"].size else 0.0
    val_top3_cal = topk_accuracy(logits_cal, va_best["y"], k=min(3, logits_cal.shape[1])) if logits_cal.size else 0.0
    collapse_cal = _prediction_collapse_analysis(pred_cal, num_classes=num_classes)
    acceptance = _compute_acceptance_thresholds(probs_cal, va_best["y"], pred_cal, num_classes=num_classes)

    per_label_conf = _per_label_confidence_distributions(probs_cal, va_best["y"], pred_cal, vocab=vocab)
    pred_counts = np.bincount(pred_cal.astype(np.int64), minlength=num_classes) if pred_cal.size else np.zeros((num_classes,), dtype=np.int64)
    dominance = [
        {"token": vocab[i], "count": int(c), "ratio": float(c / max(1, pred_cal.size))}
        for i, c in sorted(enumerate(pred_counts.tolist()), key=lambda kv: kv[1], reverse=True)
        if c > 0
    ]

    for path in [best_path, checkpoints_dir / "lstm_last.pt"]:
        if path.exists():
            ckpt = torch.load(path, map_location="cpu")
            ckpt["temperature"] = temperature
            ckpt["calibration"] = cal
            ckpt["preprocess"] = pre_cfg.to_dict()
            ckpt["acceptance"] = acceptance
            torch.save(ckpt, path)

    analysis_payload = {
        "class_weight_formula": "w_c = (N / (K * n_c))^power, n_c>0 else 0; then normalize mean(w_seen)=1",
        "class_weight_power": class_weight_power,
        "label_smoothing": label_smoothing,
        "temperature": temperature,
        "calibration": cal,
        "acceptance": acceptance,
        "val_collapse": collapse_cal,
        "prediction_dominance": dominance,
        "per_label_confidence": per_label_conf,
    }
    analysis_json.parent.mkdir(parents=True, exist_ok=True)
    analysis_json.write_text(json.dumps(analysis_payload, indent=2), encoding="utf-8")

    # Final metrics dump (best confusion already saved)
    final = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "T": T,
        "F": F,
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "best_val_f1": best_score,
        "best_val_top3_calibrated": float(val_top3_cal),
        "best_val_f1_calibrated": float(val_f1_cal),
        "temperature": temperature,
        "acceptance": acceptance,
        "preprocess": pre_cfg.to_dict(),
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "analysis_json": str(analysis_json),
        "history": history,
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"Wrote metrics: {metrics_json}")
    print(f"Wrote analysis: {analysis_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
