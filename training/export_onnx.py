from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.models import LSTMClassifier
from training.utils import repo_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Export trained LSTM checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="reports/checkpoints/lstm_best.pt")
    parser.add_argument("--out", default="artifacts/model_lstm_v1.onnx")
    parser.add_argument(
        "--torchscript-out",
        default="artifacts/model_lstm_v1.torchscript.pt",
        help="Optional TorchScript export path. Set to empty string to skip.",
    )
    parser.add_argument(
        "--meta-out",
        default="artifacts/model_lstm_v1.meta.json",
        help="Writes export metadata (T/F/vocab/config/checkpoint).",
    )
    args = parser.parse_args()

    root = repo_root()
    ckpt_path = (root / args.checkpoint).resolve()
    out_path = (root / args.out).resolve()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    T = int(ckpt["T"])
    F = int(ckpt["F"])

    # Phase 7 checkpoints store the model hyperparams under ckpt["config"]["model"].
    model_cfg = (ckpt.get("config") or {}).get("model") or {}
    hidden = int(model_cfg.get("hidden_dim", 256))
    layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.2))
    bidirectional = bool(model_cfg.get("bidirectional", False))
    if bidirectional:
        raise ValueError("Checkpoint requests bidirectional=True, but LSTMClassifier is unidirectional.")

    model = LSTMClassifier(
        input_dim=F,
        hidden_dim=hidden,
        num_layers=layers,
        num_classes=len(vocab),
        dropout=dropout,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.zeros((1, T, F), dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    # Optional TorchScript export (useful for debugging / fallback).
    if str(args.torchscript_out).strip():
        ts_path = (root / args.torchscript_out).resolve()
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        scripted = torch.jit.trace(model, dummy)
        scripted.save(str(ts_path))
        print(f"Exported TorchScript: {ts_path}")

    # Export metadata to help the backend validate contract.
    if str(args.meta_out).strip():
        meta_path = (root / args.meta_out).resolve()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "checkpoint": str(ckpt_path),
            "onnx": str(out_path),
            "T": T,
            "F": F,
            "num_classes": len(vocab),
            "vocab": vocab,
            "config": ckpt.get("config", {}),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote metadata: {meta_path}")

    print(f"Exported ONNX: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
