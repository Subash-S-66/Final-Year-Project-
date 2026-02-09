from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from training.models import LSTMClassifier
from training.utils import repo_root


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ONNX export by comparing PyTorch vs ONNXRuntime outputs."
    )
    parser.add_argument("--checkpoint", default="reports/checkpoints/lstm_best.pt")
    parser.add_argument("--onnx", default="artifacts/model_lstm_v1.onnx")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for max|diff| check.",
    )
    args = parser.parse_args()

    root = repo_root()
    ckpt_path = (root / args.checkpoint).resolve()
    onnx_path = (root / args.onnx).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    T = int(ckpt["T"])
    F = int(ckpt["F"])

    model_cfg = (ckpt.get("config") or {}).get("model") or {}
    hidden = int(model_cfg.get("hidden_dim", 256))
    layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.2))
    bidirectional = bool(model_cfg.get("bidirectional", False))
    if bidirectional:
        raise ValueError("Checkpoint requests bidirectional=True, but LSTMClassifier is unidirectional.")

    torch_model = LSTMClassifier(
        input_dim=F,
        hidden_dim=hidden,
        num_layers=layers,
        num_classes=len(vocab),
        dropout=dropout,
    )
    torch_model.load_state_dict(ckpt["state_dict"])
    torch_model.eval()

    # Deterministic random test vector
    rng = np.random.default_rng(args.seed)
    x_np = rng.standard_normal(size=(1, T, F), dtype=np.float32)

    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(x_np)).cpu().numpy()

    # ONNXRuntime
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    onnx_logits = sess.run(["logits"], {"input": x_np})[0]

    if torch_logits.shape != onnx_logits.shape:
        raise RuntimeError(
            f"Shape mismatch: torch={torch_logits.shape} onnx={onnx_logits.shape}"
        )

    diff = np.abs(torch_logits - onnx_logits)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    torch_prob = _softmax(torch_logits, axis=-1)
    onnx_prob = _softmax(onnx_logits, axis=-1)

    torch_top1 = int(torch_prob.argmax(axis=-1)[0])
    onnx_top1 = int(onnx_prob.argmax(axis=-1)[0])

    print("=== ONNX Validation ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"onnx:       {onnx_path}")
    print(f"input:      (1, {T}, {F})")
    print(f"output:     (1, {len(vocab)})")
    print(f"max|diff|:  {max_abs:.6g}")
    print(f"mean|diff|: {mean_abs:.6g}")
    print(f"top1 torch: {torch_top1} ({vocab[torch_top1]})")
    print(f"top1 onnx:  {onnx_top1} ({vocab[onnx_top1]})")

    if max_abs > args.atol:
        raise SystemExit(
            f"FAILED: max|diff|={max_abs} exceeded atol={args.atol}."
        )

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
