# Training

This folder will contain:
- dataset builders (from MediaPipe landmarks)
- a static baseline (scikit-learn)
- a dynamic LSTM model (PyTorch)
- ONNX export scripts

## Install training dependencies

We keep training dependencies separate from backend dependencies.

1) (Recommended) install CPU-only PyTorch:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

2) Install the rest:

```bash
pip install -r training/requirements.txt
```

## Step 1 — Generate train/val/test splits

Splits should be **by person**.

```bash
python tools/generate_splits.py --vocab configs/vocab_v1_51.json
```

## Step 2 — Train a static baseline (optional)

```bash
python training/train_static_baseline.py --vocab configs/vocab_v1_51.json
```

## Step 3 — Train the dynamic LSTM

```bash
python training/train_lstm.py --vocab configs/vocab_v1_51.json --epochs 20
```

## Step 4 — Export to ONNX

```bash
python training/export_onnx.py --checkpoint models/lstm_v1.pt --out models/lstm_v1.onnx
```

