# Phase 4 — Windowed Inference Pipeline

Real-time inference now uses a sliding window buffer to accumulate landmarks over time before running the model.

## How it works

1. **Frame arrives** → landmarks extracted (263-dim vector)
2. **Added to sliding window** → deque with max length `T=30`
3. **Window ready check** → need at least 30 frames buffered
4. **Inference hook** → stub currently returns `NO_SIGN`, will be replaced with ONNX Runtime
5. **Smoothing** → token must be stable for 3+ consecutive windows to be "committed"
6. **Emit prediction** → `is_committed=True` means stable result, `False` means live/unstable

## Settings (configurable via env vars)

- `ISL_WINDOW_SIZE` (default: 30 frames ≈ 2s at 15fps)
- `ISL_MIN_CONFIDENCE` (default: 0.7, threshold for committed predictions)

## Debug fields now include

- `window_ready`: whether the window has >= T frames
- `window_fill`: current number of frames in buffer

## Next: PHASE 5 (Model Training)

We'll train:
1. Static baseline (optional: scikit-learn on single-frame features)
2. Dynamic LSTM (PyTorch on windowed sequences)
3. Export to ONNX
4. Wire ONNX Runtime into the inference hook

Then we'll add smoothing, sentence formulation, and React frontend.
