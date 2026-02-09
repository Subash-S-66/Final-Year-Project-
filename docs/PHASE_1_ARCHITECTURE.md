# Phase 1 — Architecture (Summary)

Runtime pipeline:

Browser (React)
- WebRTC camera capture
- frame sampling (10–15 fps)
- JPEG/WebP encode
- WebSocket stream

Backend (FastAPI)
- decode frame
- MediaPipe landmarks extraction
- sliding window buffer
- ONNX Runtime inference (LSTM for dynamic gestures)
- smoothing + debounce
- emit token(s) back to client

Next docs:
- `docs/STREAMING_PROTOCOL.md`
- `docs/DATASET_STANDARDS.md`
- `docs/VOCABULARY_V1_50.md`
