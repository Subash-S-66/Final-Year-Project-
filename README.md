# Real-time Indian Sign Language → Text Translator

This repo will contain a full, production-grade pipeline:

**Webcam video (React) → WebSocket → FastAPI → MediaPipe landmarks → ONNX model inference → text output (React)**

## Project status

We are building this in phases.

- Phase 1: Architecture + dataflow + vocabulary decisions ✅
- Phase 2: Repo scaffolding + protocol + dataset standards (in progress)

## Quick start (Backend)

1) Create and activate a Python environment.

2) Install dependencies:

```bash
pip install -r backend/requirements.txt
```

3) Run the API:

```bash
uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

4) Health check:

- http://localhost:8000/health

Docs for the streaming protocol and dataset standards are in `docs/`.
