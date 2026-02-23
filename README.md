# Real-time Indian Sign Language → Text Translator

This repo will contain a full, production-grade pipeline:

**Webcam video (React) → WebSocket → FastAPI → MediaPipe landmarks → ONNX model inference → text output (React)**

## Project status

We are building this in phases.

- Phase 1: Architecture + dataflow + vocabulary decisions ✅
- Phase 2: Repo scaffolding + protocol + dataset standards (in progress)

## Quick start (Backend)

1) One-command run (creates `backend/.venv`, installs backend deps, starts API):

```powershell
.\run-backend.ps1
```

2) Manual setup:

- Create and activate a Python environment.
- Install dependencies:

```bash
pip install -r requirements.txt
```

3) Run the API:

```bash
uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

4) Health check:

- http://localhost:8000/health

Docs for the streaming protocol and dataset standards are in `docs/`.

## Deploy on Azure

This repo includes Azure Container Apps deployment assets for both backend and frontend.

- Backend Dockerfile: `backend/Dockerfile`
- Frontend Dockerfile: `frontend/Dockerfile`
- Deployment script: `deploy/azure/deploy-container-apps.ps1`
- Deployment guide: `deploy/azure/README.md`

Quick deploy (PowerShell, from repo root):

```powershell
.\deploy\azure\deploy-container-apps.ps1 `
  -SubscriptionId "<your-subscription-id>" `
  -ResourceGroup "isl-collage-rg" `
  -Location "eastus" `
  -AcrName "<globally-unique-acr-name>"
```
