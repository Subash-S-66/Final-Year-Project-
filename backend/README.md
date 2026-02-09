# Backend (FastAPI)

## Run locally

From repo root:

```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health`
- `WS /ws/frames`

Protocol details: `docs/STREAMING_PROTOCOL.md`
