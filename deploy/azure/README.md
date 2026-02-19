# Azure Deployment (Container Apps)

This project deploys cleanly to Azure Container Apps as two services:

- `backend`: FastAPI + WebSocket + MediaPipe/ONNX inference
- `frontend`: React static build served by nginx

## Prerequisites

- Azure subscription
- Azure CLI installed (`az`)
- Logged in: `az login`
- Docker is **not** required locally (images are built with `az acr build`)

## One-command deployment (PowerShell)

From repo root:

```powershell
.\deploy\azure\deploy-container-apps.ps1 `
  -SubscriptionId "<your-subscription-id>" `
  -ResourceGroup "isl-collage-rg" `
  -Location "eastus" `
  -AcrName "<globally-unique-acr-name>"
```

Notes:

- `AcrName` must be globally unique, lowercase letters/numbers only.
- Script builds backend first, gets backend FQDN, then builds frontend with:
  - `VITE_WS_URL=wss://<backend-fqdn>/ws`
- Script prints final frontend/backend URLs at the end.

## Re-deploy after code changes

Re-run the same script. It will:

- build new images with a timestamp tag
- update existing Container Apps if they already exist

## Useful commands

Backend logs:

```powershell
az containerapp logs show --name isl-collage-backend --resource-group isl-collage-rg --follow
```

Frontend logs:

```powershell
az containerapp logs show --name isl-collage-frontend --resource-group isl-collage-rg --follow
```
