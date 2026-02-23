Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $repoRoot "backend"
$venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"
$requirementsFile = Join-Path $repoRoot "requirements.txt"

if (-not (Test-Path $requirementsFile)) {
    throw "Missing requirements file: $requirementsFile"
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating backend virtual environment..."
    Push-Location $backendDir
    try {
        try {
            py -3.11 -m venv .venv
        }
        catch {
            python -m venv .venv
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host "Installing backend dependencies..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r $requirementsFile

Write-Host "Starting backend on http://0.0.0.0:8000"
& $venvPython -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
