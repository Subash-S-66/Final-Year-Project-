# Smoke test for manual_v1 capture (does not run unless you execute this script).
# Run from repo root:  .\reports\smoke_test_capture_manual_v1.ps1

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$Py = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$Tool = Join-Path $RepoRoot 'tools\capture_samples.py'

& $Py $Tool --label HELLO --person p0 --session S1 --out data/processed/manual_v1 --count 3 --duration 2 --fps 15 --T 30
