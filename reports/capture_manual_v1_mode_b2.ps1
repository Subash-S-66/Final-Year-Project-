# Auto-generated. Runs capture per token for 3 sessions.
# This does NOT run automatically; you must execute this .ps1 yourself.

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$Py = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$Tool = Join-Path $RepoRoot 'tools\capture_samples.py'
$OutRoot = 'data/processed/manual_v1'
$Person = 'p0'
$Camera = 0
$Fps = 15.0
$T = 30
$Duration = 2.0

Write-Host '
=== S1 ===' -ForegroundColor Cyan
Write-Host 'Output: ' $OutRoot ' Person: ' $Person
& $Py $Tool --label DOCTOR --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 8
& $Py $Tool --label HELLO --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 8
& $Py $Tool --label HERE --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 8
& $Py $Tool --label HOW --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 8
& $Py $Tool --label I_LOVE_YOU --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 8
& $Py $Tool --label NO_SIGN --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 8
& $Py $Tool --label THIRSTY --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 8
& $Py $Tool --label WHEN --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 8
& $Py $Tool --label EMERGENCY --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label FOOD --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label GO --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label HELP --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label HOME --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label HUNGRY --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label ME --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label MY --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label NO --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label NOW --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label SCHOOL --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label THAT --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label THERE --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WE --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WHAT --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WHERE --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YES --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label COME --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label FRIEND --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label PLEASE --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label STOP --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label THIS --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TIME --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TODAY --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TOILET --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TOMORROW --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WHY --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YESTERDAY --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YOU --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YOUR --person $Person --session S1 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
Write-Host '
=== S2 ===' -ForegroundColor Cyan
Write-Host 'Output: ' $OutRoot ' Person: ' $Person
& $Py $Tool --label DOCTOR --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label HELLO --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label HERE --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label HOW --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label I_LOVE_YOU --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label NO_SIGN --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label THIRSTY --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label WHEN --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label EMERGENCY --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label FOOD --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label GO --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label HELP --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label HOME --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label HUNGRY --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label ME --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label MY --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label NO --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label NOW --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label SCHOOL --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label THAT --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label THERE --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WE --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WHAT --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WHERE --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YES --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label COME --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label FRIEND --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label PLEASE --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label STOP --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label THIS --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TIME --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TODAY --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TOILET --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TOMORROW --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WHY --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YESTERDAY --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YOU --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YOUR --person $Person --session S2 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
Write-Host '
=== S3 ===' -ForegroundColor Cyan
Write-Host 'Output: ' $OutRoot ' Person: ' $Person
& $Py $Tool --label DOCTOR --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label HELLO --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label HERE --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label HOW --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label I_LOVE_YOU --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label NO_SIGN --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label THIRSTY --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label WHEN --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 6
& $Py $Tool --label EMERGENCY --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label FOOD --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label GO --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label HELP --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label HOME --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label HUNGRY --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label ME --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label MY --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label NO --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label NOW --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label SCHOOL --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label THAT --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label THERE --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WE --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WHAT --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WHERE --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YES --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label COME --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label FRIEND --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label PLEASE --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label STOP --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label THIS --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TIME --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TODAY --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TOILET --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label TOMORROW --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label WHY --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YESTERDAY --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YOU --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
& $Py $Tool --label YOUR --person $Person --session S3 --out $OutRoot --camera $Camera --fps $Fps --T $T --duration $Duration --count 5
