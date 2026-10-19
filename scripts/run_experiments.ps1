[CmdletBinding()]
param(
    [string[]]$Experiments = @('e03_attention', 'e04_position', 'e05_plane'),
    [string]$RunName = 'next-experiments',
    [int]$PollSeconds = 60,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Missing .venv. Create it and install requirements.txt first.'
}
$selectedExperiments = @($Experiments | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() })
$runnerArgs = @('-u', '-m', 'experiments.run_queue', '--run-name', $RunName,
                '--poll-seconds', "$PollSeconds", '--experiments') + $selectedExperiments
if ($DryRun) { $runnerArgs += '--dry-run' }
$previousEncoding = $env:PYTHONUTF8
Push-Location $projectRoot
try {
    $env:PYTHONUTF8 = '1'
    & $python @runnerArgs
    $resultCode = $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PYTHONUTF8 = $previousEncoding
}
exit $resultCode
