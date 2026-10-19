param(
    [string]$RunName = 'anatomy-20261019',
    [string]$WaitForRun = 'plane-depth-20261019',
    [int]$PollSeconds = 60,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Project .venv Python is missing.' }
$runnerArgs = @('-X', 'utf8', '-m', 'experiments.run_anatomy_queue', '--run-name', $RunName, '--wait-for-run', $WaitForRun, '--poll-seconds', "$PollSeconds")
if ($DryRun) { $runnerArgs += '--dry-run' }
Push-Location $projectRoot
try { & $python @runnerArgs; $resultCode = $LASTEXITCODE }
finally { Pop-Location }
exit $resultCode
