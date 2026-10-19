[CmdletBinding()]
param(
    [string]$RunName = 'spatial',
    [string]$WaitForRun = 'weak-labels',
    [int]$PollSeconds = 60,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Missing .venv. Install requirements.txt first.' }
$runnerArgs = @('-u', '-X', 'utf8', '-m', 'experiments.run_spatial_queue',
                '--run-name', $RunName, '--wait-for-run', $WaitForRun, '--poll-seconds', "$PollSeconds")
if ($DryRun) { $runnerArgs += '--dry-run' }
Push-Location $projectRoot
try {
    & $python @runnerArgs
    $resultCode = $LASTEXITCODE
}
finally { Pop-Location }
exit $resultCode
