[CmdletBinding()]
param(
    [string]$RunName = 'research-campaign-20261021',
    [int]$PollSeconds = 120,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Missing .venv.' }
$runnerArgs = @('-u', '-X', 'utf8', '-m', 'experiments.run_campaign_queue',
                '--run-name', $RunName, '--poll-seconds', "$PollSeconds")
if ($DryRun) { $runnerArgs += '--dry-run' }
Push-Location $projectRoot
try {
    & $python @runnerArgs
    $resultCode = $LASTEXITCODE
}
finally { Pop-Location }
exit $resultCode
