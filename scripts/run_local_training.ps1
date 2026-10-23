[CmdletBinding()]
param(
    [ValidateSet('check', 'smoke', 'train')][string]$Mode = 'check',
    [string]$DataDir,
    [string]$Labels,
    [string]$ParentRun,
    [string]$Output,
    [switch]$EarlyStopping,
    [switch]$PublicFinetuning
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$gpuPython = Join-Path $projectRoot '.venv-5090\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $gpuPython)) { throw 'Run scripts/setup_local_gpu.ps1 first.' }
$trainArgs = @('-u', '-X', 'utf8', '-m', 'scripts.train_local', '--mode', $Mode)
if ($EarlyStopping) { $trainArgs += '--early-stopping' }
if ($PublicFinetuning) { $trainArgs += '--public-finetuning' }
if ($DataDir) { $trainArgs += @('--data-dir', $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DataDir)) }
if ($Labels) { $trainArgs += @('--labels', $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Labels)) }
if ($ParentRun) { $trainArgs += @('--parent-run', $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ParentRun)) }
if ($Output) { $trainArgs += @('--output', $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)) }
Push-Location $projectRoot
try {
    & $gpuPython @trainArgs
    $resultCode = $LASTEXITCODE
}
finally { Pop-Location }
exit $resultCode
