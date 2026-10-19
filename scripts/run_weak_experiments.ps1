[CmdletBinding()]
param(
    [string]$RunName = 'weak-labels',
    [int]$PollSeconds = 60,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$experiments = @('e06_gold_control', 'e07_weak_labels', 'e08_triplet', 'e09_protocols',
                 'e10_class_mil', 'e11_finetune', 'e12_ensemble')
& (Join-Path $PSScriptRoot 'run_experiments.ps1') -Experiments $experiments -RunName $RunName -PollSeconds $PollSeconds -DryRun:$DryRun
exit $LASTEXITCODE
