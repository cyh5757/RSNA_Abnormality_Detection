[CmdletBinding()]
param([string]$PythonExe)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $projectRoot '.venv-5090'
$gpuPython = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $gpuPython)) {
    if ($PythonExe) { & $PythonExe -m venv $venv }
    else { & py -3.13 -m venv $venv }
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.13, or pass -PythonExe with its executable path.' }
}
& $gpuPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed.' }
& $gpuPython -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $gpuPython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
& $gpuPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.ones(8,8,device='cuda',requires_grad=True); (x@x).sum().backward(); torch.cuda.synchronize(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA forward/backward check failed.' }
Write-Host 'Ready. Run scripts/run_local_training.ps1 -Mode check after downloading the inputs.'
