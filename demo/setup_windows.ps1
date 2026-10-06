# Windows env for the HyperSteer demos (CUDA torch + bitsandbytes), in demo\.venv-win.
# Separate from the repo venv: pyproject gives CPU torch on Windows, and uv.lock (used by
# the training pod) stays untouched. Idempotent; run once from anywhere:
#   powershell -ExecutionPolicy Bypass -File demo\setup_windows.ps1
# Then: .\demo\run_demo.ps1  or  .\demo\run_live.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$venv = "demo\.venv-win"
$py = "$venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    uv venv $venv --python 3.12
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
}

# Same torch as the pod's uv.lock (2.7.0+cu128), from the CUDA index. Installed first so the
# repo install below sees it as satisfied and doesn't swap in a CPU build.
uv pip install --python $py "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

# Repo package (pins transformers==4.45.1, pyvene==0.1.7) + demo/quant extras
uv pip install --python $py -e . "gradio>=4.44,<6" bitsandbytes accelerate
if ($LASTEXITCODE -ne 0) { throw "repo install failed" }

& $py -c "import torch, transformers, pyvene, bitsandbytes as bnb; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__, 'bitsandbytes', bnb.__version__)"
