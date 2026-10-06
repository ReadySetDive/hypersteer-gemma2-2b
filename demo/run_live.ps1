# Launch the HyperSteer *live* demo (bend a generation mid-stream; single user).
#
#   .\demo\run_live.ps1                     # local only: http://127.0.0.1:7861
#   .\demo\run_live.ps1 -Share              # + public https://*.gradio.live link, asks for a login
#   .\demo\run_live.ps1 -Run train_<ts>     # another downloaded run under assets\checkpoints\hf
#
# Needs the GPU to itself on an 8 GB laptop: stop the regular demo (run_demo.ps1) first.
# Ctrl+C stops it. The public link changes on every launch.
param(
    [switch]$Share,
    [string]$Run = "train_20261005_032732706478",  # final 8k-concept, 3-epoch weights
    [ValidateSet("none", "8bit", "4bit")][string]$Quant = "none",
    [ValidateSet("none", "8bit", "4bit")][string]$HyperQuant = "4bit",
    [int]$Port = 7861
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot          # code\hypersteer
$python = Join-Path $root "demo\.venv-win\Scripts\python.exe"
$runDir = Join-Path $root "assets\checkpoints\hf\$Run"

if (-not (Test-Path $python)) { throw "Missing $python - run demo\setup_windows.ps1 first." }
if (-not (Test-Path (Join-Path $runDir "train\HyperSteer_weight.safetensors"))) {
    Write-Host "Downloading $Run from Hugging Face (~5 GB, one time)..."
    Push-Location $root
    try { & $python demo\fetch_and_run.py --fetch-only --run $Run } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw "Download failed - is huggingface-cli logged in with repo access?" }
}
if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue) {
    Write-Warning "The regular demo is running on port 7860; both won't fit in 8 GB of VRAM."
}
Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

$env:PYTHONIOENCODING = "utf-8"
Remove-Item Env:DEMO_AUTH -ErrorAction SilentlyContinue
$extra = @()
if ($Share) {
    $user = Read-Host "Login username for the public link"
    $pass = Read-Host "Login password" -AsSecureString
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($pass))
    $env:DEMO_AUTH = "${user}:${plain}"
    $extra += "--share"
}

Push-Location $root
try {
    & $python -u demo\live_app.py --quant $Quant --hyper-quant $HyperQuant `
        --run-dir $runDir --port $Port @extra
} finally {
    Pop-Location
    Remove-Item Env:DEMO_AUTH -ErrorAction SilentlyContinue
}
