# Discern Setup Script (Windows 11 / PowerShell)
# Installs backend Python virtual environment and frontend node packages.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " DISCERN: Open-Set Person Re-ID System Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check Python
Write-Host "`n[1/4] Checking Python environment..." -ForegroundColor Yellow
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10+."
    exit 1
}
$pythonVersion = python --version
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# 2. Check Node / npm
Write-Host "`n[2/4] Checking Node.js environment..." -ForegroundColor Yellow
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "Node.js is not installed or not in PATH. Please install Node.js 18+."
    exit 1
}
$nodeVersion = node --version
Write-Host "Found Node: $nodeVersion" -ForegroundColor Green

# 3. Setup Python .venv
Write-Host "`n[3/4] Setting up Python virtual environment (.venv)..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..."
    python -m venv .venv
}
Write-Host "Installing backend dependencies from requirements.txt..."
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

# 4. Install frontend packages
Write-Host "`n[4/5] Installing frontend node dependencies..." -ForegroundColor Yellow
Push-Location (Join-Path $ScriptDir "frontend")
cmd.exe /c npm install
Pop-Location

# 5. Verify model bundle
Write-Host "`n[5/5] Verifying Discern ONNX model bundle..." -ForegroundColor Yellow
if (Test-Path "model\discern_embedder.onnx") {
    Write-Host "Model bundle verified at .\model\" -ForegroundColor Green
} else {
    Write-Warning "Model bundle not found at .\model\"
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " DISCERN Setup Completed Successfully!" -ForegroundColor Green
Write-Host " - Run backend API : .\run_api.ps1" -ForegroundColor Cyan
Write-Host " - Run frontend UI : .\run_ui.ps1" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Green
