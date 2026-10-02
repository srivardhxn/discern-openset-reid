# Discern Frontend UI Launcher
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "frontend")

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    cmd.exe /c npm install
}

Write-Host "Starting Discern Frontend UI on http://localhost:5173 ..." -ForegroundColor Cyan
cmd.exe /c npm run dev
