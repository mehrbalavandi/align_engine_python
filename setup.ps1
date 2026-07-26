<#
setup.ps1 — one-time environment setup for align_engine.

Run this once after cloning the repo onto a new machine, from a FRESH
PowerShell window (or a freshly-reopened VS Code, since its integrated
terminal inherits PATH from whenever VS Code itself was launched — see
the checks below).

Usage (from inside the align_engine folder):
    .\setup.ps1
#>

$ErrorActionPreference = "Stop"

function Test-CommandExists($name) {
    $null = Get-Command $name -ErrorAction SilentlyContinue
    return $?
}

Write-Host "== Checking prerequisites ==" -ForegroundColor Cyan

if (Test-CommandExists "python") {
    Write-Host "Python found: $(python --version 2>&1)"
} else {
    Write-Host "ERROR: Python not found on PATH." -ForegroundColor Red
    Write-Host "  Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')," -ForegroundColor Red
    Write-Host "  then close ALL VS Code / terminal windows and reopen before re-running this script." -ForegroundColor Red
    exit 1
}

if (Test-CommandExists "git") {
    Write-Host "Git found: $(git --version 2>&1)"
} else {
    Write-Host "ERROR: Git not found on PATH (needed because ctc-forced-aligner installs from GitHub)." -ForegroundColor Red
    Write-Host "  Install it from https://git-scm.com/download/win," -ForegroundColor Red
    Write-Host "  then close ALL VS Code / terminal windows and reopen before re-running this script." -ForegroundColor Red
    exit 1
}

if (Test-CommandExists "ffmpeg") {
    Write-Host "FFmpeg found: $((ffmpeg -version 2>&1 | Select-Object -First 1))"
} else {
    Write-Host "WARNING: FFmpeg not found on PATH. The aligner needs it to decode audio." -ForegroundColor Yellow
    Write-Host "  Install with:  winget install --id Gyan.FFmpeg --source winget" -ForegroundColor Yellow
    Write-Host "  (or download a build from https://www.gyan.dev/ffmpeg/builds/ and add its 'bin' folder to PATH)" -ForegroundColor Yellow
    Write-Host "  Then close ALL VS Code / terminal windows and reopen before continuing." -ForegroundColor Yellow
    Write-Host "  (Continuing setup now — you can install FFmpeg before your first real run.)" -ForegroundColor Yellow
}

Write-Host "`n== Virtual environment (.venv) ==" -ForegroundColor Cyan
if (Test-Path ".venv") {
    Write-Host ".venv already exists, skipping creation"
} else {
    python -m venv .venv
    Write-Host "Created .venv"
}

Write-Host "`n== Installing Python dependencies (torch is large — this can take a few minutes) ==" -ForegroundColor Cyan
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "`n== Done ==" -ForegroundColor Green
Write-Host "From now on, in any NEW terminal, activate the environment first:"
Write-Host "    .venv\Scripts\Activate.ps1"
Write-Host "then run, e.g.:"
Write-Host "    python align_batch.py --audio_dir audio --text_dir text --out_dir output"
