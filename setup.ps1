<#
setup.ps1 - one-time environment setup for align_engine.

Run it once after cloning the repo onto a new machine. Easiest: double-click setup.bat.
Or from PowerShell, in any folder:
    & "<path to align_engine>\setup.ps1"

Open a FRESH PowerShell / VS Code window after installing Python, Git or FFmpeg:
a terminal only sees PATH as it was when the terminal (or VS Code) was started.

IMPORTANT: keep this file pure ASCII. Windows PowerShell 5.1 reads a script
without a BOM as ANSI (cp1252), so a single non-ASCII character such as an
em dash breaks parsing ("Missing closing '}'").
#>

# Always work in the script's own folder, wherever it is started from.
Push-Location -LiteralPath $PSScriptRoot
try {

function Test-CommandExists([string]$name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Write-Fail([string[]]$lines) {
    foreach ($l in $lines) { Write-Host $l -ForegroundColor Red }
}

Write-Host "== Checking prerequisites ==" -ForegroundColor Cyan

if (Test-CommandExists "python") {
    Write-Host "Python found: $(python --version)"
} else {
    Write-Fail @(
        "ERROR: Python not found on PATH.",
        "  Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'),",
        "  then close ALL VS Code / terminal windows and reopen before re-running this script.")
    return
}

if (Test-CommandExists "git") {
    Write-Host "Git found: $(git --version)"
} else {
    Write-Fail @(
        "ERROR: Git not found on PATH (needed because ctc-forced-aligner installs from GitHub).",
        "  Install it from https://git-scm.com/download/win,",
        "  then close ALL VS Code / terminal windows and reopen before re-running this script.")
    return
}

if (Test-CommandExists "ffmpeg") {
    Write-Host "FFmpeg found: $(ffmpeg -version | Select-Object -First 1)"
} else {
    Write-Host "WARNING: FFmpeg not found on PATH. The aligner needs it to decode audio." -ForegroundColor Yellow
    Write-Host "  Install with:  winget install --id Gyan.FFmpeg --source winget" -ForegroundColor Yellow
    Write-Host "  (or download a build from https://www.gyan.dev/ffmpeg/builds/ and add its 'bin' folder to PATH)" -ForegroundColor Yellow
    Write-Host "  Then close ALL VS Code / terminal windows and reopen before your first real run." -ForegroundColor Yellow
    Write-Host "  (Setup continues now; FFmpeg is only needed when aligning.)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "== Virtual environment (.venv) ==" -ForegroundColor Cyan
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    Write-Host ".venv already exists, skipping creation"
} else {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
        Write-Fail @("ERROR: could not create .venv (python -m venv failed).")
        return
    }
    Write-Host "Created .venv"
}

Write-Host ""
Write-Host "== Installing Python dependencies (torch is large, this can take a few minutes) ==" -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Fail @("ERROR: upgrading pip failed. Check the internet connection and run setup again.")
    return
}
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Fail @(
        "ERROR: installing requirements failed (see the pip messages above).",
        "  Usually a network problem; running setup again continues where it stopped.")
    return
}

Write-Host ""
Write-Host "== Done ==" -ForegroundColor Green
Write-Host "To run it: double-click run.bat, or press F5 in VS Code (it uses .venv automatically)."
Write-Host "It asks for the Word document with the audio scripts and the audio folder, then for track numbers."
Write-Host "See GUIDE.md for the whole workflow."

} finally {
    Pop-Location
}
