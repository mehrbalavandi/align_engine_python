@echo off
rem align_engine - double-click to run. Uses the .venv created by setup.ps1.
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The Python environment is missing. Run setup.bat once first ^(see GUIDE.md^).
  pause
  exit /b 1
)
".venv\Scripts\python.exe" align_batch.py %*
echo.
pause
