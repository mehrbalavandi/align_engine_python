@echo off
rem align_engine - one-time setup. Double-click this file (runs setup.ps1 without changing any system policy).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
echo.
pause
