@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Create .venv with Python 3.12 first. See README.md.
  exit /b 1
)
".venv\Scripts\python.exe" train.py %*
exit /b %errorlevel%
