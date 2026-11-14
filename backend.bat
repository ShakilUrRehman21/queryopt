@echo off
cd /d "%~dp0"
title QueryOpt backend (port 8000)
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
echo QueryOpt backend: http://127.0.0.1:8000   (API docs at /docs)
echo Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
pause
