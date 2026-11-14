@echo off
cd /d "%~dp0apps\web"
title QueryOpt frontend (port 5173)
if not exist "node_modules" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
echo QueryOpt web app: http://localhost:5173
echo Press Ctrl+C to stop.
call npm run dev -- --port 5173 --strictPort
pause
