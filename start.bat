@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" call setup.bat
start "QueryOpt backend" cmd /k "%~dp0backend.bat"
start "QueryOpt frontend" cmd /k "%~dp0frontend.bat"
echo Starting... the browser will open in a few seconds.
timeout /t 6 /nobreak >nul
start "" http://localhost:5173
