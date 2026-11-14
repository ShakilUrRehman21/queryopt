@echo off
setlocal
cd /d "%~dp0"
echo.
echo === QueryOpt setup ===
echo.

where py >nul 2>nul
if errorlevel 1 (
  echo [x] Python launcher "py" not found. Install Python 3.12+ from python.org and tick "Add to PATH".
  goto :fail
)
where npm >nul 2>nul
if errorlevel 1 (
  echo [x] npm not found. Install Node.js 18+ from nodejs.org.
  goto :fail
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating Python virtual environment in .venv ...
  py -3 -m venv .venv || goto :fail
) else (
  echo [1/4] Python virtual environment already exists.
)

echo [2/4] Installing Python packages ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -e ".[dev,api]" || goto :fail

echo [3/4] Installing web app packages (clean install) ...
pushd apps\web
call npm ci || (popd & goto :fail)
popd

echo [4/4] Preparing configuration ...
if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo       Created .env  ^(add ANTHROPIC_API_KEY there to enable AI ideas^)
) else (
  echo       .env already exists, left unchanged.
)

echo.
echo Setup complete. Run start.bat to launch QueryOpt.
echo.
pause
exit /b 0

:fail
echo.
echo Setup did not finish. Read the message above, fix it, and run setup.bat again.
pause
exit /b 1
