@echo off
setlocal EnableDelayedExpansion

REM Change to script directory
cd /d "%~dp0"

REM Use Python 3.11 to create venv if missing
if not exist ".venv311" (
  echo Creating Python 3.11 virtual environment...
  py -3.11 -m venv .venv311 || goto :error
)

REM Upgrade pip (safe, quick)
".venv311\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1

REM Install requirements if not yet installed
if not exist ".venv311\.deps_installed" (
  echo Installing dependencies...
  ".venv311\Scripts\python.exe" -m pip install -r ml_breed_classifier\requirements.txt || goto :error
  > ".venv311\.deps_installed" echo ok
)

set HOST=0.0.0.0
set PORT=9000
set URL=http://127.0.0.1:%PORT%/

echo Starting server on %HOST%:%PORT% ...
REM Open browser shortly after startup in a separate shell
start "" cmd /c "timeout /t 2 >nul & start %URL%"

REM Run the server (blocking)
".venv311\Scripts\python.exe" -m uvicorn ml_breed_classifier.backend.app:app --host %HOST% --port %PORT% --log-level info

goto :eof

:error
echo.
echo Failed to set up or start the server. See messages above.
exit /b 1
