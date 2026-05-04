@echo off
cd /d "%~dp0"
title Wallie

REM --- Auto-install if needed ---
if not exist ".venv\Scripts\python.exe" (
    echo [!] First run detected. Running setup...
    echo.
    call install.bat
    if errorlevel 1 exit /b 1
    echo.
)

echo.
echo   ========================================
echo        WALLIE - Starting...
echo   ========================================
echo.
echo   Dashboard: http://127.0.0.1:8765
echo   Press Ctrl+C to stop.
echo.

REM --- Open browser after a short delay (background) ---
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8765" >nul 2>&1

REM --- Launch ---
.venv\Scripts\python.exe wallie.py --dashboard
