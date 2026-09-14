@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0."

set PYTHONNOUSERSITE=1
set MKL_THREADING_LAYER=sequential
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1
set KMP_AFFINITY=disabled
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_MAX_THREADS=1
set TORCH_HOME=torch_hub
set HF_HUB_OFFLINE=1

echo ============================================================
echo   ADOFAI Diffusion V4
echo ============================================================

REM preflight check
if not exist "python\python.exe" (
    echo [ERROR] python\python.exe not found - extraction incomplete
    goto :fail
)
if not exist "app\web_server.py" (
    echo [ERROR] app\web_server.py not found - extraction incomplete
    goto :fail
)
if not exist "venv\Scripts\python.exe" (
    echo [WARNING] venv missing - model inference/training unavailable
)

REM start server in its own window.
REM NOTE: use "& pause" (not "&& pause") so the window STAYS OPEN even if
REM python crashes at startup - the real error will be visible.
REM cwd is already the adoai folder (set above), so no "cd /d %~dp0" needed.
REM TORCH_HOME uses a relative path to avoid the "%~dp0" trailing-backslash
REM quote-break bug inside cmd /c "..."
echo [INFO] Starting server...
start "ADOFAI-Diffusion-Server" cmd /c "set PYTHONNOUSERSITE=1 && set TORCH_HOME=torch_hub && set HF_HUB_OFFLINE=1 && python\python.exe -u app\web_server.py --port 8081 & pause"

REM wait for health (max 30s)
echo [INFO] Waiting for server (max 30s)...
set READY=0
for /l %%i in (1,1,30) do (
    timeout /t 1 >nul 2>&1
    curl -s --max-time 2 http://127.0.0.1:8081/api/health >nul 2>&1
    if !errorlevel! equ 0 (
        set READY=1
        goto :ok
    )
)

:ok
if !READY! equ 1 (
    echo [OK] Server ready: http://127.0.0.1:8081
    start "" http://127.0.0.1:8081
) else (
    echo.
    echo   [ERROR] Server did not start within 30 seconds!
    echo.
    echo   Look at the black window named "ADOFAI-Diffusion-Server".
    echo   It now STAYS OPEN showing the real error (Python traceback,
    echo   missing module, etc). Take a screenshot and open an issue.
    echo.
    echo   Common fixes:
    echo     1. Move folder to a plain English path, e.g. D:\adoai
    echo     2. No model weights? Open browser, go to Training tab
    echo     3. Port 8081 in use? Close that program and retry
    echo.
    goto :fail
)

echo ============================================================
echo   ADOFAI Diffusion ready
echo   Browser should open http://127.0.0.1:8081
echo   First use: open Training tab, select data dir, click Train All
echo   Closing this window will NOT stop the server
echo ============================================================
pause
exit /b 0

:fail
echo ============================================================
echo   Startup failed. Press any key to exit...
echo ============================================================
pause
exit /b 1
