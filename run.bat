@echo off
setlocal
cd /d %~dp0

REM 多线程限制，避免某些环境下的崩溃
set PYTHONNOUSERSITE=1
set MKL_THREADING_LAYER=sequential
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1
set KMP_AFFINITY=disabled
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_MAX_THREADS=1

REM 首次运行：自动创建虚拟环境并安装依赖
if not exist venv\Scripts\python.exe (
    echo [setup] 首次运行，正在创建虚拟环境并安装依赖（可能需要几分钟）...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

start "ADOFAI Maker 128" venv\Scripts\python.exe app\web_server.py --port 8081
timeout /t 3 >nul
start "" http://127.0.0.1:8081

echo ============================================================
echo   ADOFAI Maker 128 已启动
echo   浏览器打开: http://127.0.0.1:8081
echo.
echo   首次使用：进入"训练模型"页 -^> 选择数据目录 -^> 点"一键全训"
echo   （未训练时生成会失败，需要先得到 onset_net.pt / vae.pt / ddpm.pt）
echo ============================================================
pause
