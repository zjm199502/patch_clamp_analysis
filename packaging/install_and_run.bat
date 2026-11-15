@echo off
chcp 65001 >nul
REM 不打包、直接运行：第一次会自动建立运行环境（需要 Python 3.11，约 10 分钟），之后双击即可启动
cd /d %~dp0\..
if not exist .venv\Scripts\python.exe (
  where py >nul 2>nul && (py -3.11 -m venv .venv) || (python -m venv .venv)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -e .
)
start "" .venv\Scripts\pythonw.exe -m patchclamp_gui
