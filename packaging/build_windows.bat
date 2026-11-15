@echo off
chcp 65001 >nul
REM 打包成 Windows 程序（在 Windows 上、本仓库根目录运行）
REM 结果在 dist\PatchClampAnalysis\PatchClampAnalysis.exe ，整个文件夹可以拷给同学直接用

where py >nul 2>nul && (set PY=py -3.11) || (set PY=python)
%PY% -m venv .build-venv || goto :error
call .build-venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install . pyinstaller || goto :error
python -c "import patchclamp_gui; open('base_version.txt','w').write(patchclamp_gui.__version__)" || goto :error
pyinstaller --noconfirm --clean --windowed --name PatchClampAnalysis ^
  --add-data "base_version.txt:." --collect-submodules patchclamp_gui ^
  --collect-submodules pyqtgraph --collect-data pyqtgraph ^
  --collect-submodules miniml --collect-data miniml ^
  --exclude-module numba --exclude-module llvmlite --exclude-module umap ^
  --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module qt_material ^
  --exclude-module tkinter ^
  packaging\launcher.py || goto :error
echo.
echo 打包完成：dist\PatchClampAnalysis\PatchClampAnalysis.exe
goto :eof
:error
echo 打包失败，请把上面的报错信息发给维护者。
exit /b 1
