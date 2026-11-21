param([string]$BasePython = 'python')
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    $build = Join-Path $repo 'build\directml'
    $python = Join-Path $build 'env\Scripts\python.exe'
    if (!(Test-Path -LiteralPath $python)) {
        & $BasePython -m venv (Join-Path $build 'env')
        if ($LASTEXITCODE) { throw 'Creating DirectML build environment failed' }
    }
    & $python -m pip install 'onnx==1.17.0' 'onnxruntime-directml==1.23.0' 'numpy<2' h5py pyinstaller
    if ($LASTEXITCODE) { throw 'Installing DirectML build dependencies failed' }
    & $python -m PyInstaller --noconfirm --clean --onedir --console --name ClampTraceDirectML `
        --specpath $build --workpath "$build\work" --distpath 'dist\PatchClampAnalysis\_internal' `
        --collect-binaries onnxruntime --collect-data onnxruntime --copy-metadata onnxruntime-directml `
        --exclude-module onnx.reference --exclude-module sympy packaging/directml_worker.py
    if ($LASTEXITCODE) { throw 'Freezing DirectML worker failed' }
} finally {
    Pop-Location
}
