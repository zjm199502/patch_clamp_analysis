#!/bin/bash
# Add a separately frozen PyTorch Metal runtime to a completed Apple Silicon GUI bundle.
set -euo pipefail
app="$1"
test "$(uname -m)" = arm64
test -d "$app/Contents/Resources"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
python -m venv "$work/venv"
py="$work/venv/bin/python"
"$py" -m pip install 'torch==2.10.0' 'numpy<2' h5py pyinstaller
"$py" -m PyInstaller --noconfirm --clean --onedir --console --name ClampTraceMetal \
  --specpath "$work" --workpath "$work/build" --distpath "$work/dist" \
  patchclamp_gui/core/gpu_worker.py
ditto "$work/dist/ClampTraceMetal" "$app/Contents/Resources/ClampTraceMetal"
