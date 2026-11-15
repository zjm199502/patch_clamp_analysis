#!/usr/bin/env bash
# 不打包、直接运行（macOS / Linux）：第一次会自动建立运行环境，之后再运行即可启动。需要 Python 3.11。
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  python3.11 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e .
fi
exec .venv/bin/python -m patchclamp_gui
