"""Optional GPU worker, isolated from the desktop app's Qt and TensorFlow DLLs."""
import atexit
import json
import hashlib
import platform
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import numpy as np

# Fingerprints describe only the Keras model graph. Weights are read from the selected file.
GRAPHS = {
    "7215cb4f147e552a840553aee0292260199b72e6ed7314c261f4d923419fe891",
    "e79c20affe32071a8572317f31ddae8e5351f60fea4acdf6522d019620bd49b5",
    "6a2f61e6154c9b874063ed962a894683e203a43f63ba021f0368285429431416",
    "27e6b77511b206debc80d0c4355e299d366f495d7e55a0ff589e9fe18119f3ca",
}

@lru_cache(maxsize=8)
def model_graph(key):
    import h5py
    try:
        with h5py.File(key[0]) as model:
            graph = model.attrs.get("model_config", b"")
    except (ValueError, OSError):
        return None
    if isinstance(graph, str):
        graph = graph.encode()
    digest = hashlib.sha256(graph).hexdigest()
    return digest if digest in GRAPHS else None


_lock = threading.RLock()
_python = ""
_process = _directory = None
failure = ""


def enabled():
    return bool(_python) and not failure


BUNDLED_METAL = "@clamptrace/metal"


def bundled_worker():
    if sys.platform == "darwin" and getattr(sys, "frozen", False):
        path = Path(sys.executable).parent.parent / "Resources/ClampTraceMetal/ClampTraceMetal"
        if path.is_file():
            return path
    return None


def default_runtime():
    return BUNDLED_METAL if bundled_worker() else ""


def backend():
    return "MPS" if sys.platform == "darwin" and platform.machine() == "arm64" else "CUDA"


def prediction_tag():
    return backend().lower() + "-torch-1"


def supports(op):
    # Metal implements the network in float32. Preserve the original float64 signal mathematics.
    return enabled() and (backend() != "MPS" or op == "predict")


def candidates():
    paths = []
    if not getattr(sys, "frozen", False):
        paths.append(Path(sys.executable))
    registry = Path.home() / ".conda" / "environments.txt"
    if registry.is_file():
        for line in registry.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if line.strip():
                root = Path(line.strip())
                paths.append(root / ("python.exe" if os.name == "nt" else "bin/python"))
    return list(dict.fromkeys(str(p) for p in paths if p.is_file()))


def _options():
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                      if not Path(p).is_relative_to(bundle))
        if "LD_LIBRARY_PATH_ORIG" in env:
            env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH_ORIG"]
    return dict(env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _command(python, arg):
    if python == BUNDLED_METAL:
        path = bundled_worker()
        if path is None:
            raise FileNotFoundError("未找到内置 Metal 运行时")
        return [str(path), arg]
    return [python, "-I", "-u", str(Path(__file__).with_name("gpu_worker.py")), arg]


def probe(python):
    result = subprocess.run(_command(python, "--probe"), capture_output=True, text=True,
                            errors="replace", timeout=45, **_options())
    if result.returncode or not result.stdout.strip():
        raise RuntimeError((result.stderr.strip() or "GPU 探测失败").splitlines()[-1])
    return result.stdout.strip()


def close():
    global _directory, _process
    with _lock:
        if _process is not None:
            try:
                _process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                _process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                _process.kill()
                _process.wait()
            _process.stdout.close()
            _process = None
        if _directory is not None:
            _directory.cleanup()
            _directory = None


def configure(python):
    global _python, failure
    with _lock:
        close()
        _python = python
        failure = ""


def calculate(op, arrays, **args):
    global _process, _directory, failure
    with _lock:
        if not supports(op):
            return None
        try:
            if _process is None:
                _directory = tempfile.TemporaryDirectory(prefix="clamptrace_gpu_")
                atexit.unregister(close)
                atexit.register(close)
                with (Path(_directory.name) / "errors.log").open("wb") as errors:
                    _process = subprocess.Popen(_command(_python, _directory.name), stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, stderr=errors, **_options())
            directory = Path(_directory.name)
            np.savez(directory / "input.npz", **arrays)
            _process.stdin.write((json.dumps(dict(op=op, **args)) + "\n").encode())
            _process.stdin.flush()
            line = _process.stdout.readline()
            if not line:
                detail = (directory / "errors.log").read_text(errors="replace").strip()
                raise RuntimeError(f"GPU 进程已退出：{_process.poll()} {detail[-700:]}")
            reply = json.loads(line)
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error", "GPU 计算失败"))
            return np.load(directory / "result.npy", allow_pickle=False)
        except (OSError, RuntimeError, ValueError, EOFError) as exc:
            failure = str(exc)
            close()
            return None


def find_peaks(signal, height, distance, prominence=None):
    from scipy.signal import find_peaks as cpu_peaks, peak_prominences
    peaks = None
    if supports("peaks") and signal.size >= 250_000 and np.isfinite(height):
        peaks = calculate("peaks", {"x": signal}, height=height)
    if peaks is None:
        return cpu_peaks(signal, height=height, distance=distance, prominence=prominence)[0]
    # Use SciPy's own tie-breaking and prominence rules on CUDA's candidates.
    from scipy.signal._peak_finding_utils import _select_by_peak_distance
    peaks = np.ascontiguousarray(peaks, dtype=np.intp)
    peaks = peaks[_select_by_peak_distance(peaks, np.asarray(signal[peaks], dtype=float), distance)]
    if prominence is not None:
        peaks = peaks[peak_prominences(signal, peaks)[0] >= prominence]
    return peaks
