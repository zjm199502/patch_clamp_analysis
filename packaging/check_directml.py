"""Verify miniML using the shipped Windows worker and a restricted executable search path."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from patchclamp_gui import io
from patchclamp_gui.core import gpu
from patchclamp_gui.core.session import CellSession


def main():
    exe = Path(sys.argv[1]).resolve() / "PatchClampAnalysis.exe"
    with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", str(exe)):
        worker = gpu.bundled_worker()
        assert worker and gpu.default_runtime() == gpu.BUNDLED_DML

    raw = io.simulate_recording(8, seed=20, rate_hz=8)
    cpu = CellSession(raw)
    cpu.run_detection()
    system = Path(os.environ["SystemRoot"])
    clean_path = os.pathsep.join(map(str, (system / "System32", system)))
    with patch.object(gpu, "bundled_worker", return_value=worker), patch.dict(os.environ, PATH=clean_path):
        info = json.loads(subprocess.check_output([worker, "--runtime-info"]))
        assert "DmlExecutionProvider" in info
        probe = subprocess.run([worker, "--probe"], capture_output=True, timeout=60)
        gpu.configure(gpu.BUNDLED_DML)
        try:
            candidate = CellSession(raw)
            candidate.run_detection()
            backend = candidate.det.params["backend"]
            assert backend == ("DirectML" if probe.returncode == 0 else "TensorFlow"), gpu.failure
            if probe.returncode:
                assert gpu.failure
            np.testing.assert_allclose(cpu.det.trace, candidate.det.trace, atol=3e-5, rtol=1e-5)
            np.testing.assert_array_equal(cpu.events.peak_idx, candidate.events.peak_idx)
            print(f"Verified {len(cpu.events)} peaks with {backend}. Worker: {worker.name}", flush=True)
        finally:
            gpu.configure("")


if __name__ == "__main__":
    main()
