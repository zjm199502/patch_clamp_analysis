"""Run the shipped arm64 worker and compare miniML against TensorFlow on the build Mac."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import tensorflow as tf
from miniml.resources.util import get_resource_file_path

from patchclamp_gui import io
from patchclamp_gui.core import gpu
from patchclamp_gui.core.session import CellSession


def main():
    bundle = Path(sys.argv[1]).resolve()
    exe = bundle / "Contents/MacOS/PatchClampAnalysis"
    with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", str(exe)):
        worker = gpu.bundled_worker()
        assert worker and gpu.default_runtime() == gpu.BUNDLED_METAL
    info = json.loads(subprocess.check_output([worker, "--device-info"]))
    print("Packaged GPU runtime", info, flush=True)
    assert info["device"] == "mps" and info["mps_built"]
    path = get_resource_file_path("models/GC_lstm_model.h5")
    sample = np.random.default_rng(30).normal(size=(16, 600)).astype("float32")
    sample[0] = 1
    scaled = sample - sample.min(1, keepdims=True)
    scaled /= np.where(scaled.max(1, keepdims=True) == 0, 1, scaled.max(1, keepdims=True))
    reference = tf.keras.models.load_model(path, compile=False)(scaled[:, :, None], training=False)
    with tempfile.TemporaryDirectory() as work:
        data = Path(work) / "windows.npy"
        np.save(data, sample)
        check = json.loads(subprocess.check_output([worker, "--validate-model", path, data]))
    assert check["device"] == ("mps" if info["available"] else "cpu")
    np.testing.assert_allclose(check["scores"], np.asarray(reference).flatten(), rtol=1e-5, atol=3e-5)
    raw = io.simulate_recording(8, seed=20, rate_hz=8)
    cpu = CellSession(raw)
    cpu.run_detection()
    with patch.object(gpu, "bundled_worker", return_value=worker):
        gpu.configure(gpu.BUNDLED_METAL)
        try:
            candidate = CellSession(raw)
            candidate.run_detection()
            backend = candidate.det.params["backend"]
            assert backend == ("MPS" if info["available"] else "TensorFlow"), gpu.failure
            np.testing.assert_allclose(cpu.det.trace, candidate.det.trace, rtol=1e-5, atol=3e-5)
            np.testing.assert_array_equal(cpu.events.peak_idx, candidate.events.peak_idx)
            print(f"Verified {len(cpu.events)} peaks. Inference backend on this Mac: {backend}", flush=True)
        finally:
            gpu.configure("")


if __name__ == "__main__":
    main()
