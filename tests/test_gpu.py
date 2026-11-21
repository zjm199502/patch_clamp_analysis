import os

import numpy as np
import pytest
from scipy.signal import find_peaks

from patchclamp_gui.core import gpu


@pytest.fixture
def runtime():
    path = os.environ.get("CLAMPTRACE_TEST_CUDA_PYTHON")
    if not path:
        pytest.skip("external CUDA environment not configured")
    gpu.configure(path)
    yield
    error = gpu.failure
    gpu.configure("")
    assert not error


def test_unavailable_worker_falls_back(tmp_path):
    gpu.configure(str(tmp_path / "missing-python"))
    try:
        assert gpu.calculate("smooth", {"x": np.ones(16)}) is None
        assert gpu.failure and not gpu.enabled()
    finally:
        gpu.configure("")


def test_signal_and_pca_cuda(runtime):
    from patchclamp_gui.core import deconv, preprocess, clustering
    from unittest.mock import patch
    rng = np.random.default_rng(7)
    x = rng.normal(size=600_001)
    x[1000:1010] = x[1015:1020] = 5
    peaks = gpu.find_peaks(x.astype("float32"), 3., 30, prominence=4.)
    np.testing.assert_array_equal(peaks, find_peaks(x.astype("float32"), height=3., distance=30, prominence=4.)[0])
    waves = rng.normal(size=(300, 400))
    waves[:, :20] += np.linspace(0, 20, len(waves))[:, None]
    def compute():
        return (preprocess.rolling_baseline(x, 20_000, .8),
                deconv.deconvolve(x, 20_000),
                clustering._matrix(None, waves, "waveform", [], 3))
    with patch.object(gpu, "enabled", return_value=False):
        base, cpu, (z, _, ev) = compute()
    baseline, cuda, (zz, _, evr) = compute()
    np.testing.assert_allclose(baseline, base, atol=1e-12)
    np.testing.assert_array_equal(cuda.peaks, cpu.peaks)
    np.testing.assert_allclose(cuda.deconvolved, cpu.deconvolved, atol=1e-10)
    np.testing.assert_allclose(cuda.reconvolved, cpu.reconvolved, atol=1e-10)
    np.testing.assert_allclose(abs(z), abs(zz), atol=1e-7)
    np.testing.assert_allclose(ev, evr, atol=1e-7)


@pytest.mark.parametrize("direction", ["negative", "positive"])
def test_cuda_miniml(runtime, direction):
    from unittest.mock import patch
    from patchclamp_gui import io
    from patchclamp_gui.core.session import CellSession
    raw = io.simulate_recording(8, seed=20, rate_hz=8, direction=direction)
    def detect():
        cell = CellSession(raw)
        cell.params["direction"] = direction
        cell.run_detection()
        return cell
    with patch.object(gpu, "enabled", return_value=False):
        cpu = detect()
    cuda = detect()
    assert cuda.det.params["backend"] == "CUDA"
    np.testing.assert_allclose(cuda.det.trace, cpu.det.trace, atol=3e-5)
    np.testing.assert_array_equal(cuda.det.onsets, cpu.det.onsets)
    np.testing.assert_array_equal(cuda.events.peak_idx, cpu.events.peak_idx)
    np.testing.assert_allclose(cuda.waves, cpu.waves, atol=1e-8)
    cuda.params["threshold"] = .75
    with patch.object(gpu, "calculate", side_effect=AssertionError("unexpected inference")):
        cuda.run_detection()


def test_mid_record_failure_restarts_on_cpu(monkeypatch):
    from patchclamp_gui import io
    from patchclamp_gui.core.detection import detect_miniml

    raw = io.simulate_recording(3, seed=11, rate_hz=9)
    kwargs = dict(batch_size=32, window_ms=30)
    expected = detect_miniml(raw.data, raw.fs, **kwargs)
    calls = []
    reports = []

    def fail_second_batch(op, arrays, **args):
        calls.append(op)
        return np.zeros(len(arrays["x"]), dtype=np.float32) if len(calls) == 1 else None

    monkeypatch.setattr(gpu, "enabled", lambda: True)
    monkeypatch.setattr(gpu, "calculate", fail_second_batch)
    result = detect_miniml(raw.data, raw.fs, progress=lambda n, total: reports.append((n, total)), **kwargs)
    assert calls == ["predict", "predict"]
    assert result.params["backend"] == "TensorFlow"
    assert result.params["prediction_key"] == expected.params["prediction_key"]
    assert reports[-1][0] == reports[-1][1]
    np.testing.assert_allclose(result.window_scores, expected.window_scores, atol=1e-6)
    np.testing.assert_array_equal(result.onsets, expected.onsets)
