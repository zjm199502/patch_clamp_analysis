import numpy as np
import pytest

from patchclamp_gui.core.intrinsic import step_metrics


def test_metrics_exclude_test_and_rebound_spikes():
    voltage = np.full(1500, -65.)
    voltage[[80, 410, 610, 1320]] = 10
    result = step_metrics(voltage, 1000, 300, 1300, 0)
    assert result["peak_ms"] == [110, 310]
    assert result["rate_Hz"] == 2
    assert result["baseline_mV"] == -65
    assert step_metrics(voltage, 1000, 300, 1300, 20)["count"] == 0


def test_invalid_voltage_refused():
    with pytest.raises(ValueError):
        step_metrics(np.full(1500, np.nan), 1000, 300, 1300, 0)
