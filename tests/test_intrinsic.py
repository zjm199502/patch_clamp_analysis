import numpy as np
import pytest


def test_sweep_review(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import pyabf
    from PySide6 import QtWidgets
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from patchclamp_gui.gui.review import review_sweeps
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    abf = SimpleNamespace(dataRate=1000, sweepX=np.arange(200)/1000, sweepC=np.zeros(200))
    abf.setSweep = lambda index: setattr(abf, "sweepY", np.full(200, -65 + index))
    monkeypatch.setattr(pyabf, "ABF", lambda path: abf)
    def inspect(dialog):
        choice = dialog.findChild(QtWidgets.QComboBox)
        choice.setCurrentIndex(1)
        figure = dialog.findChild(FigureCanvasQTAgg).figure
        assert figure.axes[1].lines[1].get_xdata()[0] == pytest.approx(.07)
        assert figure.axes[1].lines[1].get_ydata()[0] == -64
    monkeypatch.setattr(QtWidgets.QDialog, "exec", inspect)
    rows = [dict(current_pA=index*25, rate_Hz=index, sweep=index+1, count=index, peak_ms=[20]*index) for index in (0, 1)]
    review_sweeps(None, "example.abf", dict(rows=rows, start_s=.05, duration_s=.1), "", tmp_path)

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
