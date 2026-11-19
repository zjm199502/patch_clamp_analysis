import numpy as np
import pytest


def test_import_preview(monkeypatch):
    from PySide6 import QtWidgets
    from patchclamp_gui.gui.review import preview_import
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    paths = ["sIPSC/file-EPSC.abf", "sIPSC/other.abf"]
    def inspect(dialog):
        table = dialog.findChild(QtWidgets.QTableWidget)
        assert "冲突" in table.item(0, 3).text()
        assert table.cellWidget(1, 2).currentIndex() == 3
        table.selectAll()
        button = next(b for b in dialog.findChildren(QtWidgets.QPushButton) if "应用" in b.text())
        button.click()
        return True
    monkeypatch.setattr(QtWidgets.QDialog, "exec", inspect)
    groups, presets = preview_import(None, paths, {paths[0]: "group"})
    assert groups[paths[0]] == "group" and not any(presets.values())

from patchclamp_gui import io
from patchclamp_gui.core import project_files, provenance
from patchclamp_gui.core.session import CellSession


def test_manual_recovery_and_freshness(tmp_path):
    root = tmp_path / "project"
    project = project_files.create_project(root)
    cell = CellSession(io.simulate_recording(4, seed=18))
    project.cells.append(cell)
    cell.params.update(detector="deconvolution")
    assert cell.run_detection() > 0
    index = len(cell.events) // 2
    event = cell.events.iloc[index]
    peak = int(event.peak_idx)
    cell.events = cell.events.drop(index).reset_index(drop=True)
    cell.waves = np.delete(cell.waves, index, axis=0)
    project_files.save_project(root, project)
    before = provenance.report_stamp(cell)
    bounds = ((peak - round(.002*cell.rec.fs))/cell.rec.fs, (peak+round(.001*cell.rec.fs))/cell.rec.fs)
    cell.accept_manual(cell.manual_candidate(*bounds))
    with pytest.raises(ValueError):
        cell.manual_candidate(*bounds)
    assert before != provenance.report_stamp(cell)
    project_files.checkpoint(root, project, 0)
    restored = project_files.open_project(root).cells[0]
    assert restored.events.manual.sum() == 1 and len(restored.events) == len(restored.waves)
    assert np.isnan(restored.events.loc[restored.events.manual, "score"].iloc[0])
    assert provenance.status(restored) == "已完成"
    restored.params["window_ms"] += 5
    assert provenance.status(restored) == "待重算"
