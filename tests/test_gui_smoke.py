import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["PATCHCLAMP_NO_DIALOGS"] = "1"
from patchclamp_gui import gui  # noqa: F401  (选择 PySide6)

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


def test_gui_full_workflow(monkeypatch, tmp_path):
    import pyqtgraph as pg

    from patchclamp_gui.gui.main_window import MainWindow

    pg.setConfigOptions(background="w", foreground="k")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.load_demo()
    assert not w.project.cells and w.project_root is None
    assert w.activate_project(tmp_path / "项目", create=True)

    def wait():
        import time
        while w.workers:
            app.processEvents(); time.sleep(0.02)
        app.processEvents()
        assert w.last_error is None, w.last_error

    w.load_demo()
    w.project.cells = w.project.cells[:2] + w.project.cells[4:6]
    w.refresh_cells()
    w.cell_table.selectAll()
    w.p["detector"].setCurrentText("去卷积")
    w.apply_selected()
    w.run_detect_all(); wait()
    assert all(c.events is not None and len(c.events) > 50 for c in w.project.cells)
    w.cell_table.selectRow(0)
    n0 = int(w.current.mask().sum())
    w.ev_table.selectRow(0); w.toggle_exclude()
    assert int(w.current.mask().sum()) == n0 - 1
    w.fit_template(); w.run_deconv(); wait()
    assert w.deconv_res is not None and len(w.deconv_res.peaks) > 50
    w.c["scope"].setCurrentIndex(1); w.run_cluster(); wait()
    assert w.cluster_panel.fig is not None
    for i in range(len(w.ANALYSES)):
        w.an.setCurrentIndex(i); w.run_analysis(); wait()
        assert w.an_panel.fig is not None and w.an_panel.text.toPlainText()
    assert len(list((w.project_root / "results").glob("*/figure.png"))) == len(w.ANALYSES) + 1
    assert len(list((w.project_root / "results").glob("*/data.npz"))) == 2
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: QtWidgets.QMessageBox.No)
    w.clear_results("batch")
    assert all(cell.events is not None for cell in w.project.cells)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: QtWidgets.QMessageBox.Yes)
    other = w.project.cells[1]
    raw, params = w.current.rec, w.read_params()
    w.clear_results("analysis")
    assert w.an_panel.fig is None and not w.an_panel.text.toPlainText()
    w.clear_results("deconv")
    assert w.deconv_res is None and not w.deconv_info.text()
    w.clear_results("cluster")
    assert all((cell.events.cluster == -1).all() for cell in w.project.cells)
    assert int(w.current.mask().sum()) == n0 - 1
    w.clear_results("review")
    assert int(w.current.mask().sum()) == n0
    w.clear_results("detect")
    assert w.current.events is None and w.current.processed is not None and other.events is not None
    assert w.ev_table.rowCount() == 0 and not w.ev_count.text()
    w.clear_results("preprocess")
    assert w.current.processed is None and w.current.rec is raw and w.read_params() == params
    w.run_detect_current(); wait()
    assert int(w.current.mask().sum()) == n0
    w.clear_results("batch")
    assert all(cell.events is None and cell.processed is not None for cell in w.project.cells)
    w.close()


def test_import_subset_and_reopen(tmp_path, monkeypatch):
    import numpy as np
    from PySide6 import QtCore
    from patchclamp_gui.gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    root = tmp_path / "subset"
    assert window.activate_project(root, create=True)
    paths = []
    for index in range(3):
        folder = tmp_path / str(index); folder.mkdir()
        path = folder / "cell.npy"; np.save(path, np.arange(1000))
        paths.append(str(path))
    window.add_files(paths)
    assert len({cell.name for cell in window.project.cells}) == 3
    table = window.cell_table
    table.selectRow(0)
    table.selectionModel().select(table.model().index(2, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    window.p["lowpass_hz"].setValue(800); window.apply_selected()
    assert [cell.params["lowpass_hz"] for cell in window.project.cells] == [800, 1000, 800]
    assert len(window.selected_cells()) == 2
    window.set_ui_points(12); window.resize(1024, 650); window.show(); app.processEvents()
    assert window.font().pointSize() == 12 and window.tabs.widget(0).count() == 2
    window.set_ui_points(10)
    assert window.save_session()
    other = MainWindow()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: None)
    assert not other.activate_project(root)
    window.close()
    assert other.activate_project(root)
    assert [cell.params["lowpass_hz"] for cell in other.project.cells] == [800, 1000, 800]
    assert len({cell.name for cell in other.project.cells}) == 3
    other.close()
