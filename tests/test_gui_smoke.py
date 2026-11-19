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
    w.run_detect_all()
    assert w.progress.maximum() == len(w.project.cells) and w.task_timer.isActive()
    wait()
    assert w.progress.value() == len(w.project.cells) and not w.task_timer.isActive()
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
    preset = table.cellWidget(1, 6)
    preset.setCurrentIndex(3)
    preset.activated.emit(3)
    assert window.project.cells[1].params["direction"] == window.p["direction"].currentText() == "positive"
    assert window.project.cells[0].params["direction"] == "negative"
    assert window.cell_splitter.handle(1).width() > 0
    table.selectRow(0)
    table.selectionModel().select(table.model().index(2, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    window.p["lowpass_hz"].setValue(800); window.apply_selected()
    assert [cell.params["lowpass_hz"] for cell in window.project.cells] == [800, 1000, 800]
    assert len(window.selected_cells()) == 2
    window.set_ui_points(12); window.resize(1024, 650); window.show(); app.processEvents()
    assert window.font().pointSize() == 12 and window.tabs.widget(0).count() == 2
    window.set_ui_points(10)
    assert window.save_session()
    from patchclamp_gui.core import import_metadata
    import csv
    template = tmp_path / "annotation.csv"
    window.project.cells[1].note = "测试,备注"
    import_metadata.template(template, root, window.project.cells, False)
    with template.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    rows[2][2] = "新组"
    with template.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows([rows[0], *rows[:0:-1]])
    assert import_metadata.template(template, root, window.project.cells, True) == 3
    assert window.save_session()
    assert {group for _, group in import_metadata.scan(tmp_path, root)} == {"0", "1", "2"}
    other = MainWindow()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: None)
    assert not other.activate_project(root)
    window.close()
    assert other.activate_project(root)
    assert (other.project.cells[1].group, other.project.cells[1].note) == ("新组", "测试,备注")
    assert [cell.params["lowpass_hz"] for cell in other.project.cells] == [800, 1000, 800]
    assert len({cell.name for cell in other.project.cells}) == 3
    other.close()


def test_cancel_paused_batch(tmp_path):
    import threading
    import time
    from types import SimpleNamespace
    from patchclamp_gui.gui.main_window import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.activate_project(tmp_path / "pause-test", create=True)
    started, release = threading.Event(), threading.Event()
    completed = []

    def preprocess():
        started.set()
        assert release.wait(5)

    cell = SimpleNamespace(name="test", run_preprocess=preprocess, run_detection=lambda: completed.append(True) or 1)
    window.detect_batch([cell, cell])
    assert started.wait(5)
    window.pause_button.click()
    release.set()
    deadline = time.monotonic() + 5
    while window.task_stage != "已暂停，等待继续" and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert window.task_stage == "已暂停，等待继续"
    window.cancel_button.click()
    deadline = time.monotonic() + 5
    while window.workers and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert not window.workers and completed == [True]
    assert "已取消" in window.statusBar().currentMessage()
    window.close()
