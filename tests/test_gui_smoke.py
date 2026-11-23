import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["PATCHCLAMP_NO_DIALOGS"] = "1"
from patchclamp_gui import gui  # noqa: F401  (选择 PySide6)

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.mark.parametrize("platform, openblas", [("darwin", "1"), ("win32", "8"), ("linux", "8")])
def test_analysis_thread_policy(monkeypatch, platform, openblas):
    from types import SimpleNamespace

    from patchclamp_gui.gui import app, main_window

    monkeypatch.setattr(app, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(main_window, "sys", SimpleNamespace(platform=platform))
    names = ("OPENBLAS_NUM_THREADS", "OPENBLAS_DEFAULT_NUM_THREADS", "MKL_NUM_THREADS",
             "OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
    for key in names:
        monkeypatch.setenv(key, "23")

    app._configure_cpu_threads(8)
    assert [os.environ[key] for key in names] == [openblas, "1" if platform == "darwin" else "23", "8", "8", "8", "1"]
    assert main_window.Worker(lambda: None).stackSize() == (8 * 1024 * 1024 if platform == "darwin" else 0)


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
    from patchclamp_gui.core.project_files import open_project
    restored = open_project(w.project_root)
    assert int(restored.cells[0].mask().sum()) == n0 - 1
    import pandas as pd
    csv_path = tmp_path / "审核.csv"
    restored.export_review_summary(csv_path)
    summary = pd.read_csv(csv_path)
    assert summary.iloc[0].excluded_events == 1 and summary.iloc[0].n_events == n0 - 1
    assert int(w.current.mask().sum()) == n0 - 1
    w.fit_template(); w.run_deconv(); wait()
    assert w.deconv_res is not None and len(w.deconv_res.peaks) > 50
    w.c["scope"].setCurrentIndex(1); w.run_cluster(); wait()
    assert w.cluster_panel.fig is not None
    assert len(list((w.project_root / "results").glob("*/figure.png"))) == 1
    assert len(list((w.project_root / "results").glob("*/data.npz"))) == 2
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: QtWidgets.QMessageBox.No)
    w.clear_results("batch")
    assert all(cell.events is not None for cell in w.project.cells)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: QtWidgets.QMessageBox.Yes)
    other = w.project.cells[1]
    raw, params = w.current.rec, w.read_params()
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
    from patchclamp_gui.gui.main_window import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.activate_project(tmp_path / "pause-test", create=True)
    started, release = threading.Event(), threading.Event()
    completed = []

    def preprocess():
        started.set()
        assert release.wait(5)

    from patchclamp_gui.core.session import CellSession
    from patchclamp_gui.io import simulate_recording
    class SlowCell(CellSession):
        def run_preprocess(self):
            preprocess()

        def run_detection(self, progress=None):
            progress(1, 3)
            completed.append(True)
            return 1
    cell = SlowCell(simulate_recording(1))
    window.detect_batch([cell, cell])
    assert started.wait(5)
    window.pause_button.click()
    release.set()
    deadline = time.monotonic() + 5
    while not window.task_stage.startswith("已暂停") and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert "已暂停" in window.task_stage and "1/3" in window.task_stage
    window.cancel_button.click()
    deadline = time.monotonic() + 5
    while window.workers and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert not window.workers and not completed
    assert "已取消" in window.statusBar().currentMessage()
    window.close()


def test_project_resource_lifetime(tmp_path):
    import time
    import weakref
    import numpy as np
    from PySide6 import QtCore
    from patchclamp_gui.gui.main_window import MainWindow, Worker
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    source = tmp_path / "sample.npy"
    np.save(source, np.zeros(20000))
    cell_ref = buffer_ref = None
    for index in range(3):
        assert window.activate_project(tmp_path / str(index), create=True)
        if index:
            assert cell_ref() is None and buffer_ref() is None
        window.add_files([str(source)])
        cell_ref = weakref.ref(window.current)
        buffer_ref = weakref.ref(window.current.rec.data)
        window.run_bg(lambda cell=window.current: cell.name, lambda _: None, "test")
        deadline = time.monotonic() + 5
        while window.workers and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        assert not window.findChildren(Worker) and not window.task_timer.isActive()
    window.close()
    assert window.project_root is None and window.project_lock is None
    assert cell_ref() is None and buffer_ref() is None


def test_cell_filter_sort(tmp_path):
    import numpy as np
    from patchclamp_gui.gui.main_window import MainWindow
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.activate_project(tmp_path / "filtered", create=True)
    paths = []
    for name in ("c", "a", "b"):
        path = tmp_path / (name + ".npy")
        np.save(path, np.zeros(1000))
        paths.append(str(path))
    window.add_files(paths, {paths[0]: "one", paths[1]: "two", paths[2]: "one"})
    table = window.cell_table
    window.sort_cell_column(0)
    assert table.verticalHeader().logicalIndex(0) == 1
    window.cell_filters[1] = {"one"}
    window.update_cell_view()
    assert table.isRowHidden(1)
    table.selectAll()
    assert {cell.group for cell in window.selected_cells()} == {"one"}
    table.selectRow(2)
    table.item(2, 5).setText("edited")
    assert window.current is window.project.cells[2] and window.current.note == "edited"
    window.remove_cell()
    assert len(window.project.cells) == 2
    window.reset_cell_view()
    assert not table.isRowHidden(1) and table.verticalHeader().logicalIndex(0) == 0
    window.close()


def test_directory_columns_and_vector_export(tmp_path, monkeypatch):
    import numpy as np
    from xml.etree import ElementTree
    from patchclamp_gui.gui.main_window import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    root = tmp_path / "project"
    assert window.activate_project(root, create=True)
    source = tmp_path / "input.npy"
    np.save(source, np.arange(1000))
    window.add_files([str(source)], levels={str(source): ["WT", "positive"]})
    assert window.cell_table.horizontalHeaderItem(9).text() == "目录 2"
    window.cell_table.item(0, 9).setText("negative")
    assert window.current.folders == ["WT", "negative"]
    assert window.tabs.count() == 4 and window.current.events is None
    segment = tmp_path / "trace.csv"
    def choose_range(dialog):
        start, end = dialog.findChildren(QtWidgets.QDoubleSpinBox)
        start.setValue(10 / window.current.rec.fs)
        end.setValue(13 / window.current.rec.fs)
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(QtWidgets.QDialog, "exec", choose_range)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(segment), ""))
    window.export_trace_segment()
    import time
    while window.workers:
        app.processEvents()
        time.sleep(.01)
    assert window.last_error is None
    data = np.loadtxt(segment, delimiter=",", encoding="utf-8-sig", skiprows=1)
    np.testing.assert_array_equal(data[:, 0], [10, 11, 12])
    np.testing.assert_array_equal(data[:, 3], window.current.rec.data[10:13])
    np.testing.assert_allclose(data[:, 2], np.arange(3) / window.current.rec.fs)
    target = tmp_path / "wave.svg"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
    window.show(); app.processEvents()
    menu = window.p_trace.getViewBox().getMenu(None)
    next(action for action in menu.actions() if "矢量" in action.text()).trigger()
    assert ElementTree.parse(target).getroot().tag.endswith("svg")
    window.close()
    from patchclamp_gui.core.project_files import open_project
    assert open_project(root).cells[0].folders == ["WT", "negative"]


def test_import_preset_override_persists(tmp_path):
    import numpy as np
    from patchclamp_gui.core.detection import preset_from_path
    from patchclamp_gui.gui.main_window import MainWindow
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    root = tmp_path / "project"
    assert window.activate_project(root, create=True)
    folder = tmp_path / "sIpSc"
    folder.mkdir()
    files = [folder / "record.npy", folder / "mePSc1.npy"]
    for path in files:
        np.save(path, np.zeros(1000))
    window.add_files(files)
    assert [cell.params["direction"] for cell in window.project.cells] == ["positive", "negative"]
    assert preset_from_path(folder / "EPSC_IPSC.npy") is None
    window.apply_preset(0, "mEPSC / sEPSC")
    assert window.save_session()
    window.close()
    other = MainWindow()
    assert other.activate_project(root)
    assert other.project.cells[0].params["direction"] == "negative"
    other.close()

def test_threshold_preview_can_be_declined(tmp_path, monkeypatch):
    import time
    from patchclamp_gui.gui.main_window import MainWindow
    from patchclamp_gui.core.session import CellSession
    from patchclamp_gui.io import simulate_recording
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.activate_project(tmp_path / "preview", create=True)
    cell = CellSession(simulate_recording(4, seed=1))
    cell.run_detection()
    window.project.cells.append(cell)
    window.refresh_cells()
    window.cell_table.selectRow(0)
    window.p["threshold"].setValue(.75)
    prompts = []
    def decline(parent, title, message):
        prompts.append(message)
        return QtWidgets.QMessageBox.No
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", decline)
    original = cell.events
    window.run_detect_current()
    deadline = time.monotonic() + 60
    while window.workers and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert not window.workers and window.last_error is None
    assert len(prompts) == 1 and "新增" in prompts[0]
    assert cell.events is original and cell.params["threshold"] == .5
    window.close()
