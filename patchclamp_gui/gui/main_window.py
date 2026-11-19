"""主窗口：文件列表 + 五个功能页（记录与检测 / 事件审核 / 去卷积平滑 / 动力学聚类 / 分析与统计）。"""
from __future__ import annotations

import os
import csv
import gc
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__, io
from ..core import clustering, deconv, detection, plots, project_files
from ..core.features import FEATURES
from ..core import import_metadata
from ..core.cpu_budget import available, budget
from ..core.session import DEFAULT_PARAMS, CellSession, Project
from .update_ui import UpdateManager

Qt = QtCore.Qt
CLUSTER_COLORS = [QtGui.QColor(c) for c in plots.PALETTE]


def wait_for_io(pool, fn, *args):
    future = pool.submit(fn, *args)
    loop = QtCore.QEventLoop()
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: loop.quit() if future.done() else None)
    timer.start(40)
    if not future.done():
        loop.exec()
    timer.stop()
    return future.result()


class Worker(QtCore.QThread):
    advanced = QtCore.Signal(int, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, parent=None, tracked=False):
        super().__init__(parent)
        self.fn = fn
        self.tracked = tracked

    def run(self):
        try:
            self.done.emit(self.fn(self.advanced.emit) if self.tracked else self.fn())
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.fn = None


def _spin(val, lo, hi, step=1.0, dec=2, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi); s.setDecimals(dec); s.setSingleStep(step); s.setValue(val)
    if suffix:
        s.setSuffix(" " + suffix)
    return s


class ZoomCanvas(FigureCanvasQTAgg):
    """按比例缩放显示的图：窗口变小时整体缩小（像看 PDF），而不是把子图挤在一起造成文字重叠。
    导出时仍使用原始尺寸（英寸）与排版。"""

    MIN_DPI = 55

    def __init__(self, fig):
        self._inches = fig.get_size_inches().copy()
        super().__init__(fig)
        self.setMinimumSize(int(self._inches[0] * self.MIN_DPI), int(self._inches[1] * self.MIN_DPI))

    def resizeEvent(self, event):
        w, h = event.size().width(), event.size().height()
        if w > 10 and h > 10:
            dpi = max(min(w / self._inches[0], h / self._inches[1]), self.MIN_DPI)
            self.figure._original_dpi = dpi
            self.figure.dpi = dpi * self.device_pixel_ratio
        super().resizeEvent(event)


class FigurePanel(QtWidgets.QWidget):
    """matplotlib 图 + 文字结果 + 导出按钮。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = None
        self.canvas = None
        self.export_folder = Path.cwd()
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.split = QtWidgets.QSplitter(Qt.Vertical)
        self.holder = QtWidgets.QWidget(); self.hl = QtWidgets.QVBoxLayout(self.holder)
        self.hl.setContentsMargins(0, 0, 0, 0)
        self.scroll = QtWidgets.QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame); self.scroll.setWidget(self.holder)
        self.text = QtWidgets.QPlainTextEdit(); self.text.setReadOnly(True)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)   # 表格类结果按列对齐
        mono.setPointSize(QtWidgets.QApplication.font().pointSize())
        self.text.setFont(mono)
        self.split.addWidget(self.scroll); self.split.addWidget(self.text)
        self.split.setStretchFactor(0, 3); self.split.setStretchFactor(1, 1)
        lay.addWidget(self.split)
        row = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("导出图（PDF/SVG/PNG）"); b1.clicked.connect(self.export_fig)
        b2 = QtWidgets.QPushButton("复制结果文字"); b2.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(self.text.toPlainText()))
        row.addStretch(1); row.addWidget(b2); row.addWidget(b1)
        lay.addLayout(row)

    def clear_result(self):
        if self.canvas is not None:
            self.hl.removeWidget(self.canvas); self.canvas.setParent(None); self.canvas.deleteLater()
        self.canvas = self.fig = None
        self.text.clear()

    def show_result(self, fig, text=""):
        self.clear_result()
        self.fig = fig
        self.canvas = ZoomCanvas(fig)
        self.hl.addWidget(self.canvas)
        if self.split.sizes()[0] < 300:
            self.split.setSizes([640, 240])
        self.canvas.draw_idle()
        self.text.setPlainText(text)

    def export_fig(self):
        if self.fig is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出图", str(self.export_folder / "figure.png"), "PNG 600 dpi (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF 600 dpi (*.tiff)")
        if path:
            dpi, orig = self.fig.dpi, getattr(self.fig, "_original_dpi", self.fig.dpi)
            size = self.fig.get_size_inches().copy()
            self.fig.set_size_inches(self.canvas._inches, forward=False)
            self.fig.savefig(path, dpi=600)
            self.fig.set_size_inches(size, forward=False)
            self.fig._original_dpi, self.fig.dpi = orig, dpi
            self.canvas.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"膜迹 ClampTrace v{__version__} — m/sEPSC · m/sIPSC")
        screen = self.screen().availableGeometry()
        self.resize(min(1500, screen.width()), min(900, screen.height()))
        self.project = Project()
        self.project_root = None
        self.project_lock = None
        self.dirty = False
        self.project_actions = []
        self.current: CellSession | None = None
        self.workers: list[Worker] = []
        self.deconv_res = None
        self.qc_result_cell = None
        self.last_error = None
        self.settings = QtCore.QSettings("PatchClampAnalysis", "PatchClampAnalysis")
        self._build_menu()
        self._build_ui()
        self.set_ui_points(int(self.settings.value("ui/points", 10)))
        self.workspace.setCurrentIndex(0)
        self.cell_dock.hide()
        self.statusBar().showMessage("请先新建项目文件夹，或打开已有项目。")

    # ================================================================ 界面
    def _build_menu(self):
        m = self.menuBar().addMenu("文件")
        m.addAction("新建项目文件夹…", self.new_project, "Ctrl+N")
        m.addAction("打开已有项目…", self.choose_project, "Ctrl+Shift+O")
        self.project_actions.append(m.addAction("保存项目", self.save_session, "Ctrl+S"))
        self.project_actions.append(m.addAction("打开项目结果文件夹", self.show_results_folder))
        m.addSeparator()
        self.project_actions.append(m.addAction("递归添加文件夹…", self.open_folder))
        self.project_actions.append(m.addAction("导出分组备注 CSV 模板…", lambda: self.annotation_csv(False)))
        self.project_actions.append(m.addAction("导入分组备注 CSV…", lambda: self.annotation_csv(True)))
        for text, fn, key in (("打开记录（ABF / H5）…", self.open_files, "Ctrl+O"), ("载入模拟演示数据", self.load_demo, None),
                              (None, None, None), ("导出结果表（事件 + 细胞汇总 CSV）…", self.export_tables, "Ctrl+E"),
                              (None, None, None), ("退出", self.close, None)):
            if text is None:
                m.addSeparator(); continue
            act = QtGui.QAction(text, self); act.triggered.connect(lambda _=False, f=fn: f())
            if key:
                act.setShortcut(key)
            m.addAction(act)
            if fn != self.close:
                self.project_actions.append(act)
        intrinsic_menu = self.menuBar().addMenu("内在兴奋性")
        self.project_actions.append(intrinsic_menu.addAction("分析电流阶跃 ABF…", self.analyze_intrinsic))
        for action in self.project_actions:
            action.setEnabled(False)
        view = self.menuBar().addMenu("视图")
        performance = self.menuBar().addMenu("性能")
        performance.addAction("分析 CPU 核心数…", self.choose_cpu_count)
        view.addAction("放大界面文字", lambda: self.set_ui_points(self.font().pointSize() + 1), "Ctrl+=")
        view.addAction("缩小界面文字", lambda: self.set_ui_points(self.font().pointSize() - 1), "Ctrl+-")
        view.addAction("恢复默认字号", lambda: self.set_ui_points(10), "Ctrl+0")
        self.updates = UpdateManager(self)      # “帮助”菜单：检查更新 / 自动检查 / 关于

    def analyze_intrinsic(self):
        if not self.project_root or self.workers:
            return
        from ..core.intrinsic import analyze_abf
        from matplotlib.figure import Figure
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择电流钳阶跃记录", "", "ABF (*.abf)")
        if not filename:
            return
        threshold, ok = QtWidgets.QInputDialog.getDouble(self, "动作电位检测", "最低峰电压 (mV)，不是起始阈电位", 0, -60, 60, 1)
        if not ok:
            return
        try:
            result = analyze_abf(filename, threshold)
            rows = result["rows"]
            fig = Figure()
            axis = fig.subplots()
            axis.plot([row["current_pA"] for row in rows], [row["rate_Hz"] for row in rows], "o")
            axis.set(xlabel="Injected current (pA)", ylabel="Firing rate (Hz)", title="F-I")
            raw = Path(filename).resolve()
            if not raw.is_relative_to(self.project_root / "raw"):
                raw = project_files.copy_raw(self.project_root, filename)
            result.update(analysis="内在兴奋性", raw=raw.relative_to(self.project_root).as_posix())
            minimum = result["lowest_spiking_positive_pA"]
            summary = f"最低诱发放电的正阶跃：{minimum} pA" if minimum is not None else "正阶跃未检出放电"
            lines = ["Sweep\t电流(pA)\tAP数\t频率(Hz)\t基线(mV)"]
            lines.extend(f"{r['sweep']}\t{r['current_pA']:g}\t{r['count']}\t{r['rate_Hz']:.2f}\t{r['baseline_mV']:.2f}" for r in rows)
            report = summary + "\n" + "\n".join(lines)
            folder = project_files.archive_result(self.project_root, fig, report, result)
            QtWidgets.QMessageBox.information(self, "内在兴奋性", report + f"\n已保存：{folder}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "分析未完成", str(exc))

    def _build_ui(self):
        # ---- 左侧：文件/细胞列表 ----
        dock = QtWidgets.QDockWidget("细胞列表（可直接编辑组别与动物编号）", self)
        self.cell_dock = dock
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        self.cell_table = QtWidgets.QTableWidget(0, 7)
        self.cell_table.setHorizontalHeaderLabels(["文件", "组别", "动物", "事件数", "频率 Hz", "备注", "事件类型"])
        self.cell_table.horizontalHeader().moveSection(6, 1)
        self.cell_table.setColumnWidth(6, 180)
        self.cell_table.horizontalHeader().setMinimumSectionSize(60)
        self.cell_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.cell_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.cell_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.cell_table.currentCellChanged.connect(self.on_cell_selected)
        self.cell_table.itemChanged.connect(self.on_cell_edited)
        self.cell_filters = {}
        self.cell_sort = None
        header = self.cell_table.horizontalHeader()
        header.setSectionsClickable(True)
        header.sectionClicked.connect(self.sort_cell_column)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.cell_header_menu)
        header.setToolTip("点击列名升序/降序；右键按值筛选")
        toolbar = QtWidgets.QHBoxLayout()
        self.filter_summary = QtWidgets.QLabel("点击列名排序，右键筛选")
        toolbar.addWidget(self.filter_summary, 1)
        reset = QtWidgets.QPushButton("重置筛选/排序")
        reset.clicked.connect(self.reset_cell_view)
        toolbar.addWidget(reset)
        v.addLayout(toolbar)
        v.addWidget(self.cell_table)
        row = QtWidgets.QHBoxLayout()
        for label, fn in (("添加文件", self.open_files), ("添加文件夹", self.open_folder), ("移除", self.remove_cell)):
            b = QtWidgets.QPushButton(label); b.clicked.connect(fn); row.addWidget(b)
        v.addLayout(row)
        dock.setWidget(w); dock.setMinimumWidth(220)
        hh = self.cell_table.horizontalHeader()
        for j in range(1, 5):
            hh.setSectionResizeMode(j, QtWidgets.QHeaderView.ResizeToContents)
        self.cell_table.verticalHeader().setVisible(False)
        self.tabs = QtWidgets.QTabWidget()
        self.cell_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.cell_splitter.setHandleWidth(9)
        self.cell_splitter.setChildrenCollapsible(False)
        self.cell_splitter.addWidget(dock)
        self.cell_splitter.addWidget(self.tabs)
        self.cell_splitter.setStretchFactor(1, 1)
        self.cell_splitter.setSizes([430, 850])
        self.cell_splitter.handle(1).setToolTip("左右拖动调整细胞列表宽度")
        self.workspace = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.workspace)
        welcome = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(welcome)
        layout.addStretch(); layout.addWidget(QtWidgets.QLabel("先选择项目，再导入记录。原始文件和分析结果将保存在项目文件夹中。"), 0, Qt.AlignCenter)
        for label, callback in (("新建项目文件夹…", self.new_project), ("打开已有项目…", self.choose_project)):
            button = QtWidgets.QPushButton(label); button.clicked.connect(callback); layout.addWidget(button, 0, Qt.AlignCenter)
        layout.addStretch(); self.workspace.addWidget(welcome); self.workspace.addWidget(self.cell_splitter)
        self.tabs.addTab(self._tab_detect(), "① 记录与检测")
        self.tabs.addTab(self._tab_events(), "② 事件审核")
        self.tabs.addTab(self._tab_deconv(), "③ 去卷积平滑")
        self.tabs.addTab(self._tab_cluster(), "④ 动力学聚类")
        self.tabs.addTab(self._tab_analysis(), "⑤ 分析与统计")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMinimumHeight(26)
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress, 1)
        self.batch_condition = threading.Condition()
        self.batch_paused = self.batch_cancelled = False
        self.pause_button = QtWidgets.QPushButton("暂停")
        self.cancel_button = QtWidgets.QPushButton("取消批处理")
        self.pause_button.clicked.connect(self.toggle_batch_pause)
        self.cancel_button.clicked.connect(self.cancel_batch)
        for button in (self.pause_button, self.cancel_button):
            self.statusBar().addPermanentWidget(button)
            button.hide()
        self.task_clock = QtCore.QElapsedTimer()
        self.task_timer = QtCore.QTimer(self)
        self.task_timer.timeout.connect(self.update_task_status)

    def _tab_detect(self):
        form_w = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(form_w)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.p = {}
        self.p["direction"] = QtWidgets.QComboBox(); self.p["direction"].addItems(["negative", "positive"])
        form.addRow("事件方向", self.p["direction"])
        self.p["window_ms"] = _spin(30, 2, 1000, 5, 1, "ms"); form.addRow("事件窗口", self.p["window_ms"])
        self.p["tau_rise_ms"] = _spin(0.4, 0.02, 50, 0.1, 2, "ms"); form.addRow("模板 τ上升", self.p["tau_rise_ms"])
        self.p["tau_decay_ms"] = _spin(4, 0.1, 500, 1, 2, "ms"); form.addRow("模板 τ衰减", self.p["tau_decay_ms"])
        form.addRow(QtWidgets.QLabel("<b>预处理</b>"))
        self.p["lowpass_hz"] = _spin(DEFAULT_PARAMS["lowpass_hz"], 0, 20000, 100, 0, "Hz"); form.addRow("低通（0=关）", self.p["lowpass_hz"])
        self.p["highpass_hz"] = _spin(0, 0, 100, 0.5, 1, "Hz"); form.addRow("高通（0=关）", self.p["highpass_hz"])
        self.p["notch_hz"] = _spin(DEFAULT_PARAMS["notch_hz"], 0, 60, 1, 0, "Hz"); form.addRow("工频陷波（0/50/60）", self.p["notch_hz"])
        self.p["detrend_s"] = _spin(1, 0, 30, 0.5, 1, "s"); form.addRow("基线去漂移窗口（0=关）", self.p["detrend_s"])
        form.addRow(QtWidgets.QLabel("<b>检测</b>"))
        self.p["detector"] = QtWidgets.QComboBox(); self.p["detector"].addItems(["miniML", "去卷积"])
        form.addRow("检测器", self.p["detector"])
        self.p["model"] = QtWidgets.QComboBox(); self.p["model"].setEditable(True)
        self.p["model"].addItems(detection.miniml_models() or ["GC_lstm_model.h5"])
        form.addRow("miniML 模型", self.p["model"])
        self.p["threshold"] = _spin(0.5, 0.05, 0.99, 0.05, 2); form.addRow("miniML 阈值", self.p["threshold"])
        self.p["deconv_threshold_sd"] = _spin(4, 1, 20, 0.5, 1, "SD"); form.addRow("去卷积阈值", self.p["deconv_threshold_sd"])
        self.p["min_amplitude"] = _spin(0, 0, 1000, 1, 1, "pA"); form.addRow("最小幅度", self.p["min_amplitude"])
        ok, why = detection.miniml_available()
        self.miniml_ok = ok
        if not ok:
            self.p["detector"].setCurrentText("去卷积")
            lab = QtWidgets.QLabel("未检测到 miniML/TensorFlow，暂用去卷积检测器。\n安装：pip install miniml-detect（Python 3.11）")
            lab.setStyleSheet("color:#b00"); lab.setWordWrap(True); lab.setToolTip(why); form.addRow(lab)
        for label, fn in (("预览预处理", self.run_preprocess), ("检测当前细胞", self.run_detect_current),
                          ("应用当前参数到选中记录", self.apply_selected), ("按各自参数检测选中记录", self.run_detect_selected),
                          ("按各自参数检测全部细胞", self.run_detect_all)):
            b = QtWidgets.QPushButton(label); b.clicked.connect(fn); form.addRow(b)
        for label, action in (("清除当前细胞预处理及检测", "preprocess"), ("清除当前细胞检测", "detect"),
                              ("清除全部细胞检测…", "batch")):
            form.addRow(self.clear_button(label, action))
        self.info = QtWidgets.QLabel(""); self.info.setWordWrap(True); form.addRow(self.info)

        self.glw = pg.GraphicsLayoutWidget()
        self.p_trace = self.glw.addPlot(row=0, col=0); self.p_trace.setLabel("left", "电流", units="A")
        self.p_det = self.glw.addPlot(row=1, col=0); self.p_det.setLabel("bottom", "时间", units="s")
        self.p_det.setLabel("left", "检测曲线"); self.p_det.setXLink(self.p_trace)
        self.glw.ci.layout.setRowStretchFactor(0, 3); self.glw.ci.layout.setRowStretchFactor(1, 1)
        for pl in (self.p_trace, self.p_det):
            pl.setDownsampling(auto=True, mode="peak"); pl.setClipToView(True); pl.showGrid(x=True, y=True, alpha=0.15)
        self.c_raw = self.p_trace.plot(pen=pg.mkPen("#bbbbbb"))
        self.c_proc = self.p_trace.plot(pen=pg.mkPen("#222222"))
        self.c_events = pg.ScatterPlotItem(size=7, pen=None)
        self.p_trace.addItem(self.c_events)
        self.c_det = self.p_det.plot(pen=pg.mkPen("#4C72B0"))
        self.thr_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#C44E52", style=Qt.DashLine)); self.p_det.addItem(self.thr_line)
        return self.split_controls(form_w, self.glw)

    def _tab_events(self):
        w = QtWidgets.QSplitter(Qt.Horizontal)
        left = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(left)
        self.ev_table = QtWidgets.QTableWidget(0, 0)
        self.ev_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ev_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ev_table.itemSelectionChanged.connect(self.on_event_selected)
        v.addWidget(self.ev_table)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("排除 / 恢复所选事件（Del）"); b.clicked.connect(self.toggle_exclude); row.addWidget(b)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_Delete), self.ev_table, self.toggle_exclude)
        self.ev_count = QtWidgets.QLabel(""); row.addWidget(self.ev_count)
        row.addWidget(self.clear_button("清除审核标记", "review"))
        v.addLayout(row)
        w.addWidget(left)
        g = pg.GraphicsLayoutWidget()
        self.p_ev = g.addPlot(row=0, col=0, title="所选事件（原位）"); self.p_ev.setLabel("bottom", "时间", units="s")
        self.c_ev_trace = self.p_ev.plot(pen=pg.mkPen("#222222"))
        self.c_ev_mark = pg.ScatterPlotItem(size=9, brush=pg.mkBrush("#C44E52"), pen=None); self.p_ev.addItem(self.c_ev_mark)
        self.p_avg = g.addPlot(row=1, col=0, title="全部有效事件叠加与平均"); self.p_avg.setLabel("bottom", "时间 (ms)")
        w.addWidget(g); w.setSizes([650, 650])
        return w

    def _tab_deconv(self):
        fw = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(fw)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.d = {"tau_rise": _spin(0.4, 0.02, 50, 0.1, 2, "ms"), "tau_decay": _spin(4, 0.1, 500, 1, 2, "ms"),
                  "smooth": _spin(0.5, 0, 20, 0.1, 2, "ms"), "thr": _spin(4, 1, 20, 0.5, 1, "SD"),
                  "hp": _spin(1, 0, 20, 0.5, 1, "Hz"), "rate": _spin(1.0, 0.05, 30, 0.5, 2, "s")}
        form.addRow("模板 τ上升", self.d["tau_rise"]); form.addRow("模板 τ衰减", self.d["tau_decay"])
        b = QtWidgets.QPushButton("用本细胞平均事件拟合模板"); b.clicked.connect(self.fit_template); form.addRow(b)
        form.addRow("平滑宽度（高斯 σ）", self.d["smooth"]); form.addRow("阈值", self.d["thr"])
        form.addRow("高通", self.d["hp"]); form.addRow("事件率曲线窗口", self.d["rate"])
        b = QtWidgets.QPushButton("运行去卷积"); b.clicked.connect(self.run_deconv); form.addRow(b)
        b = QtWidgets.QPushButton("把去卷积结果用作本细胞的检测结果"); b.clicked.connect(self.deconv_as_detection); form.addRow(b)
        tip = QtWidgets.QLabel(
            "去卷积把每个缓慢衰减的突触电流压缩成一个尖峰，因此叠在一起的事件可以分开。"
            "平滑宽度越大噪声越小、但时间分辨率越低；一般取上升时间量级。\n"
            "上图红线是“稀疏事件 × 模板”的重建曲线：它贴合原始记录说明模板合适；"
            "系统性偏离（衰减太快/太慢）说明应调整 τ，或事件动力学不均一（→ 动力学聚类）。")
        tip.setWordWrap(True); tip.setStyleSheet("color:#555"); form.addRow(tip)
        self.deconv_info = QtWidgets.QLabel(""); self.deconv_info.setWordWrap(True); form.addRow(self.deconv_info)
        form.addRow(self.clear_button("清除去卷积结果", "deconv"))
        g = pg.GraphicsLayoutWidget()
        self.pd1 = g.addPlot(row=0, col=0, title="原始记录（灰）与去噪重建（红）")
        self.pd2 = g.addPlot(row=1, col=0, title="去卷积曲线与阈值")
        self.pd3 = g.addPlot(row=2, col=0, title="事件率（Hz）"); self.pd3.setLabel("bottom", "时间", units="s")
        for pl in (self.pd1, self.pd2, self.pd3):
            pl.setDownsampling(auto=True, mode="peak"); pl.setClipToView(True)
        self.pd2.setXLink(self.pd1); self.pd3.setXLink(self.pd1)
        self.cd_raw = self.pd1.plot(pen=pg.mkPen("#999999")); self.cd_rec = self.pd1.plot(pen=pg.mkPen("#C44E52", width=1.5))
        self.cd_dec = self.pd2.plot(pen=pg.mkPen("#4C72B0"))
        self.cd_thr = pg.InfiniteLine(angle=0, pen=pg.mkPen("#C44E52", style=Qt.DashLine)); self.pd2.addItem(self.cd_thr)
        self.cd_pk = pg.ScatterPlotItem(size=6, brush=pg.mkBrush("#C44E52"), pen=None); self.pd2.addItem(self.cd_pk)
        self.cd_rate = self.pd3.plot(pen=pg.mkPen("#55A868", width=1.5))
        return self.split_controls(fw, g)

    def _tab_cluster(self):
        fw = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(fw)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.c = {}
        self.c["scope"] = QtWidgets.QComboBox(); self.c["scope"].addItems(["当前细胞", "全部细胞合并"])
        self.c["rep"] = QtWidgets.QComboBox(); self.c["rep"].addItems(["波形形状（峰值归一化 + PCA）", "标量特征"])
        self.c["method"] = QtWidgets.QComboBox(); self.c["method"].addItems(["GMM（高斯混合）", "K-means", "HDBSCAN"])
        self.c["k"] = QtWidgets.QSpinBox(); self.c["k"].setRange(0, 8); self.c["k"].setSpecialValueText("自动")
        self.c["emb"] = QtWidgets.QComboBox(); self.c["emb"].addItems(["PCA", "UMAP"])
        self.c["nooverlap"] = QtWidgets.QCheckBox("只用无重叠事件"); self.c["nooverlap"].setChecked(True)
        form.addRow("范围", self.c["scope"]); form.addRow("表征", self.c["rep"])
        box = QtWidgets.QGroupBox("标量特征"); bl = QtWidgets.QVBoxLayout(box); self.c["feat"] = {}
        for f in FEATURES:
            cb = QtWidgets.QCheckBox(plots.FEATURE_LABELS[f]); cb.setChecked(f in ("rise_10_90", "decay_tau", "half_width", "amplitude"))
            self.c["feat"][f] = cb; bl.addWidget(cb)
        form.addRow(box)
        form.addRow("聚类方法", self.c["method"]); form.addRow("簇数", self.c["k"]); form.addRow("二维展示", self.c["emb"])
        form.addRow(self.c["nooverlap"])
        b = QtWidgets.QPushButton("运行聚类"); b.clicked.connect(self.run_cluster); form.addRow(b)
        form.addRow(self.clear_button("清除所选范围的聚类", "cluster"))
        self.cluster_msg = QtWidgets.QLabel(""); self.cluster_msg.setWordWrap(True); form.addRow(self.cluster_msg)
        self.cluster_panel = FigurePanel()
        return self.split_controls(fw, self.cluster_panel)

    def split_controls(self, form, plot):
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(form)
        splitter = QtWidgets.QSplitter(Qt.Horizontal); splitter.setChildrenCollapsible(False)
        splitter.addWidget(scroll); splitter.addWidget(plot)
        splitter.setStretchFactor(1, 1); splitter.setSizes([350, 850])
        return splitter

    def set_ui_points(self, points):
        points = max(7, min(20, points))
        font = self.font(); font.setPointSize(points); self.setFont(font)
        for panel in (self.cluster_panel, self.an_panel):
            mono = panel.text.font(); mono.setPointSize(points); panel.text.setFont(mono)
        self.settings.setValue("ui/points", points)

    ANALYSES = ["细胞总览", "检测质控（注入 / 反极性 / 阈值扫描）", "上升时间–幅度相关（树突滤波诊断）",
                "事件时间依赖性（自相关 / Hawkes）", "峰值缩放非平稳噪声分析（PS-NSFA）", "量子多峰分析",
                "E/I 联合分析（当前细胞=E，另选一个=I）",
                "组间比较（层级统计）", "累积概率分布（幅度 / IEI）", "突触缩放（秩序图 / 比例图）", "各簇组成的组间比较"]

    def _tab_analysis(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        controls = QtWidgets.QWidget()
        self.an = QtWidgets.QComboBox(); self.an.addItems(self.ANALYSES)
        self.an_metric = QtWidgets.QComboBox()
        self.an_metric.addItems(["frequency"] + [f"{f}_{s}" for f in FEATURES for s in ("median", "mean")]
                                + ["iei_cv", "avg_event_tau", "noise_rms"])
        self.an_metric.setCurrentText("amplitude_median")
        self.an_other = QtWidgets.QComboBox()
        self.an_other.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.an_other.setMinimumContentsLength(16)
        self.an_df = _spin(70, 1, 200, 5, 0, "mV")
        self.an_cdf = QtWidgets.QComboBox()
        self.an_cdf.addItems(["按细胞平均", "等量抽取后合并", "全部事件合并"])
        self.an_cdf.setItemData(0, "推荐：每个细胞权重相同", Qt.ToolTipRole)
        # 下拉框按“最短可读长度”定宽，长选项在展开的列表里完整显示（避免把窗口撑宽）
        for cb, n in ((self.an, 22), (self.an_metric, 14), (self.an_cdf, 8)):
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(n)
            cb.view().setMinimumWidth(cb.view().sizeHintForColumn(0) + 30)
            cb.setToolTip("当前：" + cb.currentText())
            cb.currentTextChanged.connect(lambda t, c=cb: c.setToolTip("当前：" + t))
        b = QtWidgets.QPushButton("运行"); b.clicked.connect(self.run_analysis)
        # 两行排布，窄屏（1280 宽、150% 缩放）也不会把控件挤出窗口
        grid = QtWidgets.QGridLayout(); grid.setHorizontalSpacing(8)
        grid.addWidget(QtWidgets.QLabel("分析"), 0, 0, Qt.AlignRight)
        cells = [[],
                 [("组间指标", self.an_metric), ("累积分布画法", self.an_cdf)],
                 [("I 细胞", self.an_other), ("NSFA 驱动力", self.an_df)]]
        for r_, items in enumerate(cells):
            for c_, (lab, wd) in enumerate(items):
                grid.addWidget(QtWidgets.QLabel(lab), r_, 2 * c_, Qt.AlignRight)
                grid.addWidget(wd, r_, 2 * c_ + 1)
        grid.addWidget(self.an, 0, 1, 1, 3)
        grid.addWidget(b, 0, 4, 1, 2)
        grid.addWidget(self.clear_button("清除当前分析结果", "analysis"), 1, 4, 1, 2)
        grid.setColumnStretch(1, 3); grid.setColumnStretch(3, 2)
        for cb in (self.an, self.an_metric, self.an_cdf, self.an_other, self.an_df):
            cb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        controls.setLayout(grid)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls)
        scroll.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        scroll.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Maximum)
        v.addWidget(scroll)
        self.an_panel = FigurePanel(); v.addWidget(self.an_panel, 1)
        return w

    def clear_button(self, label, action):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(lambda: self.clear_results(action))
        return button

    def clear_deconv_view(self):
        self.deconv_res = None
        self.deconv_info.clear()
        for curve in (self.cd_raw, self.cd_rec, self.cd_dec, self.cd_pk, self.cd_rate):
            curve.setData([], [])
        self.cd_thr.hide()

    def clear_results(self, action):
        if self.workers:
            self.statusBar().showMessage("分析任务仍在运行，请在任务结束后清除。")
            return
        pooled = action == "batch" or (action == "cluster" and self.c["scope"].currentIndex() == 1)
        cells = self.project.cells if pooled else ([self.current] if self.current else [])
        if not cells and action != "analysis":
            return
        scope = f"全部 {len(cells)} 个细胞" if pooled else (self.current.name if cells else "")
        effects = {"preprocess": "清除预处理、检测、审核、质控和聚类结果", "detect": "清除检测、审核、质控和聚类结果，保留预处理",
                   "batch": "清除检测、审核、质控和聚类结果，保留预处理", "review": "恢复所有被排除的事件，并清除该细胞的旧聚类和检测质控",
                   "cluster": "删除簇标签，保留检测和审核", "deconv": "清除去卷积图和结果，保留已采用的检测结果"}
        if action != "analysis":
            answer = QtWidgets.QMessageBox.question(self, "确认清除", f"{scope}：{effects[action]}。\n保留原始记录和参数。继续吗？",
                                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        if action in ("preprocess", "detect", "batch"):
            for cell in cells:
                cell.clear_detection(preprocessing=action == "preprocess")
        elif action in ("review", "cluster"):
            for cell in cells:
                if cell.events is not None:
                    cell.events["cluster"] = -1
                    if action == "review":
                        cell.events["excluded"] = False
                        cell.clear_qc_results()
        if action in ("deconv", "preprocess", "detect", "batch"):
            self.clear_deconv_view()
        if action != "deconv":
            if action == "analysis" and self.qc_result_cell is not None:
                self.qc_result_cell.clear_qc_results()
            self.qc_result_cell = None
            self.an_panel.clear_result()
            if action != "analysis":
                self.cluster_panel.clear_result(); self.cluster_msg.clear()
                self.refresh_cells(); self.refresh_trace(); self.refresh_events()
        self.dirty = True
        self.statusBar().showMessage("已清除；可调整参数后重新运行。")

    def new_project(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "输入新项目文件夹名称（必须尚不存在）", "", "所有名称 (*)",
                                                        options=QtWidgets.QFileDialog.DontConfirmOverwrite)
        if path:
            self.activate_project(path, create=True)

    def choose_project(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择包含 project.pca.h5 的项目文件夹")
        if folder:
            self.activate_project(folder)

    def activate_project(self, path, create=False):
        if self.workers or (self.dirty and not self.save_session()):
            return False
        root = Path(path).resolve()
        lock = self.project_lock if root == self.project_root else None
        try:
            if create:
                project = project_files.create_project(root)
            if not root.is_dir():
                raise ValueError("项目文件夹不存在")
            if lock is None:
                lock = QtCore.QLockFile(str(root / ".patchclamp.lock")); lock.setStaleLockTime(0)
                if not lock.tryLock(0):
                    raise ValueError("无法锁定项目：请确认文件夹可写，且没有其他窗口正在使用此项目。")
            if not create:
                project = project_files.open_project(root)
        except Exception as error:
            if lock is not None and lock is not self.project_lock and lock.isLocked():
                lock.unlock()
            QtWidgets.QMessageBox.warning(self, "无法新建或打开项目", str(error)); return False
        if self.project_lock is not None and lock is not self.project_lock:
            self.project_lock.unlock()
        self.cell_filters.clear()
        self.cell_sort = None
        self.project_lock = lock
        self.release_project_resources()
        self.project, self.project_root, self.current = project, root, None
        for panel in (self.cluster_panel, self.an_panel):
            panel.export_folder = root / "results"
        self.dirty = False; self.qc_result_cell = None
        self.workspace.setCurrentIndex(1); self.cell_dock.show(); self.tabs.setCurrentIndex(0)
        for action in self.project_actions:
            action.setEnabled(True)
        self.cell_table.clearSelection(); self.cell_table.setCurrentCell(-1, -1)
        self.clear_deconv_view(); self.cluster_panel.clear_result(); self.an_panel.clear_result(); self.cluster_msg.clear()
        self.write_params(DEFAULT_PARAMS)
        if not self.miniml_ok:
            self.p["detector"].setCurrentText("去卷积")
        self.refresh_cells(); self.refresh_trace(); self.refresh_events()
        if project.cells:
            self.cell_table.selectRow(0)
        self.setWindowTitle(f"{root.name} — 膜迹 ClampTrace v{__version__}")
        self.statusBar().showMessage(f"项目：{root} ｜ Ctrl/Shift 多选记录；Ctrl+S 保存项目")
        return True

    def save_session(self):
        if self.project_root is None or self.workers:
            return False
        try:
            project_files.save_project(self.project_root, self.project)
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "项目保存失败", str(error)); return False
        self.dirty = False
        self.statusBar().showMessage(f"已保存项目：{self.project_root}")
        return True

    def show_results_folder(self):
        if self.project_root:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.project_root / "results")))

    def archive_result(self, fig, text, details, cells, arrays=None):
        try:
            details.update(version=__version__, records=[dict(name=cell.name, group=cell.group, animal=cell.animal,
                params=cell.params, channel=cell.rec.channel, fs=cell.rec.fs, valid_events=int(cell.mask().sum()),
                raw=Path(cell.rec.path).relative_to(self.project_root).as_posix() if cell.rec.path else None) for cell in cells])
            folder = project_files.archive_result(self.project_root, fig, text, details, arrays)
            self.statusBar().showMessage(f"分析结果已保存：{folder}")
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "分析已完成，但结果未保存", str(error))

    # ================================================================ 细胞管理
    def open_files(self):
        if self.project_root is None or self.workers:
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "打开记录", self.settings.value("last_dir", ""),
                                                          "记录文件 (*.abf *.h5 *.hdf5 *.npy);;所有文件 (*)")
        if paths:
            self.settings.setValue("last_dir", os.path.dirname(paths[0]))
            self.add_files(paths)

    def annotation_csv(self, importing):
        if not self.project_root or self.workers:
            return
        dialog = QtWidgets.QFileDialog
        path, _ = (dialog.getOpenFileName(self, "导入模板（空值将清除旧值）", str(self.project_root), "CSV (*.csv)") if importing
                   else dialog.getSaveFileName(self, "导出模板", str(self.project_root / "记录信息.csv"), "CSV (*.csv)"))
        if path:
            try:
                count = import_metadata.template(path, self.project_root, self.project.cells, importing)
                if importing:
                    self.dirty = True
                    self.refresh_cells()
                    if not self.save_session():
                        return
                self.statusBar().showMessage(f"CSV 操作完成：{count} 条记录")
            except (OSError, ValueError, csv.Error) as exc:
                QtWidgets.QMessageBox.warning(self, "CSV 模板", str(exc))

    def open_folder(self):
        if not self.project_root or self.workers:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "递归添加文件夹", self.settings.value("last_dir", ""))
        if folder:
            scan = QtWidgets.QProgressDialog("正在扫描文件夹及子目录…", "", 0, 0, self)
            scan.setCancelButton(None)
            scan.setWindowModality(Qt.WindowModal)
            scan.show()
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    groups = wait_for_io(pool, lambda: {str(path): group for path, group in import_metadata.scan(folder, self.project_root)})
            finally:
                scan.close()
                scan.deleteLater()
            if not groups:
                self.statusBar().showMessage("文件夹中没有支持的记录")
                return
            self.settings.setValue("last_dir", folder)
            self.add_files(list(groups), groups)

    def add_files(self, paths, groups=None):
        if self.project_root is None or self.workers:
            return
        paths = list(map(str, paths))
        if not paths:
            return
        total = len(paths)
        dialog = QtWidgets.QProgressDialog("准备导入…", "停止后续导入", 0, total, self)
        dialog.setWindowTitle("导入记录")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.show()
        imported = 0
        errors = []
        pool = ThreadPoolExecutor(max_workers=1)

        for index, p in enumerate(paths):
            if dialog.wasCanceled():
                break
            label = f"{index + 1}/{total} · {Path(p).name}"
            dialog.setLabelText(f"读取记录：{label}　已成功 {imported} 条")
            try:
                ch = 0
                if p.lower().endswith(".abf"):
                    inf = wait_for_io(pool, io.abf_info, p)
                    if dialog.wasCanceled():
                        break
                    if len(inf["channels"]) > 1:
                        items = [f"{c}: {n} ({u})" for c, n, u in zip(inf["channels"], inf["channel_names"], inf["units"])]
                        it, ok = QtWidgets.QInputDialog.getItem(dialog, "选择通道", os.path.basename(p), items, 0, False)
                        if not ok:
                            continue
                        ch = inf["channels"][items.index(it)]
                record = wait_for_io(pool, io.load_any, p, ch)
                dialog.setLabelText(f"复制到项目：{label}　已成功 {imported} 条")
                record.path = str(wait_for_io(pool, project_files.copy_raw, self.project_root, p))
                cell = CellSession(record, group=(groups or {}).get(p, ""))
                preset = detection.preset_from_path(p)
                if preset:
                    cell.params.update(detection.PRESETS[preset])
                self.add_cell(cell, refresh=False)
                imported += 1
            except Exception as e:
                errors.append(f"{p}：{e}")
            finally:
                if not dialog.wasCanceled():
                    dialog.setValue(index + 1)
        pool.shutdown()
        dialog.close()
        dialog.deleteLater()
        if imported:
            self.refresh_cells()
            self.select_visible_cell(len(self.project.cells) - 1)
        self.statusBar().showMessage(f"导入结束：成功 {imported}/{total} 条，失败 {len(errors)} 条")
        if errors:
            QtWidgets.QMessageBox.warning(self, "部分文件读取失败", "\n".join(errors[:10]))

    def add_cell(self, cell: CellSession, refresh=True):
        if self.project_root is None:
            return
        self.dirty = True
        if not self.miniml_ok and cell.params["detector"] == "miniML":
            cell.params["detector"] = "deconvolution"
        names = {entry.name for entry in self.project.cells}
        original = cell.name
        suffix = 1
        while cell.name in names:
            suffix += 1; cell.rec.name = f"{original} ({suffix})"
        self.project.cells.append(cell)
        if not refresh:
            return
        self.refresh_cells()
        self.select_visible_cell(len(self.project.cells) - 1)

    def select_visible_cell(self, preferred):
        table = self.cell_table
        if not table.isRowHidden(preferred):
            table.selectRow(preferred)
        else:
            for index in range(table.rowCount()):
                row = table.verticalHeader().logicalIndex(index)
                if not table.isRowHidden(row):
                    table.selectRow(row)
                    break

    def load_demo(self):
        if self.project_root is None or self.workers:
            return
        pops_c = [dict(frac=0.6, amp=18, amp_cv=0.35, tau_rise=0.3, tau_decay=3.0),
                  dict(frac=0.4, amp=10, amp_cv=0.30, tau_rise=1.2, tau_decay=9.0)]
        pops_t = [dict(frac=0.6, amp=25, amp_cv=0.35, tau_rise=0.3, tau_decay=3.0),
                  dict(frac=0.4, amp=14, amp_cv=0.30, tau_rise=1.2, tau_decay=9.0)]
        for i in range(8):
            treated = i >= 4
            rec = io.simulate_recording(60, rate_hz=3.0 + 0.4 * (i % 4), populations=pops_t if treated else pops_c,
                                        seed=100 + i, name=f"demo_{'TTX48h' if treated else 'ctrl'}_{i % 4 + 1}")
            c = CellSession(rec, group="TTX 48h" if treated else "Control", animal=f"M{i // 2 + 1}")
            self.add_cell(c)
        self.statusBar().showMessage("已载入 8 个模拟细胞。可多选记录应用参数，再按各自参数检测。")

    def remove_cell(self):
        r = self.cell_table.currentRow()
        if 0 <= r < len(self.project.cells):
            self.dirty = True
            del self.project.cells[r]
            self.current = None
            self.refresh_cells()
            if self.project.cells:
                self.select_visible_cell(min(r, len(self.project.cells) - 1)); self.on_cell_selected()
            else:
                self.refresh_trace(); self.refresh_events(); self.clear_deconv_view()

    def refresh_cells(self):
        self.cell_table.blockSignals(True)
        self.cell_table.setRowCount(len(self.project.cells))
        for i, c in enumerate(self.project.cells):
            n = int(c.mask().sum()) if c.events is not None else None
            vals = [c.name, c.group, c.animal, "" if n is None else str(n),
                    "" if n is None else f"{n / c.rec.duration:.2f}", c.note]
            for j, val in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(val)
                if j not in (1, 2, 5):
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.cell_table.setItem(i, j, it)
            combo = self.cell_table.cellWidget(i, 6)
            if combo is None:
                combo = QtWidgets.QComboBox()
                combo.addItems(["自定义", *detection.PRESETS])
                combo.activated.connect(lambda _, row=i, box=combo: self.apply_preset(row, box.currentText()))
                self.cell_table.setCellWidget(i, 6, combo)
            combo.setCurrentText(next((name for name, preset in detection.PRESETS.items()
                                       if all(c.params[key] == val for key, val in preset.items())), "自定义"))
        self.cell_table.blockSignals(False)
        self.update_cell_view()
        cur = self.an_other.currentText()
        self.an_other.clear(); self.an_other.addItems([c.name for c in self.project.cells]); self.an_other.setCurrentText(cur)

    def cell_value(self, row, column):
        if column == 6:
            return self.cell_table.cellWidget(row, 6).currentText()
        return self.cell_table.item(row, column).text()

    def sort_cell_column(self, column):
        descending = self.cell_sort == (column, False)
        self.cell_sort = column, descending
        self.update_cell_view()

    def reset_cell_view(self):
        self.cell_filters.clear()
        self.cell_sort = None
        self.update_cell_view()

    def cell_header_menu(self, position):
        header = self.cell_table.horizontalHeader()
        column = header.logicalIndexAt(position)
        if column < 0:
            return
        values = sorted({self.cell_value(row, column) for row in range(self.cell_table.rowCount())})
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("筛选：" + self.cell_table.horizontalHeaderItem(column).text())
        layout = QtWidgets.QVBoxLayout(dialog)
        search = QtWidgets.QLineEdit()
        search.setPlaceholderText("搜索选项（不改变勾选状态）")
        layout.addWidget(search)
        choices = QtWidgets.QListWidget()
        layout.addWidget(choices)
        allowed = self.cell_filters.get(column)
        for value in values:
            item = QtWidgets.QListWidgetItem(value or "（空白）", choices)
            item.setData(Qt.UserRole, value)
            item.setCheckState(Qt.Checked if allowed is None or value in allowed else Qt.Unchecked)
        def items():
            return (choices.item(index) for index in range(choices.count()))
        def search_values(term):
            for item in items():
                item.setHidden(term.casefold() not in item.text().casefold())
        search.textChanged.connect(search_values)
        def mark(state):
            for item in items():
                if not item.isHidden():
                    item.setCheckState(state)
        for label, state in (("全选搜索结果", Qt.Checked), ("取消勾选搜索结果", Qt.Unchecked)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _, value=state: mark(value))
            layout.addWidget(button)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            selected = {item.data(Qt.UserRole) for item in items() if item.checkState() == Qt.Checked}
            if len(selected) == len(values):
                self.cell_filters.pop(column, None)
            else:
                self.cell_filters[column] = selected
            self.update_cell_view()
        dialog.deleteLater()

    def update_cell_view(self):
        table = self.cell_table
        order = list(range(table.rowCount()))
        header = table.horizontalHeader()
        header.setSortIndicatorShown(self.cell_sort is not None)
        if self.cell_sort:
            column, descending = self.cell_sort
            def key(row):
                value = self.cell_value(row, column)
                return float(value or "-inf") if column in (3, 4) else value.casefold()
            order.sort(key=key, reverse=descending)
            header.setSortIndicator(column, Qt.DescendingOrder if descending else Qt.AscendingOrder)
        vertical = table.verticalHeader()
        visible = []
        for index, row in enumerate(order):
            old = vertical.visualIndex(row)
            if old != index:
                vertical.moveSection(old, index)
            hidden = any(self.cell_value(row, col) not in selected for col, selected in self.cell_filters.items())
            table.setRowHidden(row, hidden)
            if not hidden:
                visible.append(row)
            elif table.selectionModel().isRowSelected(row, QtCore.QModelIndex()):
                table.selectionModel().select(table.model().index(row, 0), QtCore.QItemSelectionModel.Deselect | QtCore.QItemSelectionModel.Rows)
        if table.currentRow() >= 0 and table.isRowHidden(table.currentRow()):
            if visible:
                table.selectRow(visible[0])
            else:
                table.setCurrentCell(-1, -1)
                self.current = None
                self.clear_deconv_view(); self.refresh_trace(); self.refresh_events()
        self.filter_summary.setText(f"显示 {len(visible)}/{len(order)} 条 · " + ("筛选中" if self.cell_filters else "右键列名筛选"))

    def on_cell_edited(self, it):
        self.dirty = True
        c = self.project.cells[it.row()]
        if it.column() == 1:
            c.group = it.text().strip()
        elif it.column() == 2:
            c.animal = it.text().strip()
        elif it.column() == 5:
            c.note = it.text()
        if self.cell_sort or self.cell_filters:
            self.update_cell_view()

    def on_cell_selected(self):
        r = self.cell_table.currentRow()
        if not (0 <= r < len(self.project.cells)):
            return
        self.current = self.project.cells[r]
        self.write_params(self.current.params)
        self.clear_deconv_view()
        self.refresh_trace(reset=True)
        self.refresh_events()

    # ================================================================ 参数
    def apply_preset(self, row, name):
        cell = self.project.cells[row]
        preset = detection.PRESETS.get(name)
        if preset and any(cell.params[key] != value for key, value in preset.items()):
            if cell.det is not None and QtWidgets.QMessageBox.question(self, "更改事件类型", "更改此记录的事件类型将清除旧检测结果。继续吗？",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                self.refresh_cells()
                return
            cell.clear_detection()
            cell.params.update(preset)
            self.dirty = True
        self.cell_table.selectRow(row)
        self.on_cell_selected()
        self.refresh_cells()

    def read_params(self) -> dict:
        out = {}
        for k, wd in self.p.items():
            out[k] = wd.currentText() if isinstance(wd, QtWidgets.QComboBox) else wd.value()
        out["detector"] = "miniML" if out["detector"] == "miniML" else "deconvolution"
        return out

    def write_params(self, params):
        for k, wd in self.p.items():
            if k not in params:
                continue
            val = params[k]
            if k == "detector":
                val = "miniML" if val == "miniML" else "去卷积"
            wd.blockSignals(True)
            wd.setCurrentText(str(val)) if isinstance(wd, QtWidgets.QComboBox) else wd.setValue(float(val))
            wd.blockSignals(False)
        self.d["tau_rise"].setValue(params["tau_rise_ms"]); self.d["tau_decay"].setValue(params["tau_decay_ms"])

    # ================================================================ 后台任务
    def update_task_status(self):
        seconds = self.task_clock.elapsed() // 1000
        state = "取消中（保留已完成记录）" if self.batch_cancelled else "暂停请求（记录之间等待）" if self.batch_paused else ""
        self.statusBar().showMessage(f"{state} {self.task_stage} · 已用 {seconds // 60:02d}:{seconds % 60:02d}")

    def toggle_batch_pause(self):
        with self.batch_condition:
            self.batch_paused = not self.batch_paused
            self.batch_condition.notify_all()
        self.pause_button.setText("继续" if self.batch_paused else "暂停")
        self.update_task_status()

    def cancel_batch(self):
        with self.batch_condition:
            self.batch_cancelled = True
            self.batch_condition.notify_all()
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.update_task_status()

    @QtCore.Slot(int, str)
    def batch_advanced(self, completed, stage):
        self.progress.setValue(completed)
        self.task_stage = stage
        self.update_task_status()

    def run_bg(self, fn, on_done, msg, total=0):
        if self.workers or self.project_root is None:
            return
        self.dirty = True
        self.tabs.setEnabled(False); self.cell_dock.setEnabled(False)
        for action in self.project_actions:
            action.setEnabled(False)
        self.last_error = None
        self.task_stage = msg
        self.batch_paused = self.batch_cancelled = False
        self.pause_button.setText("暂停")
        for button in (self.pause_button, self.cancel_button):
            button.setVisible(bool(total))
            button.setEnabled(True)
        self.task_clock.start()
        self.update_task_status()
        self.task_timer.start(1000)
        self.progress.setRange(0, total)
        self.progress.setFormat("已完成 %v / %m 条（%p%）" if total else "")
        self.progress.setValue(0)
        self.progress.show()
        wk = Worker(fn, self, tracked=bool(total))
        wk.advanced.connect(self.batch_advanced)
        self.workers.append(wk)

        def unlock():
            self.task_timer.stop()
            self.pause_button.hide()
            self.cancel_button.hide()
            self.progress.hide(); self.workers.remove(wk)
            self.tabs.setEnabled(True); self.cell_dock.setEnabled(True)
            for action in self.project_actions:
                action.setEnabled(True)

        def finish(res):
            unlock(); on_done(res)

        def fail(tb):
            unlock()
            self.statusBar().showMessage("出错")
            self.last_error = tb
            if os.environ.get("PATCHCLAMP_NO_DIALOGS"):      # 自动测试时不弹模态框
                return
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "出错", tb.strip().splitlines()[-1], parent=self)
            box.setDetailedText(tb); box.exec()

        wk.done.connect(finish); wk.failed.connect(fail)
        wk.finished.connect(wk.deleteLater)
        wk.start()
        return wk

    # ================================================================ 检测
    def run_preprocess(self):
        c = self.current
        if c is None:
            return
        c.params.update(self.read_params())
        c.run_preprocess()
        self.dirty = True
        self.refresh_trace()
        self.statusBar().showMessage(f"预处理完成，噪声 RMS ≈ {c.qc['noise_rms']:.2f} {c.rec.unit}")

    def run_detect_current(self):
        c = self.current
        if c is None:
            return
        c.params.update(self.read_params())

        def job():
            c.run_preprocess()
            return c.run_detection()

        self.run_bg(job, lambda n: self._detected(f"{c.name}：检出 {n} 个事件"), f"正在检测 {c.name} …")

    def run_detect_all(self):
        self.detect_batch(list(self.project.cells))

    def selected_cells(self):
        return [self.project.cells[index.row()] for index in self.cell_table.selectionModel().selectedRows()
                if not self.cell_table.isRowHidden(index.row())]

    def apply_selected(self):
        params = self.read_params()
        cells = [cell for cell in self.selected_cells() if any(cell.params[key] != value for key, value in params.items())]
        if not cells:
            self.statusBar().showMessage("未选中记录，或所选记录已使用这些参数。")
            return
        if any(cell.processed is not None for cell in cells):
            if QtWidgets.QMessageBox.question(self, "应用参数", f"参数变更将清除 {len(cells)} 条记录的旧预处理和检测结果。继续吗？",
                                              QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                return
        for cell in cells:
            cell.clear_detection(preprocessing=True); cell.params.update(params)
        self.dirty = True; self.clear_deconv_view(); self.cluster_panel.clear_result(); self.cluster_msg.clear()
        self.an_panel.clear_result(); self.qc_result_cell = None
        self.refresh_cells(); self.refresh_trace(); self.refresh_events()
        self.statusBar().showMessage(f"已向 {len(cells)} 条选中记录应用参数；可开始检测。")

    def run_detect_selected(self):
        self.detect_batch(self.selected_cells())

    def choose_cpu_count(self):
        capacity = available()
        previous = self.settings.value("analysis/cpus", 0, type=int)
        value, ok = QtWidgets.QInputDialog.getInt(self, "分析 CPU 核心数",
            f"可用逻辑核心：{capacity}。0 = 自动 70%（{budget(0, capacity)} 核）。\n"
            "去卷积并行数下次批处理生效；数值库和 miniML 线程设置重启后生效。",
            min(capacity, max(0, previous)), 0, capacity)
        if ok:
            self.settings.setValue("analysis/cpus", value)
            self.settings.sync()

    def detect_batch(self, cells):
        if not cells:
            self.statusBar().showMessage("请先选择要检测的记录。")
            return

        limit = budget(self.settings.value("analysis/cpus", 0), available())
        serial = any(cell.params["detector"] == "miniML" for cell in cells)
        lanes = 1 if serial else min(limit, len(cells))

        def job(report):
            from threadpoolctl import threadpool_limits
            pending = iter(enumerate(cells))
            completed = total_events = 0
            stopped = threading.Event()

            def process_lane():
                nonlocal completed, total_events
                try:
                    while True:
                        with self.batch_condition:
                            if self.batch_paused and not self.batch_cancelled:
                                report(completed, "已暂停，等待继续")
                            self.batch_condition.wait_for(lambda: not self.batch_paused or self.batch_cancelled or stopped.is_set())
                            if self.batch_cancelled or stopped.is_set():
                                return
                            item = next(pending, None)
                            if item is None:
                                return
                            index, cell = item
                            label = f"第 {index + 1} 条 · {cell.name[:36]}"
                            report(completed, f"预处理：{label}")
                        cell.run_preprocess()
                        with self.batch_condition:
                            report(completed, f"检测：{label}")
                        count = cell.run_detection()
                        with self.batch_condition:
                            total_events += count
                            completed += 1
                            report(completed, f"已完成：{label}")
                except BaseException:
                    with self.batch_condition:
                        stopped.set()
                        self.batch_condition.notify_all()
                    raise

            with threadpool_limits(limits=1 if lanes > 1 else limit), ThreadPoolExecutor(max_workers=lanes) as pool:
                futures = [pool.submit(process_lane) for _ in range(lanes)]
                for future in futures:
                    future.result()
            if completed < len(cells):
                return f"已取消：保留 {completed}/{len(cells)} 条已完成记录，共 {total_events} 个事件"
            return f"批量检测完成：{completed} 个细胞，共 {total_events} 个事件"

        self.run_bg(job, self._detected,
                    f"正在批量检测 {len(cells)} 个细胞 …", total=len(cells))

    def _detected(self, msg):
        self.statusBar().showMessage(msg)
        self.refresh_cells()
        self.refresh_trace()
        self.refresh_events()

    def refresh_trace(self, reset=False):
        c = self.current
        if c is None:
            self.info.clear(); self.thr_line.hide()
            for curve in (self.c_raw, self.c_proc, self.c_events, self.c_det):
                curve.setData([], [])
            return
        t = c.rec.time
        scale = 1e-12 if c.rec.unit.lower().startswith("p") else 1.0
        self.p_trace.setLabel("left", "电流" if scale != 1.0 else f"信号 ({c.rec.unit})", units="A" if scale != 1.0 else None)
        self.c_raw.setData(t, c.rec.data * scale)
        if c.processed is not None:
            off = np.median(c.rec.data) - np.median(c.processed)
            self.c_proc.setData(t, (c.processed + off) * scale)
        else:
            self.c_proc.setData([], [])
        self.c_events.setData([], []); self.c_det.setData([], [])
        self.thr_line.setVisible(c.det is not None and c.det.trace is not None)
        if c.events is not None and len(c.events):
            off = np.median(c.rec.data) - np.median(c.processed)
            pk = c.events["peak_idx"].to_numpy(int)
            brushes = []
            for ex, cl in zip(c.events["excluded"], c.events["cluster"]):
                brushes.append(pg.mkBrush("#cccccc") if ex else pg.mkBrush(CLUSTER_COLORS[cl % 6] if cl >= 0 else QtGui.QColor("#C44E52")))
            self.c_events.setData(pk / c.rec.fs, (c.processed[pk] + off) * scale, brush=brushes)
        if c.det is not None and c.det.trace is not None:
            tr = c.det.trace
            self.c_det.setData(np.linspace(0, c.rec.duration, len(tr)), tr)
            self.thr_line.setValue(c.det.threshold or 0)
        s = f"{c.name}｜{c.rec.fs:g} Hz｜{c.rec.duration:.1f} s｜单位 {c.rec.unit}"
        if "noise_rms" in c.qc:
            s += f"｜噪声 RMS {c.qc['noise_rms']:.2f}"
        if c.events is not None:
            n = int(c.mask().sum()); s += f"\n事件 {n} 个，{n / c.rec.duration:.2f} Hz（{c.det.method}）"
        self.info.setText(s)
        if reset:
            self.p_trace.enableAutoRange(); self.p_det.enableAutoRange()

    # ================================================================ 事件审核
    EV_COLS = ["time_s", "amplitude", "rise_10_90", "decay_tau", "half_width", "charge", "score", "cluster", "excluded"]
    EV_HEAD = ["时间 s", "幅度 pA", "上升 ms", "衰减τ ms", "半宽 ms", "电荷 fC", "得分", "簇", "排除"]

    def refresh_events(self):
        c = self.current
        self.ev_table.blockSignals(True)
        self.ev_table.clear(); self.p_avg.clear()
        self.c_ev_trace.setData([], []); self.c_ev_mark.setData([], []); self.ev_count.clear()
        if c is None or c.events is None:
            self.ev_table.setRowCount(0); self.ev_table.blockSignals(False); return
        df = c.events
        self.ev_table.setColumnCount(len(self.EV_COLS)); self.ev_table.setHorizontalHeaderLabels(self.EV_HEAD)
        self.ev_table.setRowCount(len(df))
        for i, row in enumerate(df[self.EV_COLS].itertuples(index=False)):
            for j, val in enumerate(row):
                name = self.EV_COLS[j]
                txt = ("是" if val else "") if name == "excluded" else (str(int(val) + 1) if name == "cluster" and val >= 0 else
                      ("" if name == "cluster" else f"{val:.3f}"))
                it = QtWidgets.QTableWidgetItem(txt)
                if row[-1]:
                    it.setForeground(QtGui.QBrush(QtGui.QColor("#aaaaaa")))
                self.ev_table.setItem(i, j, it)
        self.ev_table.blockSignals(False)
        m = c.mask()
        self.ev_count.setText(f"有效 {m.sum()} / 共 {len(df)}")
        sign = -1 if c.params["direction"] == "negative" else 1
        w = c.waves[m]
        if len(w):
            sel = np.random.default_rng(0).choice(len(w), min(len(w), 150), replace=False)
            x = np.tile(np.append(c.t_wave, np.nan), len(sel))
            y = np.concatenate([np.append(sign * w[i], np.nan) for i in sel])
            self.p_avg.plot(x, y, pen=pg.mkPen(180, 180, 180, 90), connect="finite")
            cl = df["cluster"].to_numpy()[m]
            if (cl >= 0).any():
                for u in sorted(set(cl[cl >= 0])):
                    self.p_avg.plot(c.t_wave, sign * np.nanmean(w[cl == u], axis=0), pen=pg.mkPen(CLUSTER_COLORS[u % 6], width=2.5))
            else:
                self.p_avg.plot(c.t_wave, sign * np.nanmean(w, axis=0), pen=pg.mkPen("#4C72B0", width=2.5))

    def on_event_selected(self):
        c = self.current
        r = self.ev_table.currentRow()
        if c is None or c.events is None or not (0 <= r < len(c.events)):
            return
        ev = c.events.iloc[r]
        win = int(c.params["window_ms"] / 1e3 * c.rec.fs)
        a, b = max(int(ev["onset_idx"]) - win, 0), min(int(ev["onset_idx"]) + 2 * win, len(c.processed))
        self.c_ev_trace.setData(np.arange(a, b) / c.rec.fs, c.processed[a:b])
        self.c_ev_mark.setData([ev["peak_idx"] / c.rec.fs], [c.processed[int(ev["peak_idx"])]])
        self.p_ev.enableAutoRange()

    def toggle_exclude(self):
        c = self.current
        if c is None or c.events is None:
            return
        rows = sorted({i.row() for i in self.ev_table.selectedIndexes()})
        if not rows:
            return
        col = c.events.columns.get_loc("excluded")
        for r in rows:
            c.events.iat[r, col] = not bool(c.events.iat[r, col])
        self.dirty = True
        self.refresh_events(); self.refresh_trace(); self.refresh_cells()
        self.ev_table.selectRow(min(rows[-1] + 1, len(c.events) - 1))

    # ================================================================ 去卷积
    def fit_template(self):
        c = self.current
        if c is None or c.events is None or c.mask().sum() < 10:
            self.deconv_info.setText("需要先检测到至少 10 个事件。"); return
        tr, td = deconv.fit_kernel_from_average(np.nanmean(c.waves[c.mask()], axis=0), c.rec.fs)
        self.d["tau_rise"].setValue(tr); self.d["tau_decay"].setValue(td)
        self.deconv_info.setText(f"由平均事件拟合：τ上升 = {tr:.2f} ms，τ衰减 = {td:.2f} ms")

    def run_deconv(self):
        c = self.current
        if c is None:
            return
        if c.processed is None:
            c.params.update(self.read_params()); c.run_preprocess()
        d = {k: wd.value() for k, wd in self.d.items()}
        direction = self.p["direction"].currentText()

        def job():
            return deconv.deconvolve(c.processed, c.rec.fs, d["tau_rise"], d["tau_decay"], direction,
                                     smooth_ms=d["smooth"], highpass_hz=d["hp"], threshold_sd=d["thr"])

        def done(res):
            if self.current is not c:
                return
            self.deconv_res = res
            self.cd_thr.show()
            t = c.rec.time
            base = c.processed - np.median(c.processed)
            self.cd_raw.setData(t, base); self.cd_rec.setData(t, res.reconvolved)
            self.cd_dec.setData(t, res.deconvolved); self.cd_thr.setValue(res.threshold)
            self.cd_pk.setData(res.peaks / c.rec.fs, res.peak_heights)
            wsec = d["rate"]
            edges = np.arange(0, c.rec.duration + wsec / 4, wsec / 4)
            cnt = np.histogram(res.peaks / c.rec.fs, edges)[0]
            rate = np.convolve(cnt, np.ones(4), mode="same") / wsec
            self.cd_rate.setData((edges[:-1] + edges[1:]) / 2, rate)
            resid = np.std(base - res.reconvolved); tot = np.std(base)
            self.deconv_info.setText(
                f"检出 {len(res.peaks)} 个事件（{len(res.peaks) / c.rec.duration:.2f} Hz）。\n"
                f"重建残差 SD = {resid:.2f}（原始记录 SD = {tot:.2f}，噪声 RMS ≈ {c.qc.get('noise_rms', np.nan):.2f}）。"
                "残差接近噪声水平说明模板与阈值合适。")
            self.archive_result(None, self.deconv_info.text(), dict(analysis="去卷积", direction=direction, settings=d,
                                threshold=res.threshold), [c], {key: getattr(res, key) for key in
                                ("peaks", "peak_heights", "deconvolved", "reconvolved", "kernel")})
            self.pd1.enableAutoRange(); self.pd2.enableAutoRange(); self.pd3.enableAutoRange()

        self.run_bg(job, done, "正在去卷积 …")

    def deconv_as_detection(self):
        c = self.current
        if c is None:
            return
        self.p["detector"].setCurrentText("去卷积")
        self.p["tau_rise_ms"].setValue(self.d["tau_rise"].value()); self.p["tau_decay_ms"].setValue(self.d["tau_decay"].value())
        self.p["deconv_threshold_sd"].setValue(self.d["thr"].value())
        c.params["deconv_smooth_ms"] = self.d["smooth"].value()
        self.run_detect_current()
        self.tabs.setCurrentIndex(0)

    # ================================================================ 聚类
    def run_cluster(self):
        pooled = self.c["scope"].currentIndex() == 1
        rep = "waveform" if self.c["rep"].currentIndex() == 0 else "features"
        method = ["gmm", "kmeans", "hdbscan"][self.c["method"].currentIndex()]
        k = self.c["k"].value() or None
        emb = "umap" if self.c["emb"].currentIndex() == 1 else "pca"
        feats = [f for f, cb in self.c["feat"].items() if cb.isChecked()]
        noov = self.c["nooverlap"].isChecked()
        if pooled:
            pw = self.project.pooled_waves()
            if pw is None:
                self.cluster_msg.setText("合并聚类要求所有细胞已检测，且采样率与事件窗口一致。"); return
            df, waves = pw
            cells = self.project.analyzed()
            t_wave, direction = cells[0].t_wave, cells[0].params["direction"]
        else:
            c = self.current
            if c is None or c.events is None:
                self.cluster_msg.setText("请先检测事件。"); return
            df, waves = c.valid()
            cells, t_wave, direction = [c], c.t_wave, c.params["direction"]
        if len(feats) < 2 and rep == "features":
            self.cluster_msg.setText("标量特征至少选 2 个。"); return

        def job():
            use = ~df["overlap"].to_numpy(bool) if noov else np.ones(len(df), bool)
            res = clustering.cluster_events(df[use].reset_index(drop=True), waves[use], rep, feats, method, k, embedding=emb)
            labels = np.full(len(df), -1, int); labels[use] = res.labels
            return res, labels, use

        def done(out):
            res, labels, use = out
            # 写回各细胞的事件表
            pos = 0
            for c in cells:
                m = c.mask(); n = int(m.sum())
                col = np.full(len(c.events), -1, int); col[m] = labels[pos:pos + n]; pos += n
                c.events["cluster"] = col
            self.cluster_msg.setText(res.message)
            if res.k >= 1:
                dfu, wu = df[use].reset_index(drop=True), waves[use]
                fig = plots.cluster_figure(res, dfu, wu, t_wave, direction)
                dur = sum(c.rec.duration for c in cells)
                summ = clustering.cluster_summary(dfu, res.labels, dur)
                txt = [res.message, f"参与聚类的事件 {int(res.valid.sum())} 个（{'全部细胞合并' if pooled else cells[0].name}）", "",
                       "各簇中位数：", summ.round(3).to_string(index=False, col_space=7), "",
                       "解读线索：", "· 幅度贴近噪声、衰减极短或电荷为负的一簇多半是误检的噪声，可在“事件审核”中按簇排序后排除，或提高检测阈值；",
                       "· 上升慢、衰减慢且幅度小的一簇 → 远端树突输入（电紧张滤波），可用“上升时间–幅度相关”佐证；",
                       "· 上升相近但衰减明显不同 → 受体亚基/类型不同（如 GluA2 缺失的快 AMPAR、含 NMDA 成分的慢 EPSC；α1 与 α2/3/5 型 GABA_A 受体），或不同来源的中间神经元（PV 胞体周围快 IPSC vs SST 树突慢 IPSC）；",
                       "· 做“全部细胞合并”聚类后，到“分析与统计 → 各簇组成的组间比较”检验处理是否选择性改变某一类输入。"]
                self.cluster_panel.show_result(fig, "\n".join(txt))
            self.refresh_trace(); self.refresh_events()
            if res.k < 1:
                self.cluster_panel.clear_result()
            self.archive_result(self.cluster_panel.fig, self.cluster_panel.text.toPlainText() or res.message,
                                dict(analysis="动力学聚类", pooled=pooled, representation=rep, method=method, k=k,
                                     embedding=emb, features=feats, exclude_overlap=noov), cells,
                                {f"labels_{index}": cell.events["cluster"].to_numpy() for index, cell in enumerate(cells)})

        self.run_bg(job, done, "正在聚类 …")

    # ================================================================ 分析
    def run_analysis(self):
        i = self.an.currentIndex()
        c = self.current
        need_cell = i <= 6
        if need_cell and (c is None or c.events is None or c.mask().sum() < 5):
            self.an_panel.text.setPlainText("请先选中一个已完成检测的细胞。"); return
        metric = self.an_metric.currentText()
        other = next((x for x in self.project.cells if x.name == self.an_other.currentText()), None)
        dfm = self.an_df.value()
        cdf_mode = ["cell_average", "pooled_equal", "pooled_all"][self.an_cdf.currentIndex()]

        def job():
            if i == 0: return plots.cell_overview(c)
            if i == 1: return plots.detection_qc(c)
            if i == 2: return plots.rise_amp_figure(c)
            if i == 3: return plots.temporal_figure(c)
            if i == 4: return plots.nsfa_figure(c, driving_force_mv=dfm)
            if i == 5: return plots.quantal_figure(c)
            if i == 6:
                if other is None or other.events is None or other is c:
                    return plots.new_figure(3, 2), "请在“I 细胞”中选择另一个已检测的记录（例如同一细胞在 0 mV 记录的 sIPSC）。"
                return plots.ei_figure(c, other)
            if len(self.project.analyzed()) < 2:
                return plots.new_figure(3, 2), "组间分析需要至少 2 个已检测的细胞，并在左侧列表填写组别。"
            if i == 7: return plots.group_comparison(self.project, metric)
            if i == 8: return plots.cumulative_figure(self.project, mode=cdf_mode)
            if i == 9: return plots.scaling_figure(self.project)
            return plots.cluster_composition(self.project)

        def done(out):
            self.qc_result_cell = c if i == 1 else None
            self.an_panel.show_result(*out)
            self.archive_result(*out, dict(analysis=self.ANALYSES[i], current=c.name if c else None, metric=metric,
                                other=other.name if other else None, driving_force_mv=dfm, cdf_mode=cdf_mode), self.project.cells)

        self.run_bg(job, done, f"正在运行：{self.an.currentText()} …")

    # ================================================================ 导出
    def export_tables(self):
        if not self.project.analyzed():
            QtWidgets.QMessageBox.information(self, "导出", "还没有检测结果。"); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出结果表（输入文件名前缀）", str(self.project_root / "results" / "results"), "CSV (*.csv)")
        if path:
            stem = path[:-4] if path.lower().endswith(".csv") else path
            files = self.project.export(stem)
            self.statusBar().showMessage("已导出：" + "，".join(os.path.basename(f) for f in files))

    def release_project_resources(self):
        self.task_timer.stop()
        self.current = self.qc_result_cell = self.last_error = None
        self.project = Project()
        self.project_root = None
        self.clear_deconv_view()
        self.cluster_panel.clear_result()
        self.an_panel.clear_result()
        self.refresh_trace()
        self.refresh_events()
        self.refresh_cells()
        gc.collect()

    def closeEvent(self, e):
        if self.workers:
            self.statusBar().showMessage("请在后台任务结束后关闭。")
            e.ignore(); return
        if self.dirty and not self.save_session():
            if QtWidgets.QMessageBox.question(self, "项目尚未保存", "要放弃未保存的更改并关闭吗？",
                                              QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                e.ignore(); return
        if self.project_lock is not None:
            self.project_lock.unlock(); self.project_lock = None
        self.release_project_resources()
        self.dirty = False
        super().closeEvent(e)
