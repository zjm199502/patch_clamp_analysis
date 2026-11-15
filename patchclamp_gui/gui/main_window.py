"""主窗口：文件列表 + 五个功能页（记录与检测 / 事件审核 / 去卷积平滑 / 动力学聚类 / 分析与统计）。"""
from __future__ import annotations

import os
import traceback

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__, io
from ..core import clustering, deconv, detection, plots
from ..core.features import FEATURES
from ..core.session import CellSession, Project
from .update_ui import UpdateManager

Qt = QtCore.Qt
CLUSTER_COLORS = [QtGui.QColor(c) for c in plots.PALETTE]


class Worker(QtCore.QThread):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception:
            self.failed.emit(traceback.format_exc())


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

    def show_result(self, fig, text=""):
        if self.canvas is not None:
            self.hl.removeWidget(self.canvas); self.canvas.setParent(None); self.canvas.deleteLater()
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
            self, "导出图", "figure.pdf", "PDF (*.pdf);;SVG (*.svg);;PNG 600 dpi (*.png);;TIFF 600 dpi (*.tiff)")
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
        self.setWindowTitle(f"膜片钳分析 PatchClampAnalysis v{__version__} — m/sEPSC · m/sIPSC")
        self.resize(1500, 900)
        self.project = Project()
        self.current: CellSession | None = None
        self.workers: list[Worker] = []
        self.deconv_res = None
        self.last_error = None
        self.settings = QtCore.QSettings("PatchClampAnalysis", "PatchClampAnalysis")
        self._build_menu()
        self._build_ui()
        self.statusBar().showMessage("打开 ABF 文件开始，或用 文件 → 载入模拟演示数据 体验完整流程。")

    # ================================================================ 界面
    def _build_menu(self):
        m = self.menuBar().addMenu("文件")
        for text, fn, key in (("打开记录（ABF / H5）…", self.open_files, "Ctrl+O"), ("载入模拟演示数据", self.load_demo, None),
                              (None, None, None), ("导出结果表（事件 + 细胞汇总 CSV）…", self.export_tables, "Ctrl+E"),
                              (None, None, None), ("退出", self.close, None)):
            if text is None:
                m.addSeparator(); continue
            act = QtGui.QAction(text, self); act.triggered.connect(lambda _=False, f=fn: f())
            if key:
                act.setShortcut(key)
            m.addAction(act)
        self.updates = UpdateManager(self)      # “帮助”菜单：检查更新 / 自动检查 / 关于

    def _build_ui(self):
        # ---- 左侧：文件/细胞列表 ----
        dock = QtWidgets.QDockWidget("细胞列表（可直接编辑组别与动物编号）", self)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        self.cell_table = QtWidgets.QTableWidget(0, 5)
        self.cell_table.setHorizontalHeaderLabels(["文件", "组别", "动物", "事件数", "频率 Hz"])
        self.cell_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.cell_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.cell_table.itemSelectionChanged.connect(self.on_cell_selected)
        self.cell_table.itemChanged.connect(self.on_cell_edited)
        v.addWidget(self.cell_table)
        row = QtWidgets.QHBoxLayout()
        for label, fn in (("添加文件", self.open_files), ("移除", self.remove_cell)):
            b = QtWidgets.QPushButton(label); b.clicked.connect(fn); row.addWidget(b)
        v.addLayout(row)
        dock.setWidget(w); dock.setMinimumWidth(300)
        hh = self.cell_table.horizontalHeader()
        for j in range(1, 5):
            hh.setSectionResizeMode(j, QtWidgets.QHeaderView.ResizeToContents)
        self.cell_table.verticalHeader().setVisible(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.resizeDocks([dock], [430], Qt.Horizontal)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._tab_detect(), "① 记录与检测")
        self.tabs.addTab(self._tab_events(), "② 事件审核")
        self.tabs.addTab(self._tab_deconv(), "③ 去卷积平滑")
        self.tabs.addTab(self._tab_cluster(), "④ 动力学聚类")
        self.tabs.addTab(self._tab_analysis(), "⑤ 分析与统计")
        self.progress = QtWidgets.QProgressBar(); self.progress.setMaximumWidth(180); self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

    def _tab_detect(self):
        w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w)
        form_w = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(form_w)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.p = {}
        self.preset = QtWidgets.QComboBox(); self.preset.addItems(list(detection.PRESETS))
        self.preset.currentTextChanged.connect(self.apply_preset)
        form.addRow("事件类型预设", self.preset)
        self.p["direction"] = QtWidgets.QComboBox(); self.p["direction"].addItems(["negative", "positive"])
        form.addRow("事件方向", self.p["direction"])
        self.p["window_ms"] = _spin(30, 2, 1000, 5, 1, "ms"); form.addRow("事件窗口", self.p["window_ms"])
        self.p["tau_rise_ms"] = _spin(0.4, 0.02, 50, 0.1, 2, "ms"); form.addRow("模板 τ上升", self.p["tau_rise_ms"])
        self.p["tau_decay_ms"] = _spin(4, 0.1, 500, 1, 2, "ms"); form.addRow("模板 τ衰减", self.p["tau_decay_ms"])
        form.addRow(QtWidgets.QLabel("<b>预处理</b>"))
        self.p["lowpass_hz"] = _spin(2000, 0, 20000, 100, 0, "Hz"); form.addRow("低通（0=关）", self.p["lowpass_hz"])
        self.p["highpass_hz"] = _spin(0, 0, 100, 0.5, 1, "Hz"); form.addRow("高通（0=关）", self.p["highpass_hz"])
        self.p["notch_hz"] = _spin(0, 0, 60, 50, 0, "Hz"); form.addRow("工频陷波（0/50/60）", self.p["notch_hz"])
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
                          ("批量检测全部细胞", self.run_detect_all)):
            b = QtWidgets.QPushButton(label); b.clicked.connect(fn); form.addRow(b)
        self.info = QtWidgets.QLabel(""); self.info.setWordWrap(True); form.addRow(self.info)
        scroll = QtWidgets.QScrollArea(); scroll.setWidget(form_w); scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        h.addWidget(scroll)

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
        h.addWidget(self.glw, 1)
        return w

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
        w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w)
        fw = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(fw); fw.setFixedWidth(330)
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
        h.addWidget(fw)
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
        h.addWidget(g, 1)
        return w

    def _tab_cluster(self):
        w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w)
        fw = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(fw); fw.setFixedWidth(330)
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
        self.cluster_msg = QtWidgets.QLabel(""); self.cluster_msg.setWordWrap(True); form.addRow(self.cluster_msg)
        h.addWidget(fw)
        self.cluster_panel = FigurePanel(); h.addWidget(self.cluster_panel, 1)
        return w

    ANALYSES = ["细胞总览", "检测质控（注入 / 反极性 / 阈值扫描）", "上升时间–幅度相关（树突滤波诊断）",
                "事件时间依赖性（自相关 / Hawkes）", "峰值缩放非平稳噪声分析（PS-NSFA）", "量子多峰分析",
                "E/I 联合分析（当前细胞=E，另选一个=I）",
                "组间比较（层级统计）", "累积概率分布（幅度 / IEI）", "突触缩放（秩序图 / 比例图）", "各簇组成的组间比较"]

    def _tab_analysis(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
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
        grid.setColumnStretch(1, 3); grid.setColumnStretch(3, 2)
        for cb in (self.an, self.an_metric, self.an_cdf, self.an_other, self.an_df):
            cb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        row.addLayout(grid)
        v.addLayout(row)
        self.an_panel = FigurePanel(); v.addWidget(self.an_panel, 1)
        return w

    # ================================================================ 细胞管理
    def open_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "打开记录", self.settings.value("last_dir", ""),
                                                          "记录文件 (*.abf *.h5 *.hdf5 *.npy);;所有文件 (*)")
        if paths:
            self.settings.setValue("last_dir", os.path.dirname(paths[0]))
            self.add_files(paths)

    def add_files(self, paths):
        for p in paths:
            try:
                ch = 0
                if p.lower().endswith(".abf"):
                    inf = io.abf_info(p)
                    if len(inf["channels"]) > 1:
                        items = [f"{c}: {n} ({u})" for c, n, u in zip(inf["channels"], inf["channel_names"], inf["units"])]
                        it, ok = QtWidgets.QInputDialog.getItem(self, "选择通道", os.path.basename(p), items, 0, False)
                        if not ok:
                            continue
                        ch = inf["channels"][items.index(it)]
                self.add_cell(CellSession(io.load_any(p, channel=ch)))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "读取失败", f"{p}\n{e}")

    def add_cell(self, cell: CellSession):
        cell.params.update(self.read_params())
        self.project.cells.append(cell)
        self.refresh_cells()
        self.cell_table.selectRow(len(self.project.cells) - 1)

    def load_demo(self):
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
        self.statusBar().showMessage("已载入 8 个模拟细胞（对照 4 个、处理 4 个；处理组幅度乘性放大 1.4 倍）。点击“批量检测全部细胞”。")

    def remove_cell(self):
        r = self.cell_table.currentRow()
        if 0 <= r < len(self.project.cells):
            del self.project.cells[r]
            self.current = None
            self.refresh_cells()

    def refresh_cells(self):
        self.cell_table.blockSignals(True)
        self.cell_table.setRowCount(len(self.project.cells))
        for i, c in enumerate(self.project.cells):
            n = len(c.valid()[0]) if c.events is not None else None
            vals = [c.name, c.group, c.animal, "" if n is None else str(n),
                    "" if n is None else f"{n / c.rec.duration:.2f}"]
            for j, val in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(val)
                if j not in (1, 2):
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.cell_table.setItem(i, j, it)
        self.cell_table.blockSignals(False)
        cur = self.an_other.currentText()
        self.an_other.clear(); self.an_other.addItems([c.name for c in self.project.cells]); self.an_other.setCurrentText(cur)

    def on_cell_edited(self, it):
        c = self.project.cells[it.row()]
        if it.column() == 1:
            c.group = it.text().strip()
        elif it.column() == 2:
            c.animal = it.text().strip()

    def on_cell_selected(self):
        r = self.cell_table.currentRow()
        if not (0 <= r < len(self.project.cells)):
            return
        self.current = self.project.cells[r]
        self.write_params(self.current.params)
        self.deconv_res = None
        self.refresh_trace(reset=True)
        self.refresh_events()

    # ================================================================ 参数
    def apply_preset(self, name):
        for k, val in detection.PRESETS[name].items():
            wd = self.p[k]
            wd.setCurrentText(val) if isinstance(wd, QtWidgets.QComboBox) else wd.setValue(val)

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
    def run_bg(self, fn, on_done, msg):
        self.statusBar().showMessage(msg)
        self.progress.setRange(0, 0); self.progress.show()
        wk = Worker(fn, self)
        self.workers.append(wk)

        def finish(res):
            self.progress.hide(); self.workers.remove(wk); on_done(res)

        def fail(tb):
            self.progress.hide(); self.workers.remove(wk)
            self.statusBar().showMessage("出错")
            self.last_error = tb
            if os.environ.get("PATCHCLAMP_NO_DIALOGS"):      # 自动测试时不弹模态框
                return
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "出错", tb.strip().splitlines()[-1], parent=self)
            box.setDetailedText(tb); box.exec()

        wk.done.connect(finish); wk.failed.connect(fail)
        wk.start()
        return wk

    # ================================================================ 检测
    def run_preprocess(self):
        c = self.current
        if c is None:
            return
        c.params.update(self.read_params())
        c.run_preprocess()
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
        params = self.read_params()
        cells = list(self.project.cells)
        if not cells:
            return

        def job():
            tot = 0
            for c in cells:
                c.params.update(params)
                c.run_preprocess()
                tot += c.run_detection()
            return tot

        self.run_bg(job, lambda n: self._detected(f"批量检测完成：{len(cells)} 个细胞，共 {n} 个事件"),
                    f"正在批量检测 {len(cells)} 个细胞 …")

    def _detected(self, msg):
        self.statusBar().showMessage(msg)
        row = self.cell_table.currentRow()
        self.refresh_cells()
        if row >= 0:
            self.cell_table.blockSignals(True); self.cell_table.selectRow(row); self.cell_table.blockSignals(False)
        self.refresh_trace()
        self.refresh_events()

    def refresh_trace(self, reset=False):
        c = self.current
        if c is None:
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
        self.refresh_events(); self.refresh_trace(); self.refresh_cells_keep()
        self.ev_table.selectRow(min(rows[-1] + 1, len(c.events) - 1))

    def refresh_cells_keep(self):
        row = self.cell_table.currentRow()
        self.refresh_cells()
        self.cell_table.blockSignals(True); self.cell_table.selectRow(row); self.cell_table.blockSignals(False)

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
            self.deconv_res = res
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
            self.statusBar().showMessage("去卷积完成")
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
            self.statusBar().showMessage("聚类完成")

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
            self.an_panel.show_result(*out)
            self.statusBar().showMessage("分析完成")

        self.run_bg(job, done, f"正在运行：{self.an.currentText()} …")

    # ================================================================ 导出
    def export_tables(self):
        if not self.project.analyzed():
            QtWidgets.QMessageBox.information(self, "导出", "还没有检测结果。"); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出结果表（输入文件名前缀）", "results", "CSV (*.csv)")
        if path:
            stem = path[:-4] if path.lower().endswith(".csv") else path
            files = self.project.export(stem)
            self.statusBar().showMessage("已导出：" + "，".join(os.path.basename(f) for f in files))

    def closeEvent(self, e):
        for wk in list(self.workers):
            wk.wait(3000)
        super().closeEvent(e)
