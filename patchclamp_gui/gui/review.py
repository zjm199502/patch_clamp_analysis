from pathlib import Path

from PySide6 import QtWidgets as W
from PySide6.QtCore import Qt

from ..core import detection


def preview_import(parent, paths, groups):
    dialog = W.QDialog(parent)
    dialog.setWindowTitle("导入预览：检查分组与事件类型")
    layout = W.QVBoxLayout(dialog)
    table = W.QTableWidget(len(paths), 4)
    table.setHorizontalHeaderLabels(["文件", "分组（可编辑）", "事件类型", "识别依据"])
    table.setSelectionBehavior(W.QAbstractItemView.SelectRows)
    options = ["使用默认参数", *detection.PRESETS]
    for index, path in enumerate(paths):
        source = Path(path)
        matches = [part for part in (source.stem, *(p.name for p in source.parents))
                   if "epsc" in part.lower() or "ipsc" in part.lower()]
        keywords = {key for part in matches for key in ("epsc", "ipsc") if key in part.lower()}
        reason = " → ".join(matches) or "无关键词"
        if len(keywords) == 2:
            reason = "⚠ 命名冲突：" + reason
        for col, value in ((0, source.name), (1, groups.get(path, "")), (3, reason)):
            item = W.QTableWidgetItem(value)
            if col != 1:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setToolTip(str(source) if col == 0 else reason)
            table.setItem(index, col, item)
        choice = W.QComboBox()
        choice.addItems(options)
        choice.setCurrentText(detection.preset_from_path(path) or options[0])
        table.setCellWidget(index, 2, choice)
    layout.addWidget(table)
    batch = W.QComboBox()
    batch.addItems(options)
    layout.addWidget(batch)
    apply = W.QPushButton("应用此事件类型到选中行")
    apply.clicked.connect(lambda: [table.cellWidget(idx.row(), 2).setCurrentText(batch.currentText())
                                   for idx in table.selectionModel().selectedRows()])
    layout.addWidget(apply)
    buttons = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    table.resizeColumnsToContents()
    dialog.resize(850, 450)
    accepted = dialog.exec()
    result = ({p: table.item(i, 1).text() for i, p in enumerate(paths)},
              {p: table.cellWidget(i, 2).currentText() if table.cellWidget(i, 2).currentIndex() else None
               for i, p in enumerate(paths)}) if accepted else None
    dialog.deleteLater()
    return result


def review_sweeps(parent, path, result, report, folder):
    import numpy as np
    import pyabf
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    abf = pyabf.ABF(str(path))
    dialog = W.QDialog(parent)
    dialog.setWindowTitle("逐 sweep 复核 · " + Path(path).name)
    layout = W.QVBoxLayout(dialog)
    figure = Figure(layout="constrained")
    fi, voltage, command = figure.subplots(3)
    canvas = FigureCanvasQTAgg(figure)
    from .vector_export import figure_menu
    figure_menu(canvas)
    layout.addWidget(canvas)
    rows = result["rows"]
    fi.plot([row["current_pA"] for row in rows], [row["rate_Hz"] for row in rows], "o", picker=6)
    fi.set(xlabel="Current (pA)", ylabel="Rate (Hz)")
    choice = W.QComboBox()
    choice.addItems([f"Sweep {row['sweep']} · {row['current_pA']:g} pA · {row['count']} AP" for row in rows])
    layout.addWidget(choice)
    def draw(index):
        row = rows[index]
        abf.setSweep(row["sweep"] - 1)
        start = result["start_s"]
        voltage.clear(); command.clear()
        voltage.plot(abf.sweepX, abf.sweepY, linewidth=.8)
        peaks = np.rint((start + np.asarray(row["peak_ms"])/1000) * abf.dataRate).astype(int)
        voltage.plot(abf.sweepX[peaks], abf.sweepY[peaks], "r.")
        command.plot(abf.sweepX, abf.sweepC)
        for axis in (voltage, command):
            axis.axvspan(start, start+result["duration_s"], alpha=.12)
        voltage.set_ylabel("Voltage (mV)")
        command.set(xlabel="Time (s)", ylabel="Command (pA)")
        canvas.draw_idle()
    choice.currentIndexChanged.connect(draw)
    connection = canvas.mpl_connect("pick_event", lambda event: choice.setCurrentIndex(int(event.ind[0])) if len(event.ind) else None)
    layout.addWidget(W.QLabel(f"点击 F–I 数据点或选择 sweep。报告已保存：{folder.name}"))
    draw(0)
    dialog.resize(850, 680)
    dialog.exec()
    canvas.mpl_disconnect(connection)
    dialog.deleteLater()
