import os
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication


def main():
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=False, background="w", foreground="k")
    app = QApplication(sys.argv)
    # Fusion + 浅色调色板：在所有电脑上外观一致，避免 Windows 深色模式下文字看不清
    app.setStyle("Fusion")
    app.setPalette(app.style().standardPalette())
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    except AttributeError:
        pass
    f = QFont()
    f.setFamilies(["Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "sans-serif"])
    f.setPointSize(10)
    app.setFont(f)
    from .main_window import MainWindow

    w = MainWindow()
    w.show()
    paths = [p for p in sys.argv[1:] if os.path.exists(p)]
    if paths:
        from ..core.project_files import SNAPSHOT
        if os.path.isdir(paths[0]) or os.path.basename(paths[0]) == SNAPSHOT:
            target = paths.pop(0)
            w.activate_project(target if os.path.isdir(target) else os.path.dirname(os.path.abspath(target)))
        else:
            w.new_project()
        if w.project_root:
            w.add_files(paths)
    QTimer.singleShot(3000, w.updates.maybe_check_on_start)
    sys.exit(app.exec())
