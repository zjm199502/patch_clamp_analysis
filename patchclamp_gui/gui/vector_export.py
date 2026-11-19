from PySide6 import QtCore, QtGui, QtWidgets


def export_action(widget, render):
    action = QtGui.QAction("导出可编辑矢量图（SVG）…", widget)
    def save():
        path, _ = QtWidgets.QFileDialog.getSaveFileName(widget, "导出可编辑矢量图", "figure.svg", "SVG (*.svg)")
        if path:
            try:
                render(path if path.lower().endswith(".svg") else path + ".svg")
            except Exception as error:
                QtWidgets.QMessageBox.warning(widget, "导出失败", str(error))
    action.triggered.connect(save)
    return action


def figure_menu(canvas):
    from matplotlib import rc_context
    def render(path):
        with rc_context({"svg.fonttype": "none"}):
            canvas.figure.savefig(path, format="svg")
    canvas.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
    canvas.addAction(export_action(canvas, render))


def plot_menu(plot):
    from PySide6.QtSvg import QSvgGenerator
    def render(path):
        scene = plot.scene()
        source = plot.sceneBoundingRect()
        if source.isEmpty():
            raise ValueError("请先显示预览图。")
        target = QtCore.QRectF(0, 0, source.width(), source.height())
        generator = QSvgGenerator()
        generator.setFileName(path)
        generator.setSize(source.size().toSize())
        generator.setViewBox(target)
        painter = QtGui.QPainter(generator)
        if not painter.isActive():
            raise OSError("不能写入 SVG 文件")
        states = []
        try:
            for item in scene.items(source):
                if hasattr(item, "opts") and "useCache" in item.opts:
                    states.append((item, item.opts["useCache"]))
                    item.opts["useCache"] = False
            scene.render(painter, target, source)
        finally:
            painter.end()
            for item, previous in states:
                item.opts["useCache"] = previous
    menu = plot.getViewBox().getMenu(None)
    menu.addAction(export_action(menu, render))
