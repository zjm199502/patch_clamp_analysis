import os

# 统一使用 PySide6（miniML 自带的界面依赖 PyQt5，可能同时装在环境里）
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
