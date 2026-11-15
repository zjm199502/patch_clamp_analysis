"""Menu entry, automatic check at start-up and the update dialog."""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QObject, QSettings, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from .. import __version__, updater

CHECK_INTERVAL_S = 24 * 3600


class _Bridge(QObject):
    checked = Signal(object, object, bool)    # release | None, error | None, manual
    progress = Signal(float)
    prepared = Signal(object, object)          # new_dir | None, error | None


class UpdateManager(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.win = window
        self.settings = QSettings("PatchClampAnalysis", "PatchClampAnalysis")
        self.bridge = _Bridge()
        self.bridge.checked.connect(self._on_checked)
        self.bridge.progress.connect(self._on_progress)
        self.bridge.prepared.connect(self._on_prepared)
        self._busy = False
        self._dlg = None
        self._cancel = False
        updater.cleanup_stale()

        menu = window.menuBar().addMenu("帮助")
        act = QAction("检查更新…", window)
        act.triggered.connect(lambda: self.check(manual=True))
        menu.addAction(act)
        self.auto_act = QAction("启动时自动检查更新", window, checkable=True)
        self.auto_act.setChecked(self.auto_enabled())
        self.auto_act.toggled.connect(lambda v: self.settings.setValue("update/auto", bool(v)))
        menu.addAction(self.auto_act)
        about = QAction("关于", window)
        about.triggered.connect(self._about)
        menu.addAction(about)

    # ---------------------------------------------------------------- check
    def auto_enabled(self) -> bool:
        return str(self.settings.value("update/auto", "true")).lower() in ("true", "1")

    def maybe_check_on_start(self):
        if not self.auto_enabled():
            return
        last = float(self.settings.value("update/last_check", 0) or 0)
        if time.time() - last < CHECK_INTERVAL_S:
            return
        self.check(manual=False)

    def check(self, manual: bool):
        if self._busy:
            return
        self._busy = True

        def run():
            try:
                rel, err = updater.check_latest(), None
            except Exception as e:  # offline, rate-limited, ...
                rel, err = None, e
            self.bridge.checked.emit(rel, err, manual)
        threading.Thread(target=run, daemon=True).start()

    def _on_checked(self, rel, err, manual):
        self._busy = False
        if err is None:
            self.settings.setValue("update/last_check", time.time())
        if err is not None:
            if manual:
                QMessageBox.warning(self.win, "检查更新", f"无法连接更新服务器：\n{err}")
            return
        if rel is None:
            if manual:
                QMessageBox.information(self.win, "检查更新", f"已经是最新版本（v{__version__}）。")
            return
        skipped = self.settings.value("update/skip", "")
        if not manual and skipped == rel.version:
            return
        self._offer(rel)

    def _offer(self, rel):
        notes = rel.notes.strip() or "（没有更新说明）"
        if len(notes) > 1500:
            notes = notes[:1500] + "…"
        box = QMessageBox(self.win)
        box.setWindowTitle("发现新版本")
        box.setText(f"有新版本 v{rel.version}（当前 v{__version__}）。")
        box.setInformativeText(notes)
        can_auto = updater.install_root() is not None and rel.asset_url is not None and rel.kind != "page"
        b_now = box.addButton("立即更新" if can_auto else "打开下载页面", QMessageBox.AcceptRole)
        b_skip = box.addButton("跳过这个版本", QMessageBox.RejectRole)
        box.addButton("以后再说", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is b_skip:
            self.settings.setValue("update/skip", rel.version)
        elif box.clickedButton() is b_now:
            if can_auto:
                self._install(rel)
            else:
                QDesktopServices.openUrl(QUrl(rel.url))

    # -------------------------------------------------------------- install
    def _install(self, rel):
        self._rel = rel
        if getattr(self.win, "workers", None):
            QMessageBox.information(self.win, "更新", "请等当前计算完成后再更新。")
            return
        self._cancel = False
        mb = rel.asset_size / 1e6 if rel.asset_size else 0
        if rel.kind == "patch" and mb < 1:
            mb = 0
        self._dlg = QProgressDialog(f"正在下载 v{rel.version}" + (f"（{mb:.0f} MB）" if mb else "") + "…",
                                    "取消", 0, 1000, self.win)
        self._dlg.setWindowTitle("更新")
        self._dlg.setWindowModality(Qt.WindowModal)
        self._dlg.setMinimumDuration(0)
        self._dlg.canceled.connect(self._on_cancel)
        self._dlg.show()

        def run():
            try:
                d = updater.prepare_update(rel, self.bridge.progress.emit, lambda: self._cancel)
                self.bridge.prepared.emit(d, None)
            except Exception as e:
                self.bridge.prepared.emit(None, e)
        threading.Thread(target=run, daemon=True).start()

    def _on_cancel(self):
        self._cancel = True

    def _on_progress(self, f):
        if self._dlg is not None:
            self._dlg.setValue(int(f * 1000))

    def _on_prepared(self, new_dir, err):
        if self._dlg is not None:
            self._dlg.close()
            self._dlg = None
        if err is not None:
            if not isinstance(err, InterruptedError):
                QMessageBox.warning(self.win, "更新失败", f"{err}\n\n可以到发布页面手动下载：\n{updater.RELEASES_PAGE}")
            return
        small = getattr(self, "_rel", None) is not None and self._rel.kind == "patch"
        msg = ("更新已安装，重新打开软件后生效。现在重新打开吗？\n" if small else
               "新版本已下载。软件将关闭并自动重新打开，大约需要半分钟。\n现在更新吗？\n")
        if QMessageBox.question(self.win, "更新", msg + "你的分析结果保存在输出文件夹里，不受影响。") != QMessageBox.Yes:
            return
        try:
            if small:
                updater.restart_app()
            else:
                updater.launch_installer(new_dir)
        except Exception as e:
            QMessageBox.warning(self.win, "更新失败", str(e))
            return
        QApplication.quit()

    def _about(self):
        from .. import patching
        base = patching.base_version()
        build = f"（安装包 v{base}，已通过软件内更新升级）" if base and base != __version__ else ""
        QMessageBox.about(self.win, "关于",
                          f"膜片钳分析 PatchClampAnalysis v{__version__}{build}\n\n"
                          "m/sEPSC 与 m/sIPSC 的检测、质控、动力学聚类、去卷积与统计分析。\n"
                          "事件检测使用 miniML（O'Neill et al., 2025, eLife 13:RP98485）。\n\n"
                          f"源代码与更新：https://github.com/{updater.REPO}")
