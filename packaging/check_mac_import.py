"""Exercise a signed Mac app with real file imports inside an isolated patch profile."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from zipfile import ZipFile

import h5py
import numpy as np
from pyabf.abfWriter import writeABF1


def run(bundle, archive):
    with tempfile.TemporaryDirectory(prefix="clamptrace-import-") as work:
        root = Path(work)
        patches = root / "Library/Application Support/PatchClampAnalysis/patches"
        code = patches / "probe"
        if archive:
            with ZipFile(archive) as z:
                for name in z.namelist():
                    if name.startswith("patchclamp_gui/"):
                        z.extract(name, code)
        else:
            shutil.copytree("patchclamp_gui", code / "patchclamp_gui", ignore=shutil.ignore_patterns("__pycache__"))
        # Instrument only the test profile, leaving the signed bundle intact.
        (patches / "current.json").write_text(json.dumps(dict(version="999.0.1", min_base="0", path=str(code))))
        appfile = code / "patchclamp_gui/gui/app.py"
        with appfile.open("a") as f:
            f.write("""

_entry = main
def main():
    from .main_window import MainWindow
    from PySide6.QtWidgets import QMessageBox
    import faulthandler, traceback
    faulthandler.enable()
    root = Path(os.environ['HOME'])
    orig = MainWindow.show
    def show(w):
        orig(w)
        def exercise():
            try:
                assert QApplication.platformName() == 'cocoa'
                assert w.activate_project(root / 'import-project', create=True)
                for n, name in enumerate(('sweeps.abf', 'long.h5'), 1):
                    print('importing', name, flush=True)
                    w.add_files([root / name])
                    assert len(w.project.cells) == n and w.current.name == name
                    QApplication.processEvents()
                    assert w.c_raw.xData.size and w.save_session()
                assert w.activate_project(root / 'import-project')
                assert len(w.project.cells) == 2
                w.cell_table.selectRow(0)
                QApplication.processEvents()
                w.grab()
                (root / 'passed').touch()
                w.close()
                QApplication.instance().quit()
            except BaseException:
                traceback.print_exc()
                os._exit(2)
        def fail_dialog(*args):
            raise RuntimeError(str(args[2]))
        QMessageBox.warning = fail_dialog
        QMessageBox.critical = fail_dialog
        QTimer.singleShot(100, exercise)
    MainWindow.show = show
    _entry()
""")
        data = np.sin(np.arange(500_000, dtype=float) / 128)
        writeABF1(data.reshape(10, -1), str(root / "sweeps.abf"), 20000)
        with h5py.File(root / "long.h5", "w") as out:
            out.create_dataset("mini_data", data=np.tile(data, 8))
        env = dict(os.environ, HOME=str(root), QT_QPA_PLATFORM="cocoa", PYTHONFAULTHANDLER="1")
        subprocess.run([bundle / "Contents/MacOS/PatchClampAnalysis"], env=env, cwd=root, check=True, timeout=100)
        assert (root / "passed").exists()
        print("Cocoa import, preview, save and reopen passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args()
    run(args.bundle.resolve(), args.patch)
