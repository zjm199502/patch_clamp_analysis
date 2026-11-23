"""Exercise a signed Mac app with real file imports and deconvolution inside an isolated patch profile."""
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
    from PySide6.QtWidgets import QApplication as QtApplication, QMessageBox
    import faulthandler, time, traceback
    faulthandler.enable()
    root = Path(os.environ['HOME'])
    class ProbeApplication(QtApplication):
        def __init__(self, *args):
            assert "patchclamp_gui.gui.main_window" not in sys.modules
            assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "OpenBLAS cap missing at app construction"
            super().__init__(*args)
            QTimer.singleShot(100, self.exercise)

        def exercise(self):
            try:
                # main() has now set numerical thread policy and shown the real window.
                import numpy as np
                from threadpoolctl import threadpool_info
                from .main_window import MainWindow
                for pool in threadpool_info():
                    if pool["internal_api"] == "openblas":
                        assert pool["num_threads"] == 1, pool
                assert self.platformName() == 'cocoa'
                w = next(win for win in self.topLevelWidgets() if isinstance(win, MainWindow))
                assert w.activate_project(root / 'import-project', create=True)
                for n, name in enumerate(('sweeps.abf', 'long.h5'), 1):
                    print('importing', name, flush=True)
                    w.add_files([root / name])
                    assert len(w.project.cells) == n and w.current.name == name
                    self.processEvents()
                    assert w.c_raw.xData.size and w.save_session()
                assert w.activate_project(root / 'import-project')
                assert len(w.project.cells) == 2
                w.cell_table.selectRow(0)
                self.processEvents()
                assert w.d["hp"].value() == 1.0, "default deconvolution high-pass was not used"
                print("running default deconvolution in the Cocoa app worker", flush=True)
                w.run_deconv()
                assert len(w.workers) == 1, "deconvolution did not start one Worker"
                assert w.workers[0].stackSize() >= 8 * 1024 * 1024
                deadline = time.monotonic() + 30
                while w.workers and time.monotonic() < deadline:
                    self.processEvents()
                    time.sleep(0.01)
                self.processEvents()
                assert not w.workers, "deconvolution worker timed out"
                assert not w.last_error, w.last_error
                res = w.deconv_res
                assert res is not None and w.deconv_info.text()
                assert res.deconvolved.shape == res.reconvolved.shape == w.current.processed.shape
                assert np.isfinite(res.deconvolved).all() and np.isfinite(res.reconvolved).all()
                assert np.isfinite(res.threshold) and res.threshold > 0
                assert np.ptp(res.deconvolved) > 0 and len(res.peaks) > 0
                assert w.cd_dec.xData.size == res.deconvolved.size
                w.grab()
                (root / 'passed').touch()
                w.close()
                self.quit()
            except BaseException:
                traceback.print_exc()
                os._exit(2)
    def fail_dialog(*args):
        raise RuntimeError(str(args[2]))
    QMessageBox.warning = fail_dialog
    QMessageBox.critical = fail_dialog
    # Replace the constructor referenced by main() without importing MainWindow or NumPy early.
    globals()["QApplication"] = ProbeApplication
    _entry()
""")
        rng = np.random.default_rng(23)
        data = rng.normal(0, 0.35, 500_000)
        t = np.arange(500) / 20_000
        event = np.exp(-t / 0.004) - np.exp(-t / 0.0004)
        for onset in range(2000, len(data) - len(event), 5000):
            data[onset:onset + len(event)] -= 30 * event
        writeABF1(data.reshape(10, -1), str(root / "sweeps.abf"), 20000)
        with h5py.File(root / "long.h5", "w") as out:
            out.create_dataset("mini_data", data=np.tile(data, 8))
        env = dict(os.environ, HOME=str(root), QT_QPA_PLATFORM="cocoa", PYTHONFAULTHANDLER="1", OPENBLAS_NUM_THREADS="8")
        subprocess.run([bundle / "Contents/MacOS/PatchClampAnalysis"], env=env, cwd=root, check=True, timeout=100)
        assert (root / "passed").exists()
        print("Cocoa import, preview, save, reopen and Worker deconvolution passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args()
    run(args.bundle.resolve(), args.patch)
