"""Entry point for the packaged program (Windows and macOS)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # in-app updates: use a newer patchclamp_gui from the user's data folder if one is installed
    from patchclamp_gui import patching   # stdlib-only module
    patched = None
    if getattr(sys, "frozen", False):
        for name in [m for m in sys.modules if m == "patchclamp_gui" or m.startswith("patchclamp_gui.")]:
            del sys.modules[name]          # re-import from the patch, if any
        patched = patching.activate()
    try:
        from patchclamp_gui.gui.app import main
        main()
    except SystemExit:
        raise
    except BaseException:
        if patched:
            patching.mark_bad(patched)     # next start falls back to the bundled version
        raise
