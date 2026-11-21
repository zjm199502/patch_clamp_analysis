"""Build a patch readable by the original 0.3.0 in-app updater."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZIP_DEFLATED, ZipFile


def add_directml(patch, windows_zip):
    prefix = "PatchClampAnalysis/_internal/ClampTraceDirectML/"
    required = {"ClampTraceDirectML.exe", "_internal/python311.dll",
                "_internal/onnxruntime/capi/DirectML.dll",
                "_internal/onnxruntime/capi/onnxruntime_pybind11_state.pyd"}
    with ZipFile(windows_zip) as source:
        entries = {}
        for info in source.infolist():
            path = info.filename.replace("\\", "/")
            if path.startswith(prefix) and not path.endswith("/"):
                name = path[len(prefix):]
                if not name or PurePosixPath(name).is_absolute() or ".." in name.split("/") or name in entries:
                    raise ValueError(f"Invalid runtime member: {name!r}")
                entries[name] = info
        if missing := required.difference(entries):
            raise ValueError(f"Windows archive is missing DirectML files: {sorted(missing)}")
        for name, info in sorted(entries.items()):
            with source.open(info) as src, patch.open("runtimes/ClampTraceDirectML/" + name, "w") as dest:
                shutil.copyfileobj(src, dest, 1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--windows-zip", type=Path)
    args = parser.parse_args()
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as patch:
        for source in sorted(Path("patchclamp_gui").rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc":
                patch.write(source, source.as_posix())
        if args.windows_zip:
            add_directml(patch, args.windows_zip)


if __name__ == "__main__":
    main()
