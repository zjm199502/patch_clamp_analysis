"""Small in-app updates ("patches") for the packaged app.

A packaged build ("base") contains Python, all libraries and a copy of the
patchclamp_gui package. Most new versions only change patchclamp_gui itself, so
they are shipped as a small zip holding just that package. It is unpacked to
the user's data folder and put in front of the bundled copy on sys.path at
start-up. A patch is used only when:
    * its version is newer than the base build, and
    * the base build is at least the patch's ``min_base`` (no library changes
      since that build).

This module must import nothing but the standard library: the launcher calls
it before patchclamp_gui is imported.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

BASE_FILE = "base_version.txt"   # written next to the bundle's Python files at build time
CURRENT = "current.json"


def parse_version(v: str) -> tuple[int, ...]:
    nums = [int(n) for n in re.findall(r"\d+", str(v).split("+")[0])[:3]]
    return tuple(nums) + (0,) * (3 - len(nums))


def data_dir() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return root / "PatchClampAnalysis"


def patch_dir() -> Path:
    return data_dir() / "patches"


def base_version() -> str | None:
    """Version of the packaged build that is running (None when run from source)."""
    if not getattr(sys, "frozen", False):
        return None
    for d in (getattr(sys, "_MEIPASS", None), Path(sys.executable).parent):
        if d and (Path(d) / BASE_FILE).exists():
            return (Path(d) / BASE_FILE).read_text(encoding="utf-8").strip()
    return None


def read_current() -> dict | None:
    try:
        return json.loads((patch_dir() / CURRENT).read_text(encoding="utf-8"))
    except Exception:
        return None


def usable(info: dict | None, base: str | None) -> bool:
    if not info or not base or info.get("bad"):
        return False
    try:
        return (parse_version(info["version"]) > parse_version(base)
                and parse_version(base) >= parse_version(info.get("min_base", "0"))
                and (Path(info["path"]) / "patchclamp_gui" / "__init__.py").exists())
    except Exception:
        return False


def activate() -> str | None:
    """Put an installed patch in front of the bundled package. Returns its version."""
    base = base_version()
    info = read_current()
    if not usable(info, base):
        return None
    sys.path.insert(0, str(info["path"]))
    os.environ["PATCHCLAMP_PATCH_VERSION"] = str(info["version"])
    return str(info["version"])


def mark_bad(version: str) -> None:
    """Called when a patched start crashes: the next start uses the bundled code."""
    info = read_current()
    if info and info.get("version") == version:
        info["bad"] = True
        (patch_dir() / CURRENT).write_text(json.dumps(info), encoding="utf-8")


def install(src_pkg_parent: Path, version: str, min_base: str) -> Path:
    """Move an unpacked patch (a folder containing patchclamp_gui/) into place."""
    import shutil
    dest = patch_dir() / version
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_pkg_parent), str(dest))
    tmp = patch_dir() / (CURRENT + ".tmp")
    tmp.write_text(json.dumps({"version": version, "min_base": min_base, "path": str(dest)}),
                   encoding="utf-8")
    os.replace(tmp, patch_dir() / CURRENT)
    # keep only the active patch
    for p in patch_dir().iterdir():
        if p.is_dir() and p != dest:
            shutil.rmtree(p, ignore_errors=True)
    return dest
