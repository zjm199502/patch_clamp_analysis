"""Check for a newer version and install it from inside the app.

Update channel: a single pre-release named "updates" on GitHub holds
    latest.json                          what the newest version is and how to get it
    patchclamp_gui-patch.zip              small update: just the patchclamp_gui package
    PatchClampAnalysis-*.zip              full builds, needed when libraries change
Major versions (x.y.0; the version after x.y.10 is x.(y+1).0) are also published as a normal
GitHub Release with installers.

Full build assets (built by .github/workflows/release.yml):
    PatchClampAnalysis-Windows.zip        the app folder "PatchClampAnalysis/"
    PatchClampAnalysis-macOS-arm64.zip    "PatchClampAnalysis.app" (made with ditto)
    PatchClampAnalysis-macOS-arm64.dmg    for first-time installs

Installing works only for the packaged app: the new build is downloaded and
unpacked next to the old one, then a small helper script waits for this
process to exit, swaps the folders and starts the new version.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import __version__

REPO = "zjm199502/patch_clamp_analysis"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
CHANNEL_MANIFEST = f"https://github.com/{REPO}/releases/download/updates/latest.json"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
UA = {"User-Agent": "PatchClampAnalysis-updater", "Accept": "application/vnd.github+json"}


@dataclass
class Release:
    version: str
    notes: str
    url: str                 # release page
    asset_url: str | None    # update package for this platform (None = not available)
    asset_size: int = 0
    kind: str = "full"       # "patch" (small, in place), "full" (whole app) or "page" (manual)
    min_base: str = "0"


def parse_version(v: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", v.split("+")[0])
    return tuple(int(n) for n in nums[:3]) + (0,) * (3 - len(nums[:3]))


def is_newer(remote: str, local: str = __version__) -> bool:
    return parse_version(remote) > parse_version(local)


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def asset_name() -> str | None:
    if sys.platform == "win32":
        return "PatchClampAnalysis-Windows.zip"
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "PatchClampAnalysis-macOS-arm64.zip"
    return None


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def install_root() -> Path | None:
    """Folder (Windows) or .app bundle (macOS) of the running packaged app."""
    if not is_frozen():
        return None
    exe = Path(sys.executable).resolve()
    if sys.platform == "darwin":
        for p in exe.parents:
            if p.suffix == ".app":
                return p
        return None
    return exe.parent


def _get_json(url: str, timeout: float):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        return json.load(r)


def check_latest(timeout: float = 8.0) -> Release | None:
    """Newest available version, or None when the running one is up to date."""
    from . import patching
    try:
        m = _get_json(CHANNEL_MANIFEST, timeout)
    except Exception:
        m = None
    if m is None:          # no channel yet: fall back to the newest GitHub Release
        data = _get_json(API_LATEST, timeout)
        want = asset_name()
        full = {a["name"]: (a["browser_download_url"], int(a.get("size", 0)))
                for a in data.get("assets", [])}
        m = {"version": str(data.get("tag_name", "")).lstrip("vV"), "notes": data.get("body") or "",
             "page": data.get("html_url") or RELEASES_PAGE,
             "full_version": str(data.get("tag_name", "")).lstrip("vV"),
             "full": {want: {"url": full[want][0], "size": full[want][1]}} if want in full else {}}
    ver = str(m.get("version", "")).lstrip("vV")
    if not ver or not is_newer(ver):
        return None
    notes, page = m.get("notes") or "", m.get("page") or RELEASES_PAGE
    base = patching.base_version()
    min_base = str(m.get("min_base", ver))
    patch = m.get("patch") or {}
    if base and patch.get("url") and patching.parse_version(base) >= patching.parse_version(min_base):
        return Release(ver, notes, page, patch["url"], int(patch.get("size", 0)), "patch", min_base)
    full = (m.get("full") or {}).get(asset_name() or "", {})
    root = install_root()
    if sys.platform == "win32" and root and any(root.glob("unins*.exe")):
        return Release(ver, notes, page, None, 0, "page", min_base)
    if full.get("url") and is_newer(str(m.get("full_version", ver)), base or "0"):
        return Release(ver, notes, page, full["url"], int(full.get("size", 0)), "full", min_base)
    return Release(ver, notes, page, None, 0, "page", min_base)


def download(url: str, dest: Path, progress: Callable[[float], None] | None = None,
             should_stop: Callable[[], bool] | None = None) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            if should_stop and should_stop():
                raise InterruptedError("已取消")
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total)
    return dest


def prepare_patch(rel: Release, progress=None, should_stop=None) -> Path:
    """Download and install a small update; it is used from the next start."""
    from . import patching
    work = Path(tempfile.mkdtemp(prefix="patchclamp_patch_"))
    try:
        zpath = download(rel.asset_url, work / "patch.zip",
                         (lambda f: progress(0.9 * f)) if progress else None, should_stop)
        out = work / "pkg"
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
        if not (out / "patchclamp_gui" / "__init__.py").exists():
            raise RuntimeError("更新包内容不完整")
        dest = patching.install(out, rel.version, rel.min_base)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if progress:
        progress(1.0)
    return dest


def restart_app() -> None:
    """Start a fresh copy of the packaged app; the caller then quits."""
    root = install_root()
    if sys.platform == "darwin" and root is not None:
        subprocess.Popen(["/bin/sh", "-c", f"sleep 1; open -n \"{root}\""], start_new_session=True)
    else:
        subprocess.Popen([sys.executable], close_fds=True,
                         creationflags=(0x00000008 | 0x00000200) if sys.platform == "win32" else 0)


def prepare_update(rel: Release, progress=None, should_stop=None) -> Path:
    """Download and unpack the update; returns the folder holding the new app."""
    if rel.kind == "patch":
        return prepare_patch(rel, progress, should_stop)
    root = install_root()
    if root is None or rel.asset_url is None:
        raise RuntimeError("当前版本不支持自动更新，请到发布页面下载新版本。")
    if _writable_dir(root.parent) != root.parent:
        raise RuntimeError(f"没有权限写入 {root.parent}，请到发布页面手动下载新版本。")
    work = Path(tempfile.mkdtemp(prefix="patchclamp_update_", dir=root.parent))
    zpath = download(rel.asset_url, work / "update.zip",
                     (lambda f: progress(0.9 * f)) if progress else None, should_stop)
    new_dir = work / "new"
    if sys.platform == "darwin":
        subprocess.run(["ditto", "-x", "-k", str(zpath), str(new_dir)], check=True)
        _check_mac_app(new_dir / "PatchClampAnalysis.app")
    else:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(new_dir)
    zpath.unlink(missing_ok=True)
    if progress:
        progress(1.0)
    return new_dir


def _check_mac_app(app: Path) -> None:
    """Make the unpacked app open without a Gatekeeper prompt: drop any quarantine
    flag and refuse an app whose signature is broken (macOS would block it)."""
    if not app.is_dir():
        raise RuntimeError("更新包内容不完整")
    subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(app)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("更新包签名校验失败，请到发布页面手动下载新版本。\n" + (r.stderr or "").strip())


def _writable_dir(preferred: Path) -> Path:
    try:
        test = preferred / f".patchclamp_write_test_{os.getpid()}"
        test.write_text("x")
        test.unlink()
        return preferred
    except OSError:
        return Path(tempfile.gettempdir())


def launch_installer(new_dir: Path) -> None:
    """Start the helper that swaps in the new version after this process exits.
    The caller must quit the application right after this returns."""
    root = install_root()
    pid = os.getpid()
    if sys.platform == "win32":
        bat = new_dir.parent / "install_update.bat"
        # paths travel as environment variables: the script itself stays ASCII
        bat.write_text(
            "@echo off\r\n"
            ":wait\r\n"
            "tasklist /FI \"PID eq %MS_PID%\" | find \"%MS_PID%\" >nul && (ping -n 2 127.0.0.1 >nul & goto wait)\r\n"
            "robocopy \"%MS_SRC%\" \"%MS_DST%\" /MIR /R:5 /W:1 /NFL /NDL /NJH /NJS >nul\r\n"
            "start \"\" \"%MS_DST%\\PatchClampAnalysis.exe\"\r\n", encoding="ascii")
        env = dict(os.environ, MS_PID=str(pid), MS_SRC=str(new_dir / "PatchClampAnalysis"), MS_DST=str(root))
        subprocess.Popen(["cmd", "/c", str(bat)], env=env, close_fds=True,
                         creationflags=0x00000008 | 0x00000200 | 0x08000000)  # detached, new group, no window
    elif sys.platform == "darwin":
        src = new_dir / "PatchClampAnalysis.app"
        sh = new_dir.parent / "install_update.sh"
        sh.write_text(
            "#!/bin/sh\n"
            f"while kill -0 {pid} 2>/dev/null; do sleep 1; done\n"
            f"rm -rf \"{root}.old\" && mv \"{root}\" \"{root}.old\" && mv \"{src}\" \"{root}\" "
            f"&& rm -rf \"{root}.old\"\n"
            f"xattr -dr com.apple.quarantine \"{root}\" 2>/dev/null\n"
            f"open \"{root}\"\n"
            f"rm -rf \"{new_dir.parent}\"\n")
        sh.chmod(0o755)
        subprocess.Popen(["/bin/sh", str(sh)], start_new_session=True, close_fds=True)
    else:
        raise RuntimeError("不支持的系统")


def cleanup_stale(root: Path | None = None) -> None:
    """Remove leftovers of an interrupted update."""
    root = root or install_root()
    if root is None:
        return
    for p in root.parent.glob("patchclamp_update_*"):
        shutil.rmtree(p, ignore_errors=True)
