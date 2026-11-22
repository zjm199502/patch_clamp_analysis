import json
import pytest
import zipfile

from patchclamp_gui import updater


def test_version_compare():
    assert updater.is_newer("0.2.0", "0.1.9")
    assert updater.is_newer("v1.0", "0.9.9")
    assert not updater.is_newer("0.1.0", "0.1.0")
    assert not updater.is_newer("0.1.0", "0.2.0")
    assert updater.parse_version("v0.3") == (0, 3, 0)


class _Resp:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n=-1):
        if n < 0:
            n = len(self.data)
        out = self.data[self.pos:self.pos + n]
        self.pos += len(out)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_check_latest_and_prepare(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "install_root", lambda: tmp_path)
    name = updater.asset_name() or "PatchClampAnalysis-Windows.zip"
    monkeypatch.setattr(updater, "asset_name", lambda: name)
    rel_json = {"tag_name": "v99.0.0", "body": "notes", "html_url": "https://x/r",
                "assets": [{"name": name, "browser_download_url": "https://x/a.zip", "size": 10}]}
    zbuf = tmp_path / "src.zip"
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("PatchClampAnalysis/PatchClampAnalysis.exe", "new")

    def fake_open(req, timeout=None, context=None):
        url = req.full_url
        if "api.github.com" in url:
            return _Resp(json.dumps(rel_json).encode())
        return _Resp(zbuf.read_bytes())
    monkeypatch.setattr(updater.urllib.request, "urlopen", fake_open)
    rel = updater.check_latest()
    assert rel.version == "99.0.0" and rel.asset_url == "https://x/a.zip"

    app = tmp_path / "apps" / "PatchClampAnalysis"
    app.mkdir(parents=True)
    monkeypatch.setattr(updater, "install_root", lambda: app)
    monkeypatch.setattr(updater, "_ssl_context", lambda: None)
    monkeypatch.setattr(updater.sys, "platform", "win32")
    seen = []
    new_dir = updater.prepare_update(rel, seen.append)
    assert (new_dir / "PatchClampAnalysis" / "PatchClampAnalysis.exe").read_text() == "new"
    assert seen[-1] == 1.0
    assert new_dir.parent.parent == app.parent
    updater.cleanup_stale(app)
    assert not new_dir.exists()


def test_no_update_when_same(monkeypatch):
    def fake_open(req, timeout=None, context=None):
        return _Resp(json.dumps({"version": updater.__version__}).encode())
    monkeypatch.setattr(updater.urllib.request, "urlopen", fake_open)
    assert updater.check_latest() is None


def test_channel_api_after_timeout(monkeypatch):
    seen = []

    def get(url, timeout, accept=None):
        seen.append(url)
        if url == updater.CHANNEL_MANIFEST:
            raise TimeoutError()
        if "/tags/updates?" in url:
            return {"assets": [{"name": "latest.json", "id": 123}]}
        assert url.startswith(updater.API_ROOT + "/assets/123?")
        assert accept == "application/octet-stream"
        return {"version": "99.3.0"}

    monkeypatch.setattr(updater, "_get_json", get)
    assert updater.check_latest().version == "99.3.0"
    assert updater.API_LATEST not in seen


@pytest.mark.parametrize("major_reachable", [False, True])
def test_unreachable_channel_does_not_report_up_to_date(monkeypatch, major_reachable):
    def get(url, timeout):
        if major_reachable and url == updater.API_LATEST:
            return {"tag_name": "v0.3.0"}
        raise TimeoutError()

    monkeypatch.setattr(updater, "_get_json", get)
    with pytest.raises(ConnectionError, match="GitHub API：连接超时"):
        updater.check_latest()


def test_cut_off_download_restarts_and_checks_hash(monkeypatch, tmp_path):
    import hashlib
    payload = b"complete"
    monkeypatch.setattr(updater, "_api_asset", lambda name, timeout: "https://api.github.com/asset")
    monkeypatch.setattr(updater, "_ssl_context", lambda: None)

    def get(req, **kw):
        return _Resp(payload if req.get_header("Accept") == "application/octet-stream" else b"cut")

    monkeypatch.setattr(updater.urllib.request, "urlopen", get)
    url = updater.CHANNEL_ROOT + "patch.zip"
    dest = tmp_path / "patch.zip"
    assert updater.download(url, dest, size=len(payload), sha256=hashlib.sha256(payload).hexdigest()).read_bytes() == payload
    with pytest.raises(ConnectionError, match="SHA-256"):
        updater.download(url, dest, size=len(payload), sha256="0" * 64)
    with pytest.raises(InterruptedError):
        updater.download(url, dest, should_stop=lambda: True)


def test_channel_patch_and_full(monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "install_root", lambda: tmp_path)
    from patchclamp_gui import patching
    man = {"version": "99.0.3", "notes": "n", "min_base": "99.0.1",
           "patch": {"url": "https://x/p.zip", "size": 5},
           "full_version": "99.0.1", "full": {"PatchClampAnalysis-Windows.zip": {"url": "https://x/w.zip", "size": 9}}}

    monkeypatch.setattr(updater, "_get_json", lambda url, timeout: man)
    monkeypatch.setattr(updater, "asset_name", lambda: "PatchClampAnalysis-Windows.zip")
    # base new enough -> small update
    monkeypatch.setattr(patching, "base_version", lambda: "99.0.1")
    r = updater.check_latest()
    assert r.kind == "patch" and r.asset_url == "https://x/p.zip"
    # old base -> whole app first
    monkeypatch.setattr(patching, "base_version", lambda: "0.2.0")
    r = updater.check_latest()
    assert r.kind == "full" and r.asset_url == "https://x/w.zip"
    (tmp_path / "unins000.exe").touch()
    monkeypatch.setattr(updater.sys, "platform", "win32")
    assert updater.check_latest().kind == "page"
    slot = tmp_path / "ClampTrace-runtime-2"
    slot.mkdir()
    monkeypatch.setattr(updater, "install_root", lambda: slot)
    assert updater.check_latest().kind == "page"
    # running from source -> only the download page
    monkeypatch.setattr(patching, "base_version", lambda: None)
    assert updater.check_latest().kind in ("full", "page")


def test_patch_install_and_activate(monkeypatch, tmp_path):
    import sys
    from patchclamp_gui import patching
    monkeypatch.setattr(patching, "data_dir", lambda: tmp_path / "data")
    src = tmp_path / "unz"
    (src / "patchclamp_gui").mkdir(parents=True)
    (src / "patchclamp_gui" / "__init__.py").write_text('__version__ = "1.0.5"\n')
    dest = patching.install(src, "1.0.5", "1.0.1")
    assert (dest / "patchclamp_gui" / "__init__.py").exists()
    info = patching.read_current()
    assert patching.usable(info, "1.0.1") and patching.usable(info, "1.0.4")
    assert not patching.usable(info, "1.0.0")        # base too old for this patch
    assert not patching.usable(info, "1.0.5")        # base already this new
    monkeypatch.setattr(patching, "base_version", lambda: "1.0.2")
    path = list(sys.path)
    try:
        assert patching.activate() == "1.0.5" and sys.path[0] == str(dest)
    finally:
        sys.path[:] = path
    patching.mark_bad("1.0.5")
    assert not patching.usable(patching.read_current(), "1.0.2")


def test_mac_app_check(monkeypatch, tmp_path):
    app = tmp_path / "PatchClampAnalysis.app"
    calls = []

    class R:
        def __init__(self, rc): self.returncode, self.stderr = rc, "invalid signature"

    rc = {"codesign": 0}
    def fake_run(cmd, **kw):
        calls.append(cmd[0])
        return R(rc.get(cmd[0], 0))
    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        updater._check_mac_app(app)          # missing app
    app.mkdir()
    updater._check_mac_app(app)
    assert calls == ["xattr", "codesign"]    # quarantine removed, then signature verified
    rc["codesign"] = 1
    with pytest.raises(RuntimeError, match="签名"):
        updater._check_mac_app(app)


def test_version_rule():
    """0.1.0 → 0.1.1 … 0.1.20 → 0.2.0（大版本）。"""
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("bump", pathlib.Path(__file__).parents[1] / "tools" / "bump_version.py")
    bump = importlib.util.module_from_spec(spec); spec.loader.exec_module(bump)
    v, seq = "0.1.0", []
    for _ in range(21):
        v = bump.bump(v, "patch"); seq.append(v)
    assert seq[0] == "0.1.1" and seq[19] == "0.1.20" and seq[20] == "0.2.0"
