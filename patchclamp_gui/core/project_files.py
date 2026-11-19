"""Portable project storage. The HDF5 snapshot contains no pickled objects."""
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..io import Recording
from .detection import Detection
from .session import CellSession, Project

SNAPSHOT = "project.pca.h5"


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=lambda item: item.tolist())


def create_project(root):
    root = Path(root)
    root.mkdir(exist_ok=False)
    (root / "raw").mkdir()
    (root / "results").mkdir()
    project = Project()
    save_project(root, project)
    return project


def copy_raw(root, source):
    source = Path(source)
    folder = Path(tempfile.mkdtemp(prefix="record-", dir=Path(root) / "raw"))
    try:
        return Path(shutil.copy2(source, folder / source.name))
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def save_project(root, project):
    root = Path(root)
    with tempfile.NamedTemporaryFile(dir=root, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with h5py.File(temporary, "w") as store:
            store.attrs["patchclamp_format"] = 1
            for index, cell in enumerate(project.cells):
                group = store.create_group(str(index))
                rec = cell.rec
                relative = Path(rec.path).resolve().relative_to(root.resolve()).as_posix() if rec.path else ""
                group.attrs["record"] = _json(dict(name=rec.name, fs=rec.fs, unit=rec.unit, channel=rec.channel, path=relative))
                group.create_dataset("metadata", data=_json(rec.meta))
                group.attrs["cell"] = _json(dict(group=cell.group, animal=cell.animal, params=cell.params, qc=cell.qc))
                if not relative:
                    group.create_dataset("data", data=rec.data)
                for key in ("processed", "waves", "t_wave"):
                    if getattr(cell, key) is not None:
                        group.create_dataset(key, data=getattr(cell, key))
                if cell.events is not None:
                    events = group.create_group("events")
                    events.attrs["columns"] = _json(list(cell.events.columns))
                    for column in cell.events:
                        values = cell.events[column].to_numpy()
                        events.create_dataset(column, data=values.astype(float) if not len(values) and values.dtype == object else values)
                if cell.det is not None:
                    det = group.create_group("detection")
                    det.attrs["settings"] = _json(dict(method=cell.det.method, params=cell.det.params, threshold=cell.det.threshold))
                    for key in ("onsets", "scores", "trace"):
                        if getattr(cell.det, key) is not None:
                            det.create_dataset(key, data=getattr(cell.det, key))
        os.replace(temporary, root / SNAPSHOT)
    finally:
        temporary.unlink(missing_ok=True)


def open_project(root):
    from ..io import load_any

    root = Path(root).resolve()
    project = Project()
    with h5py.File(root / SNAPSHOT, "r") as store:
        if store.attrs.get("patchclamp_format") != 1:
            raise ValueError("无法识别此项目格式")
        for index in range(len(store)):
            group = store[str(index)]
            info = json.loads(group.attrs["record"])
            if info["path"]:
                path = (root / info["path"]).resolve()
                if not path.is_relative_to(root / "raw"):
                    raise ValueError("原始记录路径不在项目 raw 文件夹中")
                rec = load_any(str(path), channel=info["channel"])
                rec.name = info["name"]
            else:
                rec = Recording(data=group["data"][()], **info)
            rec.meta = json.loads(group["metadata"][()])
            cell = CellSession(rec, **json.loads(group.attrs["cell"]))
            for key in ("processed", "waves", "t_wave"):
                if key in group:
                    setattr(cell, key, group[key][()])
            if "events" in group:
                events = group["events"]
                cell.events = pd.DataFrame({col: events[col][()] for col in json.loads(events.attrs["columns"])})
            if "detection" in group:
                det = group["detection"]
                cell.det = Detection(**json.loads(det.attrs["settings"]), **{key: det[key][()] for key in det})
            project.cells.append(cell)
    return project


def archive_result(root, fig, text, details, arrays=None):
    results = Path(root) / "results"
    with tempfile.TemporaryDirectory(dir=results) as staging:
        report = Path(staging) / "report"
        report.mkdir()
        if fig is not None:
            fig.savefig(report / "figure.png", dpi=150)
        if arrays:
            np.savez(report / "data.npz", **arrays)
        (report / "result.txt").write_text(text, encoding="utf-8")
        (report / "settings.json").write_text(_json(details), encoding="utf-8")
        label = details["analysis"].split("（")[0]
        label = "".join(char for char in label if char.isalnum() or char == "-")
        target = results / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}-{Path(staging).name}"
        report.rename(target)
    return target
