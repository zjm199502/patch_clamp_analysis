"""Portable project storage. The HDF5 snapshot contains no pickled objects."""
import json
import os
import shutil
import tempfile
import time
import uuid
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


def save_project(root, project, destination=None):
    root = Path(root)
    with tempfile.NamedTemporaryFile(dir=root, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with h5py.File(temporary, "w") as store:
            generation = uuid.uuid4().hex
            store.attrs["generation"] = generation
            store.attrs["patchclamp_format"] = 1
            for index, cell in enumerate(project.cells):
                group = store.create_group(str(index))
                rec = cell.rec
                relative = Path(rec.path).resolve().relative_to(root.resolve()).as_posix() if rec.path else ""
                group.attrs["record"] = _json(dict(name=rec.name, fs=rec.fs, unit=rec.unit, channel=rec.channel, path=relative))
                group.create_dataset("metadata", data=_json(rec.meta))
                group.attrs["cell"] = _json(dict(group=cell.group, animal=cell.animal, note=cell.note, folders=cell.folders, params=cell.params, qc=cell.qc,
                                                detected_signature=cell.detected_signature, failure=cell.failure))
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
        os.replace(temporary, destination or root / SNAPSHOT)
        if destination is None:
            old = project.generation
            project.generation = generation
            if len(old) == 32 and all(char in "0123456789abcdef" for char in old):
                shutil.rmtree(root / ".checkpoints" / old, ignore_errors=True)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint(root, project, index):
    folder = Path(root) / ".checkpoints" / project.generation
    folder.mkdir(parents=True, exist_ok=True)
    save_project(root, Project([project.cells[index]]), folder / f"{index}.h5")


def open_project(root, source=None):
    from ..io import load_any

    root = Path(root).resolve()
    project = Project()
    with h5py.File(source or root / SNAPSHOT, "r") as store:
        project.generation = store.attrs.get("generation", "")
        if store.attrs.get("patchclamp_format") != 1:
            raise ValueError("无法识别此项目格式")
        for index in range(len(store)):
            if source is None and len(project.generation) == 32 and all(char in "0123456789abcdef" for char in project.generation):
                recovered = root / ".checkpoints" / project.generation / f"{index}.h5"
                if recovered.is_file():
                    project.cells.extend(open_project(root, recovered).cells)
                    continue
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
            attributes = json.loads(group.attrs["cell"])
            if "folders" not in attributes and "/" in attributes.get("group", ""):
                attributes["folders"] = attributes["group"].split("/")
            cell = CellSession(rec, **attributes)
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
