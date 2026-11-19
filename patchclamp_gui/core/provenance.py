import hashlib
import json
from pathlib import Path

from .. import __version__


def stamp(path):
    try:
        info = Path(path).stat()
        return [info.st_size, info.st_mtime_ns]
    except OSError:
        return None


def signature(cell):
    rec = cell.rec
    model = cell.params.get("model") if cell.params.get("detector") == "miniML" else None
    payload = [__version__, cell.params, Path(rec.path).name if rec.path else rec.name,
               stamp(rec.path) if rec.path else rec.data.size, rec.fs, rec.channel,
               stamp(model) if model and Path(model).is_absolute() else None]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def status(cell):
    if cell.failure:
        return "失败"
    if cell.events is None:
        return "待重算" if cell.detected_signature else "未开始"
    return "已完成" if cell.detected_signature == signature(cell) else "待重算"


def report_stamp(cell):
    review = ""
    if cell.events is not None:
        review = hashlib.sha256(cell.events.to_numpy(dtype=float).tobytes()).hexdigest()
    return [signature(cell), cell.detected_signature, cell.group, cell.animal, review]
