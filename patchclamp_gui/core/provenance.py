import hashlib
import json
from pathlib import Path

DETECTION_REVISION = 'detection-1'


def stamp(path):
    try:
        info = Path(path).stat()
        return [info.st_size, info.st_mtime_ns]
    except OSError:
        return None


def signature(cell, *, legacy_version=None):
    rec = cell.rec
    model = cell.params.get("model") if cell.params.get("detector") == "miniML" else None
    params = cell.params
    if legacy_version is None:
        from .session import DEFAULT_PARAMS
        fields = ["detector", "direction", "window_ms", "min_amplitude", "lowpass_hz",
                  "highpass_hz", "notch_hz", "detrend_s"]
        if model:
            fields.extend(["model", "threshold", "miniml_pipeline"])
            if params.get("miniml_pipeline") == "整批滑窗（实验）":
                fields.append("packed_batch")
        else:
            fields.extend(["tau_rise_ms", "tau_decay_ms", "deconv_threshold_sd", "deconv_smooth_ms"])
        params = {key: params.get(key, DEFAULT_PARAMS[key]) for key in fields}
    payload = [legacy_version or DETECTION_REVISION, params, Path(rec.path).name if rec.path else rec.name,
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
    stamp = [signature(cell), cell.detected_signature, cell.group, cell.animal, review]
    return stamp + [cell.folders.copy()] if cell.folders else stamp
