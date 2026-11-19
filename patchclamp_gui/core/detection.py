"""事件检测。首选 miniML（深度学习），另提供去卷积检测作为无 TensorFlow 环境下的后备。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .deconv import deconvolve

# 不同类型事件的经验默认参数（可在界面中修改）
PRESETS = {
    "mEPSC / sEPSC": dict(direction="negative", window_ms=30.0, tau_rise_ms=0.4, tau_decay_ms=4.0),
    "mIPSC / sIPSC（高氯内液，内向）": dict(direction="negative", window_ms=80.0, tau_rise_ms=0.6, tau_decay_ms=12.0),
    "mIPSC / sIPSC（低氯内液，外向）": dict(direction="positive", window_ms=80.0, tau_rise_ms=0.6, tau_decay_ms=12.0),
}


def preset_from_path(path):
    path = Path(path)
    for part in (path.stem, *(parent.name for parent in path.parents)):
        excitatory = "epsc" in part.casefold()
        inhibitory = "ipsc" in part.casefold()
        if excitatory and inhibitory:
            return None
        if excitatory:
            return "mEPSC / sEPSC"
        if inhibitory:
            return "mIPSC / sIPSC（低氯内液，外向）"
    return None


@dataclass
class Detection:
    onsets: np.ndarray                      # 样本索引
    scores: np.ndarray                      # miniML 置信度或去卷积峰高（SD）
    method: str
    trace: np.ndarray | None = None         # 检测曲线（预测概率 / 去卷积曲线）
    threshold: float | None = None
    params: dict = field(default_factory=dict)


def miniml_available() -> tuple[bool, str]:
    try:
        import miniml  # noqa: F401
        import tensorflow  # noqa: F401
        return True, ""
    except Exception as e:  # pragma: no cover - 取决于环境
        return False, f"{type(e).__name__}: {e}"


def miniml_models() -> list[str]:
    try:
        from miniml.resources.util import get_available_models
        return sorted(get_available_models())
    except Exception:
        return []


def _graph_minmax(window):
    """miniML 1.1.2 的逐窗口归一化；独立版没有源码供 AutoGraph 转换 Python if。"""
    import tensorflow as tf

    low = tf.reduce_min(window)
    return tf.math.divide_no_nan(window - low, tf.reduce_max(window) - low)


_model_cache = {}


def _packed_predict(detector):
    import tensorflow as tf
    from scipy.signal import resample
    data = resample(detector.trace.data, round(len(detector.trace.data) * detector.resampling_factor))
    if detector.event_direction != detector.training_direction:
        data *= -1
    width = round(detector.window_size * detector.resampling_factor)
    step = round(detector.stride_length * detector.resampling_factor)
    windows = np.lib.stride_tricks.sliding_window_view(data.astype(np.float32), width)[::step]
    infer = getattr(detector.model, "_clamp_infer", None)
    if infer is None or getattr(detector.model, "_clamp_width", None) != width:
        model = detector.model
        @tf.function(input_signature=[tf.TensorSpec((None, width, 1), tf.float32)], autograph=False)
        def infer(batch):
            low = tf.reduce_min(batch, axis=1, keepdims=True)
            scaled = tf.math.divide_no_nan(batch-low, tf.reduce_max(batch, axis=1, keepdims=True)-low)
            return model(scaled, training=False)
        model._clamp_infer = infer
        model._clamp_width = width
    predictions = []
    for start in range(0, len(windows), detector.batch_size):
        batch = tf.convert_to_tensor(np.ascontiguousarray(windows[start:start+detector.batch_size, :, None]))
        predictions.append(infer(batch).numpy().reshape(-1))
    detector.prediction = tf.convert_to_tensor(np.concatenate(predictions))


def clear_model_cache():
    _model_cache.clear()


def detect_miniml(y: np.ndarray, fs: float, direction: str = "negative", window_ms: float = 30.0,
                  model: str = "GC_lstm_model.h5", threshold: float = 0.5, batch_size: int | None = None,
                  unit: str = "pA", callbacks: list | None = None, packed=False) -> Detection:
    """调用 miniML 的 EventDetection。

    window_ms 应覆盖一个完整事件（约为衰减时间常数的 8–12 倍）；miniML 会把该窗口重采样到
    模型训练时的 600 点，因此同一个模型可用于动力学不同的事件。model 可以是 miniML 自带模型
    的相对路径，也可以是用户迁移学习得到的 .h5 文件的绝对路径。
    """
    import os
    import sys

    from miniml.core import event
    from miniml.core.trace import MiniTrace
    from miniml.resources.util import get_resource_file_path

    model_path = model if os.path.isabs(model) and os.path.exists(model) \
        else get_resource_file_path(f"models/{model}")
    win = int(round(window_ms / 1e3 * fs))
    if batch_size is None:
        batch_size = 128 if packed else 512
    trace = MiniTrace(data=np.asarray(y, float), sampling_interval=1 / fs, y_unit=unit)
    if getattr(sys, "frozen", False):
        event.minmax_scaling = _graph_minmax
    path = Path(model_path).resolve()
    info = path.stat()
    key = (str(path), info.st_size, info.st_mtime_ns)
    det = event.EventDetection(data=trace, model=_model_cache.get(key, str(path)), window_size=win, model_threshold=threshold,
                         batch_size=batch_size, event_direction=direction,
                         training_direction="negative", compile_model=False, verbose=0,
                         callbacks=callbacks)
    if packed and not callbacks and hasattr(det, "_EventDetection__predict"):
        det._EventDetection__predict = lambda: _packed_predict(det)
    try:
        det.detect_events(eval=False, resample_to_600=True)
    finally:
        det.__dict__.pop("_EventDetection__predict", None)
    clear_model_cache()
    _model_cache[key] = det.model
    onsets = np.asarray(det.event_locations, int)
    scores = np.asarray(det.event_scores, float)
    order = np.argsort(onsets)
    return Detection(onsets=onsets[order], scores=scores[order] if len(scores) == len(onsets) else scores,
                     method="miniML", trace=np.asarray(det.prediction, float), threshold=threshold,
                     params=dict(model=model, window_ms=window_ms, threshold=threshold, direction=direction, batch_size=batch_size))


def detect_deconv(y: np.ndarray, fs: float, direction: str = "negative", tau_rise_ms: float = 0.4,
                  tau_decay_ms: float = 4.0, threshold_sd: float = 4.0, smooth_ms: float = 0.5,
                  min_interval_ms: float = 1.5) -> Detection:
    res = deconvolve(y, fs, tau_rise_ms, tau_decay_ms, direction, smooth_ms=smooth_ms,
                     threshold_sd=threshold_sd, min_interval_ms=min_interval_ms)
    return Detection(onsets=res.peaks, scores=res.peak_heights / res.sd if res.sd else res.peak_heights,
                     method="deconvolution", trace=res.deconvolved, threshold=res.threshold,
                     params=dict(tau_rise_ms=tau_rise_ms, tau_decay_ms=tau_decay_ms,
                                 threshold_sd=threshold_sd, smooth_ms=smooth_ms, direction=direction))
