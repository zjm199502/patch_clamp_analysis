"""事件检测。首选 miniML（深度学习），另提供去卷积检测作为无 TensorFlow 环境下的后备。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .deconv import deconvolve

# 不同类型事件的经验默认参数（可在界面中修改）
PRESETS = {
    "mEPSC / sEPSC": dict(direction="negative", window_ms=30.0, tau_rise_ms=0.4, tau_decay_ms=4.0),
    "mIPSC / sIPSC（高氯内液，内向）": dict(direction="negative", window_ms=80.0, tau_rise_ms=0.6, tau_decay_ms=12.0),
    "mIPSC / sIPSC（低氯内液，外向）": dict(direction="positive", window_ms=80.0, tau_rise_ms=0.6, tau_decay_ms=12.0),
}


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


def detect_miniml(y: np.ndarray, fs: float, direction: str = "negative", window_ms: float = 30.0,
                  model: str = "GC_lstm_model.h5", threshold: float = 0.5, batch_size: int = 512,
                  unit: str = "pA", callbacks: list | None = None) -> Detection:
    """调用 miniML 的 EventDetection。

    window_ms 应覆盖一个完整事件（约为衰减时间常数的 8–12 倍）；miniML 会把该窗口重采样到
    模型训练时的 600 点，因此同一个模型可用于动力学不同的事件。model 可以是 miniML 自带模型
    的相对路径，也可以是用户迁移学习得到的 .h5 文件的绝对路径。
    """
    import os

    from miniml.core.event import EventDetection
    from miniml.core.trace import MiniTrace
    from miniml.resources.util import get_resource_file_path

    model_path = model if os.path.isabs(model) and os.path.exists(model) \
        else get_resource_file_path(f"models/{model}")
    win = int(round(window_ms / 1e3 * fs))
    trace = MiniTrace(data=np.asarray(y, float), sampling_interval=1 / fs, y_unit=unit)
    det = EventDetection(data=trace, model=model_path, window_size=win, model_threshold=threshold,
                         batch_size=batch_size, event_direction=direction,
                         training_direction="negative", compile_model=True, verbose=0,
                         callbacks=callbacks)
    det.detect_events(eval=False, resample_to_600=True)
    onsets = np.asarray(det.event_locations, int)
    scores = np.asarray(det.event_scores, float)
    order = np.argsort(onsets)
    return Detection(onsets=onsets[order], scores=scores[order] if len(scores) == len(onsets) else scores,
                     method="miniML", trace=np.asarray(det.prediction, float), threshold=threshold,
                     params=dict(model=model, window_ms=window_ms, threshold=threshold, direction=direction))


def detect_deconv(y: np.ndarray, fs: float, direction: str = "negative", tau_rise_ms: float = 0.4,
                  tau_decay_ms: float = 4.0, threshold_sd: float = 4.0, smooth_ms: float = 0.5,
                  min_interval_ms: float = 1.5) -> Detection:
    res = deconvolve(y, fs, tau_rise_ms, tau_decay_ms, direction, smooth_ms=smooth_ms,
                     threshold_sd=threshold_sd, min_interval_ms=min_interval_ms)
    return Detection(onsets=res.peaks, scores=res.peak_heights / res.sd if res.sd else res.peak_heights,
                     method="deconvolution", trace=res.deconvolved, threshold=res.threshold,
                     params=dict(tau_rise_ms=tau_rise_ms, tau_decay_ms=tau_decay_ms,
                                 threshold_sd=threshold_sd, smooth_ms=smooth_ms, direction=direction))
