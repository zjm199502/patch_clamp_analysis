"""预处理：低通、高通/去趋势、工频陷波。"""
from __future__ import annotations

import numpy as np
from scipy import signal


def lowpass(y: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if not cutoff_hz or cutoff_hz >= fs / 2:
        return y
    sos = signal.butter(order, cutoff_hz / (fs / 2), btype="low", output="sos")
    return signal.sosfiltfilt(sos, y)


def highpass(y: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    if not cutoff_hz:
        return y
    sos = signal.butter(order, cutoff_hz / (fs / 2), btype="high", output="sos")
    return signal.sosfiltfilt(sos, y)


def notch(y: np.ndarray, fs: float, freq_hz: float, q: float = 30.0, harmonics: int = 3) -> np.ndarray:
    if not freq_hz:
        return y
    out = y
    for h in range(1, harmonics + 1):
        f0 = freq_hz * h
        if f0 >= fs / 2:
            break
        b, a = signal.iirnotch(f0, q, fs)
        out = signal.filtfilt(b, a, out)
    return out


def rolling_baseline(y: np.ndarray, fs: float, win_s: float = 1.0) -> np.ndarray:
    """分段中位数 + 线性插值得到的慢基线（对事件稳健）。"""
    w = max(int(win_s * fs), 10)
    nseg = max(len(y) // w, 1)
    edges = np.linspace(0, len(y), nseg + 1).astype(int)
    xc = (edges[:-1] + edges[1:]) / 2
    med = np.array([np.median(y[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    return np.interp(np.arange(len(y)), xc, med)


def preprocess(y: np.ndarray, fs: float, lowpass_hz: float = 0.0, highpass_hz: float = 0.0,
               notch_hz: float = 0.0, detrend_s: float = 0.0) -> np.ndarray:
    out = np.asarray(y, float)
    if detrend_s:
        out = out - rolling_baseline(out, fs, detrend_s)
    out = highpass(out, fs, highpass_hz)
    out = notch(out, fs, notch_hz)
    out = lowpass(out, fs, lowpass_hz)
    return out


def noise_rms(y: np.ndarray) -> float:
    """基于 MAD 的稳健噪声估计（对事件不敏感）。"""
    d = y - np.median(y)
    return float(1.4826 * np.median(np.abs(d - np.median(d))))
