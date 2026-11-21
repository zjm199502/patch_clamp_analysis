"""去卷积平滑。

思路（Pernía-Andrade et al., 2012, Biophys J）：把记录看成"事件发生序列 * 单位事件波形"，
在频域除以模板波形即可把每个缓慢衰减的突触电流压缩成一个尖峰，重叠事件因此被分开。
直接相除会放大高频噪声，所以这里用 Wiener 正则化并叠加一个高斯低通（即"平滑"）。

输出三条曲线：
* deconvolved —— 平滑后的去卷积曲线（峰面积≈事件幅度），可直接阈值检测；
* sparse —— 阈值化后的稀疏事件序列；
* reconvolved —— 稀疏序列与模板重新卷积得到的"去噪重建"曲线，可与原始记录叠加检查拟合。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq

from . import gpu
from ..io import biexp_kernel
from .preprocess import highpass, noise_rms


@dataclass
class DeconvResult:
    deconvolved: np.ndarray
    threshold: float
    peaks: np.ndarray           # 事件起始点索引
    peak_heights: np.ndarray
    sparse: np.ndarray
    reconvolved: np.ndarray     # 与原始记录同极性
    kernel: np.ndarray
    sd: float


def deconvolve(y: np.ndarray, fs: float, tau_rise_ms: float = 0.5, tau_decay_ms: float = 5.0,
               direction: str = "negative", smooth_ms: float = 0.5, highpass_hz: float = 1.0,
               threshold_sd: float = 4.0, min_interval_ms: float = 1.0,
               reg: float = 1e-4, kernel: np.ndarray | None = None) -> DeconvResult:
    sign = -1.0 if direction == "negative" else 1.0
    s = sign * (np.asarray(y, float) - np.median(y))
    s = highpass(s, fs, highpass_hz)
    n = len(s)
    nfft = next_fast_len(n + int(0.05 * fs), real=True)
    if kernel is None:
        kernel = biexp_kernel(fs, tau_rise_ms, tau_decay_ms)
    sigma = smooth_ms / 1e3
    use_gpu = gpu.supports("wiener") and n >= 250_000
    d = K = None
    if use_gpu:
        d = gpu.calculate("wiener", {"x": s, "kernel": kernel}, nfft=nfft, reg=reg, fs=fs, sigma=sigma)
    if d is None:
        K = rfft(kernel, nfft)
        S = rfft(s, nfft)
        f = rfftfreq(nfft, 1 / fs)
        wiener = np.conj(K) / (np.abs(K) ** 2 + reg * np.max(np.abs(K)) ** 2)
        G = np.exp(-0.5 * (2 * np.pi * f * sigma) ** 2) if sigma > 0 else 1.0
        d = irfft(S * wiener * G, nfft)[:n]

    sd = noise_rms(d)
    thr = threshold_sd * sd
    dist = max(int(min_interval_ms / 1e3 * fs), 1)
    peaks = gpu.find_peaks(d, height=thr, distance=dist)

    # 稀疏化：只保留超过阈值的峰附近区域，再与模板卷积重建
    sparse = np.zeros(n)
    if len(peaks):
        half = max(int(3 * sigma * fs), 1)
        mask = np.zeros(n, bool)
        for p in peaks:
            mask[max(p - half, 0):p + half + 1] = True
        sparse[mask] = np.clip(d[mask], 0, None)
    recon = None
    if use_gpu:
        recon = gpu.calculate("reconstruct", {"x": sparse, "kernel": kernel}, nfft=nfft)
    if recon is None:
        if K is None:
            K = rfft(kernel, nfft)
        recon = irfft(rfft(sparse, nfft) * K, nfft)[:n]
    return DeconvResult(deconvolved=d, threshold=thr, peaks=peaks,
                        peak_heights=d[peaks],
                        sparse=sparse, reconvolved=sign * recon, kernel=kernel, sd=sd)


def fit_kernel_from_average(avg: np.ndarray, fs: float) -> tuple[float, float]:
    """由平均事件波形（正向、基线为 0）拟合双指数模板，返回 (tau_rise_ms, tau_decay_ms)。"""
    from scipy.optimize import curve_fit

    avg = np.asarray(avg, float)
    i0 = int(np.argmax(avg > 0.05 * np.nanmax(avg)))
    seg = avg[max(i0 - 1, 0):]
    t = np.arange(len(seg)) / fs * 1e3

    def model(t, a, tr, ratio):
        td = tr * ratio
        k = np.exp(-t / td) - np.exp(-t / tr)
        return a * k / k.max()

    ipk = int(np.argmax(seg))
    tp = max(t[ipk], 1e3 / fs)
    below = np.nonzero(seg[ipk:] < seg.max() / np.e)[0]
    td0 = t[below[0]] if len(below) else t[-1] / 3
    p0 = (seg.max(), max(tp / 2.5, 0.05), max(td0 / max(tp / 2.5, 0.05), 2.0))
    try:
        popt, _ = curve_fit(model, t, seg, p0=p0,
                            bounds=([0, 0.02, 1.2], [np.inf, 20, 1000]), maxfev=5000)
        return float(popt[1]), float(popt[1] * popt[2])
    except Exception:
        return float(p0[1]), float(p0[1] * p0[2])
