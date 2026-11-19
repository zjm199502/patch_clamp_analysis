"""逐事件的动力学特征提取。所有检测器共用同一套测量，保证结果可比。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

FEATURES = ["amplitude", "rise_10_90", "decay_tau", "half_width", "charge", "max_rise_slope"]
FEATURE_LABELS = {
    "amplitude": "Amplitude (pA)",
    "rise_10_90": "10–90% rise time (ms)",
    "decay_tau": "Decay τ (ms)",
    "half_width": "Half-width (ms)",
    "charge": "Charge (fC)",
    "max_rise_slope": "Max rise slope (pA/ms)",
    "iei": "Inter-event interval (s)",
    "frequency": "Frequency (Hz)",
}


def _cross_back(seg: np.ndarray, level: float) -> float:
    """seg 末端为峰值，向前找第一次低于 level 的位置（线性插值），返回距 seg 起点的小数索引。"""
    below = np.nonzero(seg < level)[0]
    if len(below) == 0:
        return np.nan
    i = below[-1]
    if i + 1 >= len(seg) or seg[i + 1] == seg[i]:
        return float(i)
    return i + (level - seg[i]) / (seg[i + 1] - seg[i])


def _cross_fwd(seg: np.ndarray, level: float) -> float:
    """seg 起点为峰值，向后找第一次低于 level 的位置。"""
    below = np.nonzero(seg < level)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0 or seg[i - 1] == seg[i]:
        return float(i)
    return (i - 1) + (seg[i - 1] - level) / (seg[i - 1] - seg[i])


def _exp(t, a, tau):
    return a * np.exp(-t / tau)


def extract_events(y: np.ndarray, fs: float, onsets: np.ndarray, direction: str = "negative",
                   window_ms: float = 30.0, baseline_ms: float = 2.0, smooth_ms: float = 0.3,
                   scores: np.ndarray | None = None, min_amplitude: float = 0.0,
                   fit_decay: bool = True, peak_stop=None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """返回 (事件表, 波形矩阵, 波形时间轴 ms)。

    波形以上升相 50% 点对齐、减去局部基线、事件方向翻为正，便于后续叠加与聚类。
    """
    sign = -1.0 if direction == "negative" else 1.0
    s = sign * np.asarray(y, float)
    n = len(s)
    k = max(int(round(smooth_ms / 1e3 * fs)), 1)
    if k > 1:
        w = np.hanning(k + 2)[1:-1]
        ys = np.convolve(s, w / w.sum(), mode="same")
    else:
        ys = s
    win = int(round(window_ms / 1e3 * fs))
    bsl = max(int(round(baseline_ms / 1e3 * fs)), 3)
    gap = max(int(round(0.5e-3 * fs)), 1)
    search = max(win // 3, 5)
    pre = win // 4
    onsets = np.asarray(onsets, int)
    order = np.argsort(onsets)
    onsets = onsets[order]
    scores = None if scores is None or len(scores) != len(order) else np.asarray(scores)[order]

    rows, waves = [], []
    ms = 1e3 / fs
    prev_peak = -1
    for j, loc in enumerate(onsets):
        nxt = onsets[j + 1] if j + 1 < len(onsets) else n
        b0, b1 = loc - gap - bsl, loc - gap
        if b0 < 0 or loc + 5 >= n:
            continue
        p_end = min(loc + search, n - 1, max(nxt + gap, loc + 5))
        if peak_stop is not None:
            p_end = min(peak_stop, n - 1)
        peak = loc + int(np.argmax(ys[loc:p_end + 1]))
        base = float(np.median(ys[b0:b1]))
        amp = float(np.mean(ys[max(peak - 1, 0):peak + 2]) - base)
        if not amp > min_amplitude:
            continue
        rs = min(max(b1, prev_peak + 1), peak)       # 上升相的搜索不越过前一个事件的峰
        rise_seg = ys[rs:peak + 1] - base
        i10, i50, i90 = (_cross_back(rise_seg, f * amp) for f in (0.1, 0.5, 0.9))
        rise = (i90 - i10) * ms if np.isfinite(i10) and np.isfinite(i90) else np.nan
        d_end = min(peak + win, n, nxt - gap if nxt - gap > peak + 5 else peak + win)
        d_end = min(d_end, n)
        dec = ys[peak:d_end] - base
        h50 = _cross_fwd(dec, 0.5 * amp)
        half_decay = h50 * ms if np.isfinite(h50) else np.nan
        half_width = ((peak - rs - i50) + h50) * ms if np.isfinite(h50) and np.isfinite(i50) else np.nan
        tau = np.nan
        if fit_decay and len(dec) > 8:
            t = np.arange(len(dec)) * ms
            try:
                tau0 = half_decay / np.log(2) if np.isfinite(half_decay) and half_decay > 0 else window_ms / 5
                popt, _ = curve_fit(_exp, t, dec, p0=(amp, tau0),
                                    bounds=([0, ms / 2], [10 * amp + 1, 20 * window_ms]), maxfev=200)
                tau = float(popt[1])
            except Exception:
                tau = np.nan
        on_i = rs + (int(i10) if np.isfinite(i10) else 0)
        charge = float(np.sum(ys[on_i:d_end] - base) * ms)          # pA·ms = fC
        slope = float(np.max(np.diff(ys[on_i:peak + 1])) / ms) if peak - on_i >= 1 else np.nan
        a50 = rs + (int(round(i50)) if np.isfinite(i50) else peak - rs)
        w0, w1 = a50 - pre, a50 + win
        if w0 < 0 or w1 > n:
            continue
        if peak <= prev_peak:
            continue
        prev_peak = peak
        waves.append(s[w0:w1] - base)
        rows.append(dict(onset_idx=int(on_i), peak_idx=int(peak), time_s=on_i / fs,
                         amplitude=amp, rise_10_90=rise, decay_tau=tau, half_decay=half_decay,
                         half_width=half_width, charge=charge, max_rise_slope=slope,
                         baseline=sign * base, overlap=bool(nxt - loc < win),
                         score=float(scores[j]) if scores is not None else np.nan))
    df = pd.DataFrame(rows)
    if len(df):
        waves_arr = np.asarray(waves)
        keep = ~df.duplicated("peak_idx").to_numpy()
        df, waves_arr = df[keep].reset_index(drop=True), waves_arr[keep]
        df["iei"] = df["time_s"].diff()
        df["inst_freq"] = 1.0 / df["iei"]
    else:
        waves_arr = np.zeros((0, pre + win))
        df = pd.DataFrame(columns=["onset_idx", "peak_idx", "time_s", *FEATURES, "half_decay",
                                   "baseline", "overlap", "score", "iei", "inst_freq"])
    t_wave = (np.arange(pre + win) - pre) * ms
    return df, waves_arr, t_wave


def summarize(df: pd.DataFrame, duration_s: float, waves: np.ndarray | None = None,
              fs: float | None = None) -> dict:
    """单个细胞（记录）的汇总指标：统计推断应以细胞为单位。"""
    out = {"n_events": int(len(df)), "duration_s": duration_s,
           "frequency": len(df) / duration_s if duration_s else np.nan}
    for f in FEATURES:
        v = pd.to_numeric(df.get(f, pd.Series(dtype=float)), errors="coerce").dropna()
        out[f"{f}_mean"] = float(v.mean()) if len(v) else np.nan
        out[f"{f}_median"] = float(v.median()) if len(v) else np.nan
        out[f"{f}_cv"] = float(v.std() / v.mean()) if len(v) > 1 and v.mean() else np.nan
    iei = pd.to_numeric(df.get("iei", pd.Series(dtype=float)), errors="coerce").dropna()
    out["iei_median"] = float(iei.median()) if len(iei) else np.nan
    out["iei_cv"] = float(iei.std() / iei.mean()) if len(iei) > 1 else np.nan
    if waves is not None and len(waves) and fs:
        avg = np.nanmean(waves, axis=0)
        pk = int(np.argmax(avg))
        dec = avg[pk:]
        t = np.arange(len(dec)) * 1e3 / fs
        try:
            popt, _ = curve_fit(_exp, t, dec, p0=(dec[0], max(t[-1] / 5, 0.1)), maxfev=2000)
            out["avg_event_tau"] = float(popt[1])
        except Exception:
            out["avg_event_tau"] = np.nan
        out["avg_event_amplitude"] = float(avg[pk])
    return out
