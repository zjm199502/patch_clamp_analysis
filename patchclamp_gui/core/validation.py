"""检测可信度评估。

* 合成事件注入（recovery / 灵敏度曲线）：把已知幅度的模板事件加到真实记录上再检测，
  得到"多大的事件能被检出"——没有它，组间频率差异可能只是幅度变化跨过检测阈值造成的假象。
* 反极性检测（假阳性率）：把记录翻转后用同样参数检测。噪声是对称的而真实事件只有一个方向，
  翻转后检出的事件数近似等于假阳性数。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..io import biexp_kernel

Detector = Callable[[np.ndarray], np.ndarray]   # 输入记录，返回事件起始点索引


def injection_recovery(y: np.ndarray, fs: float, detector: Detector, existing_onsets: np.ndarray,
                       amplitudes: list[float], tau_rise_ms: float, tau_decay_ms: float,
                       direction: str = "negative", n_per_amp: int = 40, tol_ms: float = 3.0,
                       seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sign = -1.0 if direction == "negative" else 1.0
    k = biexp_kernel(fs, tau_rise_ms, tau_decay_ms)
    n = len(y)
    guard = int(3 * tau_decay_ms / 1e3 * fs) + len(k) // 4
    tol = int(tol_ms / 1e3 * fs)
    rows = []
    for a in amplitudes:
        # 避开已有事件，互相间隔足够远
        pos, tries = [], 0
        while len(pos) < n_per_amp and tries < n_per_amp * 200:
            tries += 1
            p = int(rng.integers(guard, n - len(k) - guard))
            if len(existing_onsets) and np.min(np.abs(existing_onsets - p)) < guard:
                continue
            if pos and np.min(np.abs(np.asarray(pos) - p)) < 2 * guard:
                continue
            pos.append(p)
        yy = np.array(y, float)
        for p in pos:
            yy[p:p + len(k)] += sign * a * k
        found = np.asarray(detector(yy), int)
        hit = sum(bool(len(found) and np.min(np.abs(found - p)) <= tol) for p in pos)
        rows.append({"amplitude": a, "n_injected": len(pos), "n_recovered": hit,
                     "recovery": hit / len(pos) if pos else np.nan})
    return pd.DataFrame(rows)


def amplitude_at_recovery(curve: pd.DataFrame, level: float = 0.5) -> float:
    """线性插值得到检出率达到 level 时的幅度（检测下限）。"""
    c = curve.sort_values("amplitude")
    a, r = c["amplitude"].to_numpy(), c["recovery"].to_numpy()
    idx = np.nonzero(r >= level)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(a[0])
    return float(a[i - 1] + (level - r[i - 1]) * (a[i] - a[i - 1]) / (r[i] - r[i - 1]))


def inverted_polarity_fp(y: np.ndarray, fs: float, detector: Detector, n_detected: int) -> dict:
    """翻转记录后检测，估计假阳性率。"""
    y = np.asarray(y, float)
    flipped = 2 * np.median(y) - y
    n_fp = int(len(detector(flipped)))
    dur = len(y) / fs
    return {"n_false_positive": n_fp, "fp_rate_hz": n_fp / dur,
            "fp_fraction": n_fp / n_detected if n_detected else np.nan}
