"""事件动力学、检测阈值与噪声分析。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats


# ---------------------------------------------------------------- 峰值缩放非平稳噪声分析
def peak_scaled_nsfa(waves: np.ndarray, fs: float, n_bins: int = 12, driving_force_mv: float | None = None,
                     max_events: int = 2000) -> dict:
    """峰值缩放非平稳噪声分析（Traynelis et al., 1993; Hartveit & Veruki, 2007）。

    把平均波形缩放到每个事件的峰值，求衰减相的方差—电流关系：σ² = i·I − I²/N + σ_b²。
    得到突触后受体的单通道电流 i（给出驱动力后换算为电导 γ）和峰值时开放的通道数 N。
    要求事件对齐良好、无重叠、动力学均一——建议先聚类，在单个簇内做。
    """
    w = np.asarray(waves, float)
    w = w[np.isfinite(w).all(axis=1)][:max_events]
    if len(w) < 30:
        return {"message": "事件数不足（<30）"}
    mean = w.mean(axis=0)
    pk = int(np.argmax(mean))
    sp = max(int(0.0002 * fs), 1)
    peaks = w[:, max(pk - sp, 0):pk + sp + 1].mean(axis=1)
    mpk = mean[max(pk - sp, 0):pk + sp + 1].mean()
    ok = peaks > 0
    w, peaks = w[ok], peaks[ok]
    scaled = mean[None, :] * (peaks[:, None] / mpk)
    var = ((w - scaled) ** 2).sum(axis=0) / (len(w) - 1)
    pre = max(int(np.argmax(mean > 0.05 * mpk)) - sp, 2)
    bg = float(np.mean(w[:, :pre].var(axis=0, ddof=1))) if pre > 2 else 0.0
    I, V = mean[pk:], var[pk:]
    keep = I > 0.05 * mpk
    I, V = I[keep], V[keep]
    edges = np.linspace(I.min(), I.max(), n_bins + 1)
    idx = np.clip(np.digitize(I, edges) - 1, 0, n_bins - 1)
    Ib = np.array([I[idx == b].mean() for b in range(n_bins) if np.any(idx == b)])
    Vb = np.array([V[idx == b].mean() for b in range(n_bins) if np.any(idx == b)])

    def model(I_, i_, N_):
        return i_ * I_ - I_ ** 2 / N_ + bg

    try:
        popt, pcov = optimize.curve_fit(model, Ib, Vb, p0=(max(Vb.max() / Ib.max() * 2, 0.1), 50.0),
                                        bounds=([1e-3, 1.0], [100.0, 1e6]))
        i_, N_ = popt
        err = np.sqrt(np.diag(pcov))
    except Exception:
        return {"message": "拟合失败", "I": Ib, "var": Vb}
    out = {"i_pA": float(i_), "i_se": float(err[0]), "N": float(N_), "N_se": float(err[1]),
           "background_var": bg, "I": Ib, "var": Vb, "fit_I": np.linspace(0, Ib.max(), 100),
           "n_events": int(len(w)), "p_open_peak": float(mpk / (i_ * N_))}
    out["fit_var"] = model(out["fit_I"], i_, N_)
    if driving_force_mv:
        out["gamma_pS"] = float(i_ / abs(driving_force_mv) * 1e3)
    return out


# ---------------------------------------------------------------- 阈值稳健性
def threshold_scan(df: pd.DataFrame, duration_s: float, thresholds=None) -> pd.DataFrame:
    """按检测得分逐级提高阈值，看事件数/频率/幅度如何变化。结论只在某个阈值成立则不稳健。"""
    sc = pd.to_numeric(df["score"], errors="coerce")
    if thresholds is None:
        thresholds = np.linspace(np.nanmin(sc), np.nanpercentile(sc, 90), 12)
    rows = []
    for th in thresholds:
        sub = df[sc >= th]
        rows.append({"threshold": float(th), "n_events": len(sub), "frequency": len(sub) / duration_s,
                     "amplitude_median": float(sub["amplitude"].median()) if len(sub) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 树突滤波诊断
def rise_amplitude_relation(df: pd.DataFrame) -> dict:
    """上升时间–幅度相关。显著负相关提示幅度差异部分来自电紧张衰减（远端输入又慢又小）；
    无相关则支持事件来自电紧张距离相近的突触，或空间钳位良好。同时给出上升–衰减相关。"""
    d = df[["rise_10_90", "amplitude", "decay_tau"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 10:
        return {"n": len(d)}
    r1 = stats.spearmanr(d["rise_10_90"], d["amplitude"])
    r2 = stats.spearmanr(d["rise_10_90"], d["decay_tau"])
    fast = d[d["rise_10_90"] <= d["rise_10_90"].quantile(1 / 3)]
    slow = d[d["rise_10_90"] >= d["rise_10_90"].quantile(2 / 3)]
    return {"n": len(d), "rho_rise_amp": float(r1.statistic), "p_rise_amp": float(r1.pvalue),
            "rho_rise_decay": float(r2.statistic), "p_rise_decay": float(r2.pvalue),
            "amp_fast_third": float(fast["amplitude"].median()),
            "amp_slow_third": float(slow["amplitude"].median())}


# ---------------------------------------------------------------- 量子/多峰分析
def quantal_peaks(amplitudes: np.ndarray, k_max: int = 5, noise_sd: float | None = None, seed: int = 0) -> dict:
    """幅度直方图的多峰拟合（1D 高斯混合 + BIC）。

    若最优 k>1 且各峰均值近似等间距（q, 2q, 3q…），提示量子化/多囊泡释放；
    mEPSC 的幅度分布通常是单峰右偏的（各突触 q 不同），多峰并不常见，解读需谨慎。
    """
    from sklearn.mixture import GaussianMixture

    a = np.asarray(amplitudes, float)
    a = a[np.isfinite(a) & (a > 0)]
    if len(a) < 50:
        return {"message": "事件数不足（<50）"}
    X = a[:, None]
    bic, models = {}, {}
    for k in range(1, k_max + 1):
        g = GaussianMixture(k, n_init=3, random_state=seed).fit(X)
        bic[k], models[k] = float(g.bic(X)), g
    k = min(bic, key=bic.get)
    g = models[k]
    o = np.argsort(g.means_.ravel())
    means, sds, wts = g.means_.ravel()[o], np.sqrt(g.covariances_.ravel()[o]), g.weights_[o]
    out = {"k": int(k), "bic": bic, "means": means, "sds": sds, "weights": wts,
           "cv": float(a.std() / a.mean()), "skew": float(stats.skew(a))}
    if k > 1:
        q = means[0]
        ratios = means / q
        out["q"] = float(q)
        out["peak_ratios"] = ratios
        out["equally_spaced"] = bool(np.all(np.abs(ratios - np.round(ratios)) < 0.2))
    # 对数正态拟合（单峰右偏的常见替代解释）
    sh, loc, sc = stats.lognorm.fit(a, floc=0)
    ll_ln = np.sum(stats.lognorm.logpdf(a, sh, loc, sc))
    out["bic_lognormal"] = float(2 * np.log(len(a)) - 2 * ll_ln)
    out["lognormal_preferred"] = bool(out["bic_lognormal"] < bic[k])
    return out


# ---------------------------------------------------------------- 平均事件双指数衰减
def biexp_decay_fit(avg: np.ndarray, fs: float) -> dict:
    """平均事件衰减相的单/双指数拟合，返回加权 τ。双指数明显更优提示混合受体成分
    （如 AMPA+NMDA、不同 GABA_A 亚基）或混合突触来源。"""
    pk = int(np.nanargmax(avg))
    y = avg[pk:]
    t = np.arange(len(y)) * 1e3 / fs
    out = {}
    try:
        p1, _ = optimize.curve_fit(lambda t, a, tau: a * np.exp(-t / tau), t, y,
                                   p0=(y[0], t[-1] / 5), maxfev=5000)
        r1 = np.sum((y - p1[0] * np.exp(-t / p1[1])) ** 2)
        out.update(tau_mono=float(p1[1]))
        f2 = lambda t, a1, t1, a2, t2: a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2)
        p2, _ = optimize.curve_fit(f2, t, y, p0=(y[0] * 0.7, p1[1] / 2, y[0] * 0.3, p1[1] * 3),
                                   bounds=([0, 1e-2, 0, 1e-2], [np.inf, 1e3, np.inf, 1e4]), maxfev=20000)
        r2 = np.sum((y - f2(t, *p2)) ** 2)
        (a1, t1), (a2, t2) = sorted([(p2[0], p2[1]), (p2[2], p2[3])], key=lambda x: x[1])
        n = len(y)
        out.update(tau_fast=float(t1), tau_slow=float(t2), frac_fast=float(a1 / (a1 + a2)),
                   tau_weighted=float((a1 * t1 + a2 * t2) / (a1 + a2)),
                   delta_aic=float(n * np.log(r1 / n) + 4 - (n * np.log(r2 / n) + 8)))
        out["biexp_preferred"] = bool(out["delta_aic"] > 10 and 0.05 < out["frac_fast"] < 0.95
                                      and t2 / t1 > 2)
    except Exception as e:
        out["message"] = str(e)
    return out


# ---------------------------------------------------------------- E/I 联合分析
def ei_balance(epsc: pd.DataFrame, ipsc: pd.DataFrame, dur_e: float, dur_i: float) -> dict:
    """同一细胞的兴奋/抑制比：频率比、幅度比，以及单位时间突触电荷比（最接近"总驱动"的指标）。"""
    qe = epsc["charge"].clip(lower=0).sum() / dur_e
    qi = ipsc["charge"].clip(lower=0).sum() / dur_i
    return {"freq_E": len(epsc) / dur_e, "freq_I": len(ipsc) / dur_i,
            "ei_freq_ratio": (len(epsc) / dur_e) / max(len(ipsc) / dur_i, 1e-12),
            "ei_amp_ratio": float(epsc["amplitude"].median() / ipsc["amplitude"].median()),
            "charge_rate_E": float(qe), "charge_rate_I": float(qi),
            "ei_charge_ratio": float(qe / qi) if qi else np.nan,
            "e_fraction": float(qe / (qe + qi)) if qe + qi else np.nan}
