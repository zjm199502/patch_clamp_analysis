"""事件时序依赖：前一个事件是否提高后一个事件发生的概率？

独立的自发释放近似泊松过程（IEI 呈指数分布、CV≈1、相邻 IEI 不相关）。偏离它的方式有：
簇状发放（burst；自相关在短延迟处抬高、分支比 > 0）或不应期/耗竭（短延迟处凹陷）。
所有显著性都用替代数据（surrogate）给出，而不依赖渐近公式。

注意：慢的频率漂移（rundown、药物洗入）本身会制造"聚集"的假象，
分析前请确认记录平稳，或截取平稳区段。
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def iei_stats(times: np.ndarray) -> dict:
    iei = np.diff(np.sort(times))
    iei = iei[iei > 0]
    if len(iei) < 5:
        return {"n": len(iei)}
    cv = iei.std(ddof=1) / iei.mean()
    cv2 = np.mean(2 * np.abs(np.diff(iei)) / (iei[1:] + iei[:-1]))       # 对慢漂移不敏感的局部 CV
    ks = stats.kstest(iei, "expon", args=(0, iei.mean()))
    rho, p = stats.spearmanr(iei[:-1], iei[1:])
    return {"n": len(iei), "cv": float(cv), "cv2": float(cv2), "ks_expon_D": float(ks.statistic),
            "ks_expon_p": float(ks.pvalue), "serial_rho": float(rho), "serial_p": float(p),
            "burst_fraction": float(np.mean(iei < 0.1 * iei.mean()))}


def serial_correlation(times: np.ndarray, max_lag: int = 10, n_shuffle: int = 1000, seed: int = 0) -> dict:
    """IEI 序列相关（lag 1..max_lag），用 IEI 打乱得到 95% 置信带。"""
    rng = np.random.default_rng(seed)
    iei = np.diff(np.sort(times))
    lags = np.arange(1, max_lag + 1)

    def sc(x):
        r = stats.rankdata(x)
        return np.array([np.corrcoef(r[:-l], r[l:])[0, 1] for l in lags])

    obs = sc(iei)
    null = np.array([sc(rng.permutation(iei)) for _ in range(n_shuffle)])
    lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
    p = (np.sum(np.abs(null) >= np.abs(obs), axis=0) + 1) / (n_shuffle + 1)
    return {"lags": lags, "rho": obs, "lo": lo, "hi": hi, "p": p}


def _acg(times, edges):
    t = np.sort(times)
    counts = np.zeros(len(edges) - 1)
    maxlag = edges[-1]
    for i in range(len(t)):
        hi = np.searchsorted(t, t[i] + maxlag, side="right")
        d = t[i + 1:hi] - t[i]
        if len(d):
            counts += np.histogram(d, edges)[0]
    return counts


def autocorrelogram(times: np.ndarray, duration_s: float, bin_ms: float = 20.0, max_lag_ms: float = 1000.0,
                    n_surrogate: int = 200, jitter_ms: float | None = None, seed: int = 0) -> dict:
    """事件自相关（条件发放率 / 平均发放率）。值 > 1 表示一个事件之后该延迟处更容易出现事件。

    替代数据用区间抖动：把每个事件在长度为 jitter_ms 的窗口内随机重排，保留慢于该窗口的
    速率变化（rundown、慢振荡），只破坏精细时序。默认窗口取 max(1 s, 5 个平均 IEI)。
    注意不能用"打乱 IEI"做这里的零假设——它保留了 IEI 分布，短间隔的富集也就一并保留了。
    """
    rng = np.random.default_rng(seed)
    t = np.sort(np.asarray(times, float))
    edges = np.arange(0, max_lag_ms + bin_ms, bin_ms) / 1e3
    n = len(t)
    rate = n / duration_s
    norm = n * rate * (bin_ms / 1e3)
    obs = _acg(t, edges) / norm
    sur = np.zeros((n_surrogate, len(obs)))
    w = (jitter_ms / 1e3) if jitter_ms else max(1.0, 5 / rate, 2 * max_lag_ms / 1e3)
    for s in range(n_surrogate):
        ts = np.clip(np.floor(t / w) * w + rng.uniform(0, w, n), 0, duration_s)
        sur[s] = _acg(ts, edges) / norm
    lo, hi = np.percentile(sur, [2.5, 97.5], axis=0)
    return {"lag_ms": (edges[:-1] + edges[1:]) / 2 * 1e3, "ratio": obs, "lo": lo, "hi": hi,
            "surrogate_mean": sur.mean(axis=0), "rate_hz": rate}


def cross_correlogram(times_a: np.ndarray, times_b: np.ndarray, duration_s: float, bin_ms: float = 5.0,
                      max_lag_ms: float = 200.0, n_surrogate: int = 200, jitter_ms: float = 50.0,
                      seed: int = 0) -> dict:
    """两列事件（如同一细胞的 EPSC 与 IPSC）的互相关；正延迟表示 b 滞后于 a。"""
    rng = np.random.default_rng(seed)
    a, b = np.sort(times_a), np.sort(times_b)
    edges = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms) / 1e3

    def cc(a_, b_):
        c = np.zeros(len(edges) - 1)
        for x in a_:
            lo, hi = np.searchsorted(b_, [x + edges[0], x + edges[-1]])
            if hi > lo:
                c += np.histogram(b_[lo:hi] - x, edges)[0]
        return c

    norm = len(a) * len(b) / duration_s * (bin_ms / 1e3)
    obs = cc(a, b) / norm
    w = jitter_ms / 1e3
    sur = np.array([cc(a, np.sort(np.floor(b / w) * w + rng.uniform(0, w, len(b)))) / norm
                    for _ in range(n_surrogate)])
    lo, hi = np.percentile(sur, [2.5, 97.5], axis=0)
    return {"lag_ms": (edges[:-1] + edges[1:]) / 2 * 1e3, "ratio": obs, "lo": lo, "hi": hi}


def _hawkes_nll(params, t, T, beta_min=0.0):
    mu, alpha, beta = np.exp(params[0]), 1 / (1 + np.exp(-params[1])), beta_min + np.exp(params[2])
    dt = np.diff(t)
    A = np.zeros(len(t))
    e = np.exp(-beta * dt)
    for i in range(1, len(t)):
        A[i] = e[i - 1] * (1 + A[i - 1])
    lam = mu + alpha * beta * A
    comp = mu * T + alpha * np.sum(1 - np.exp(-beta * (T - t)))
    return -(np.sum(np.log(lam)) - comp)


def hawkes_fit(times: np.ndarray, duration_s: float, max_trigger_tau_s: float | None = None) -> dict:
    """指数核 Hawkes 过程的极大似然拟合：λ(t) = μ + n·β·Σ exp(−β(t−t_i))。

    分支比 n = 每个事件平均直接触发的后续事件数（0 = 完全独立；越接近 1 越"自激"）。
    1/β 是触发效应的时间常数；μ 是独立的背景发放率。与齐次泊松做似然比检验。
    """
    t = np.sort(np.asarray(times, float))
    t = t - 0.0
    N, T = len(t), float(duration_s)
    if N < 30:
        return {"n_events": N, "message": "事件数不足（<30）"}
    rate = N / T
    # 触发时间常数设上限（默认 5 个平均 IEI，且不超过 2 s）：更慢的"聚集"属于速率漂移而不是事件间触发
    if max_trigger_tau_s is None:
        max_trigger_tau_s = min(5 / rate, 2.0)
    bmin = 1 / max_trigger_tau_s
    best = None
    for a0 in (0.1, 0.3, 0.6):
        for b0 in (rate * 2, rate * 10, rate * 50):
            x0 = [np.log(rate * (1 - a0)), np.log(a0 / (1 - a0)), np.log(b0 + bmin)]
            r = optimize.minimize(_hawkes_nll, x0, args=(t, T, bmin), method="Nelder-Mead",
                                  options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
            if best is None or r.fun < best.fun:
                best = r
    mu, alpha, beta = np.exp(best.x[0]), 1 / (1 + np.exp(-best.x[1])), bmin + np.exp(best.x[2])
    ll_h = -best.fun
    ll_p = N * np.log(rate) - N
    lr = 2 * (ll_h - ll_p)
    p = float(stats.chi2.sf(max(lr, 0), 2))
    return {"n_events": N, "mu_hz": float(mu), "branching_ratio": float(alpha),
            "trigger_tau_ms": float(1e3 / beta), "loglik_hawkes": float(ll_h),
            "loglik_poisson": float(ll_p), "lr_stat": float(lr), "p_vs_poisson": p,
            "delta_aic": float(2 * 2 - lr)}


def amplitude_dependence(times: np.ndarray, amps: np.ndarray) -> dict:
    """幅度的时序依赖：相邻事件幅度是否相关、幅度是否依赖于前一个 IEI（短时程可塑性/耗竭的线索）。"""
    o = np.argsort(times)
    t, a = np.asarray(times)[o], np.asarray(amps)[o]
    iei = np.diff(t)
    r1, p1 = stats.spearmanr(a[:-1], a[1:])
    r2, p2 = stats.spearmanr(iei, a[1:])
    return {"amp_serial_rho": float(r1), "amp_serial_p": float(p1),
            "amp_vs_preceding_iei_rho": float(r2), "amp_vs_preceding_iei_p": float(p2)}


def stability(times: np.ndarray, values: np.ndarray, duration_s: float, bin_s: float = 30.0) -> dict:
    """记录稳定性：分段频率与幅度及其线性趋势（检查 rundown / 串联电阻漂移）。"""
    edges = np.arange(0, duration_s + bin_s, bin_s)
    if len(edges) < 3:
        edges = np.linspace(0, duration_s, 4)
    centers = (edges[:-1] + edges[1:]) / 2
    cnt = np.histogram(times, edges)[0] / np.diff(edges)
    idx = np.digitize(times, edges) - 1
    med = np.array([np.nanmedian(values[idx == i]) if np.any(idx == i) else np.nan
                    for i in range(len(centers))])
    out = {"t": centers, "frequency": cnt, "value": med}
    ok = np.isfinite(med)
    if ok.sum() >= 3:
        lr = stats.linregress(centers[ok], med[ok])
        out["value_drift_pct"] = float(lr.slope * duration_s / np.nanmean(med) * 100)
        out["value_drift_p"] = float(lr.pvalue)
        lf = stats.linregress(centers, cnt)
        out["freq_drift_pct"] = float(lf.slope * duration_s / max(cnt.mean(), 1e-12) * 100)
        out["freq_drift_p"] = float(lf.pvalue)
    return out


def fano_factor(times: np.ndarray, duration_s: float, windows_s=None) -> dict:
    """计数 Fano 因子随窗口长度的变化。泊松过程恒为 1；随窗口增大而上升说明存在簇状发放或慢速率波动。"""
    if windows_s is None:
        windows_s = np.geomspace(0.05, max(duration_s / 10, 0.1), 12)
    ff = []
    for w in windows_s:
        c = np.histogram(times, np.arange(0, duration_s + w, w))[0][:-1]
        ff.append(c.var(ddof=1) / c.mean() if len(c) > 2 and c.mean() > 0 else np.nan)
    return {"window_s": np.asarray(windows_s), "fano": np.asarray(ff)}
