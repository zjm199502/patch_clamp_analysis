"""统计分析：以细胞/动物为单位的层级统计、累积分布、突触缩放、峰值缩放非平稳噪声分析。"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats


# ---------------------------------------------------------------- 组间比较
def hedges_g(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    return float((b.mean() - a.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def _holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def compare_groups(cells: pd.DataFrame, metric: str, group: str = "group",
                   animal: str | None = "animal") -> dict:
    """细胞水平的组间比较（n = 细胞数）。

    若提供了动物编号且同一动物有多个细胞，则用线性混合模型 metric ~ group + (1|animal)，
    避免把来自同一只动物的细胞当作独立样本（伪重复）。
    """
    d = cells[[metric, group] + ([animal] if animal and animal in cells else [])].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric, group])
    d = d[d[group].astype(str) != ""]
    groups = list(pd.unique(d[group]))
    samples = [d.loc[d[group] == g, metric].to_numpy(float) for g in groups]
    out = {"metric": metric, "groups": groups, "n": [len(s) for s in samples],
           "mean": [float(np.mean(s)) if len(s) else np.nan for s in samples],
           "sem": [float(stats.sem(s)) if len(s) > 1 else np.nan for s in samples],
           "median": [float(np.median(s)) if len(s) else np.nan for s in samples]}
    if len(groups) < 2 or min(out["n"]) < 2:
        out.update(test="—", p=np.nan, note="每组至少需要 2 个细胞。")
        return out
    normal = all(len(s) < 3 or stats.shapiro(s).pvalue > 0.05 for s in samples)
    out["normal"] = bool(normal)

    nested = (animal and animal in d and d[animal].astype(str).ne("").all()
              and d[animal].nunique() < len(d) and d[animal].nunique() >= 6)
    if nested:
        try:
            import statsmodels.formula.api as smf
            warnings.simplefilter("ignore")
            dd = d.rename(columns={metric: "y", group: "g", animal: "a"})
            m = smf.mixedlm("y ~ C(g)", dd, groups=dd["a"]).fit(reml=True)
            m0 = smf.mixedlm("y ~ C(g)", dd, groups=dd["a"]).fit(reml=False)
            n0 = smf.mixedlm("y ~ 1", dd, groups=dd["a"]).fit(reml=False)
            lr = 2 * (m0.llf - n0.llf)
            out.update(test="线性混合模型 (1|animal)，似然比检验", stat=float(lr),
                       p=float(stats.chi2.sf(max(lr, 0), len(groups) - 1)),
                       n_animals=int(d[animal].nunique()),
                       lmm_table=m.summary().tables[1].to_string())
        except Exception as e:  # 模型不收敛时退回细胞水平检验
            nested = False
            out["note"] = f"混合模型拟合失败（{e}），已退回细胞水平检验。"
    if not nested:
        if len(groups) == 2:
            a, b = samples
            if normal:
                r = stats.ttest_ind(a, b, equal_var=False)
                out.update(test="Welch t 检验", stat=float(r.statistic), p=float(r.pvalue))
            else:
                r = stats.mannwhitneyu(a, b, alternative="two-sided")
                out.update(test="Mann–Whitney U 检验", stat=float(r.statistic), p=float(r.pvalue))
            out["hedges_g"] = hedges_g(a, b)
        else:
            if normal:
                r = stats.f_oneway(*samples)
                out.update(test="单因素 ANOVA", stat=float(r.statistic), p=float(r.pvalue))
            else:
                r = stats.kruskal(*samples)
                out.update(test="Kruskal–Wallis 检验", stat=float(r.statistic), p=float(r.pvalue))
            pairs, ps = [], []
            for i, j in itertools.combinations(range(len(groups)), 2):
                pp = (stats.ttest_ind(samples[i], samples[j], equal_var=False).pvalue if normal
                      else stats.mannwhitneyu(samples[i], samples[j], alternative="two-sided").pvalue)
                pairs.append((groups[i], groups[j]))
                ps.append(pp)
            out["posthoc"] = [dict(a=a, b=b, p_holm=float(p)) for (a, b), p in zip(pairs, _holm(ps))]
    return out


def format_comparison(res: dict) -> str:
    lines = [f"指标：{res['metric']}"]
    for g, n, m, s, md in zip(res["groups"], res["n"], res["mean"], res["sem"], res["median"]):
        lines.append(f"  {g}: n = {n} 个细胞，均值 ± SEM = {m:.4g} ± {s:.3g}，中位数 = {md:.4g}")
    if "p" in res and np.isfinite(res.get("p", np.nan)):
        lines.append(f"检验：{res['test']}，统计量 = {res.get('stat', np.nan):.4g}，p = {res['p']:.4g}")
    if "hedges_g" in res:
        lines.append(f"效应量 Hedges g = {res['hedges_g']:.3g}")
    if "n_animals" in res:
        lines.append(f"动物数 = {res['n_animals']}")
    for ph in res.get("posthoc", []):
        lines.append(f"  事后比较 {ph['a']} vs {ph['b']}: Holm 校正 p = {ph['p_holm']:.4g}")
    if res.get("note"):
        lines.append("注意：" + res["note"])
    if "lmm_table" in res:
        lines.append(res["lmm_table"])
    return "\n".join(lines)


def event_level_lmm(events: pd.DataFrame, metric: str, group: str = "group", cell: str = "cell",
                    log: bool = True) -> dict:
    """事件水平的混合模型：metric ~ group + (1|cell)。用上全部事件，同时承认事件嵌套在细胞内。"""
    import statsmodels.formula.api as smf

    warnings.simplefilter("ignore")
    d = events[[metric, group, cell]].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d[np.isfinite(d[metric]) & (d[metric] > 0 if log else True)]
    d["y"] = np.log(d[metric]) if log else d[metric]
    d = d.rename(columns={group: "g", cell: "c"})
    cm = d.groupby(["c", "g"], as_index=False)["y"].mean()
    gm = cm.groupby("g")["y"].mean()
    k = d["g"].nunique() - 1
    out = {"metric": metric, "log": log, "n_events": len(d), "n_cells": int(d["c"].nunique())}
    ok = False
    for method in ("lbfgs", "powell", "nm"):
        try:
            full = smf.mixedlm("y ~ C(g)", d, groups=d["c"]).fit(reml=False, method=method)
            null = smf.mixedlm("y ~ 1", d, groups=d["c"]).fit(reml=False, method=method)
            lr = 2 * (full.llf - null.llf)
            eff = {name: float(v) for name, v in full.fe_params.items() if name != "Intercept"}
            # 合理性检查：固定效应应与"各细胞均值的组间差"一致，否则视为未收敛
            ref = np.array([gm[g] - gm.iloc[0] for g in gm.index[1:]])
            if np.isfinite(lr) and np.all(np.isfinite(full.bse_fe)) and np.allclose(list(eff.values()), ref, atol=0.25 + 0.5 * np.abs(ref).max()):
                out.update(method="线性混合模型 (1|cell)，似然比检验", lr_stat=float(lr),
                           p=float(stats.chi2.sf(max(lr, 0), k)), fixed_effects=eff,
                           table=full.summary().tables[1].to_string())
                ok = True
                break
        except Exception:
            continue
    if not ok:
        # 退回两阶段法：先求每个细胞的均值，再在细胞间做检验（对嵌套设计同样有效，只是功效略低）
        samples = [cm.loc[cm["g"] == g, "y"].to_numpy() for g in gm.index]
        r = stats.ttest_ind(*samples, equal_var=False) if len(samples) == 2 else stats.f_oneway(*samples)
        eff = {f"C(g)[T.{g}]": float(gm[g] - gm.iloc[0]) for g in gm.index[1:]}
        out.update(method="混合模型未收敛，改用两阶段法（细胞均值间检验）", lr_stat=float(r.statistic),
                   p=float(r.pvalue), fixed_effects=eff, table="")
    out["fold_change"] = {n: float(np.exp(v)) for n, v in out["fixed_effects"].items()} if log else {}
    return out


def pooled_cdf(events: pd.DataFrame, metric: str, group: str = "group", cell: str = "cell",
               n_per_cell: int = 50, seed: int = 0) -> dict:
    """每个细胞抽取相同数量的事件再合并，避免事件多的细胞主导累积分布。"""
    rng = np.random.default_rng(seed)
    out = {}
    for g, sub in events.groupby(group):
        if str(g) == "":
            continue
        vals = []
        for _, c in sub.groupby(cell):
            v = pd.to_numeric(c[metric], errors="coerce").dropna().to_numpy()
            if len(v) == 0:
                continue
            vals.append(rng.choice(v, n_per_cell, replace=len(v) < n_per_cell))
        if vals:
            out[g] = np.sort(np.concatenate(vals))
    return out


# ---------------------------------------------------------------- 突触缩放
def scaling_analysis(ctrl: np.ndarray, treat: np.ndarray, n: int = 500, seed: int = 0) -> dict:
    """秩次排序法检验乘性突触缩放（Turrigiano et al., 1998; Kim et al., 2012）。

    两组各取等量事件按大小排序后做线性回归 treat = a·ctrl + b。
    纯乘性缩放时 b≈0，且把处理组除以 a 后的分布应与对照组重合（KS 检验不显著）。
    检测阈值会截断小事件、使截距偏离 0，因此同时给出 Kim et al. 的迭代缩放因子。
    """
    rng = np.random.default_rng(seed)
    c = np.sort(rng.choice(ctrl, n, replace=len(ctrl) < n))
    t = np.sort(rng.choice(treat, n, replace=len(treat) < n))
    lr = stats.linregress(c, t)
    scaled_back = (t - lr.intercept) / lr.slope
    ks_lin = stats.ks_2samp(c, scaled_back)
    ks_raw = stats.ks_2samp(c, t)
    # 迭代法：寻找使 KS 的 p 最大的纯乘性因子（缩小后的事件若低于对照组最小值则剔除）
    best = (np.nan, -1.0)
    thr = c.min()
    for f in np.arange(0.5, 2.5001, 0.01):
        v = t / f
        v = v[v >= thr]
        if len(v) < 20:
            continue
        p = stats.ks_2samp(c, v).pvalue
        if p > best[1]:
            best = (f, p)
    return {"slope": float(lr.slope), "intercept": float(lr.intercept), "r2": float(lr.rvalue ** 2),
            "ks_raw_p": float(ks_raw.pvalue), "ks_scaled_p": float(ks_lin.pvalue),
            "iterative_factor": float(best[0]), "iterative_ks_p": float(best[1]),
            "multiplicative": bool(best[1] > 0.05), "ctrl_sorted": c, "treat_sorted": t}


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


# ---------------------------------------------------------------- 层级 bootstrap
def hierarchical_bootstrap(events: pd.DataFrame, metric: str, group: str = "group", cell: str = "cell",
                           animal: str | None = None, n_boot: int = 2000, stat: str = "median",
                           seed: int = 0) -> dict:
    """层级 bootstrap（Saravanan et al., 2020）：先重抽动物、再重抽细胞、再重抽事件。

    返回每组统计量的 bootstrap 分布与 95% CI；两组时给出 P(组2 > 组1) 及对应的双侧 p。
    """
    rng = np.random.default_rng(seed)
    f = np.nanmedian if stat == "median" else np.nanmean
    d = events[[metric, group, cell] + ([animal] if animal and animal in events else [])].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric])
    d = d[d[group].astype(str) != ""]
    use_animal = bool(animal and animal in d and d[animal].astype(str).ne("").all())
    boots, groups = {}, list(pd.unique(d[group]))
    for g in groups:
        sub = d[d[group] == g]
        tree = {}
        for c, cs in sub.groupby(cell):
            a = cs[animal].iloc[0] if use_animal else c
            tree.setdefault(a, []).append(cs[metric].to_numpy(float))
        animals = list(tree)
        res = np.empty(n_boot)
        for b in range(n_boot):
            cell_stats = []
            for a in rng.choice(len(animals), len(animals)):
                cells_ = tree[animals[a]]
                for ci in rng.choice(len(cells_), len(cells_)):
                    v = cells_[ci]
                    cell_stats.append(f(rng.choice(v, len(v))))
            res[b] = np.mean(cell_stats)
        boots[g] = res
    out = {"metric": metric, "groups": groups, "boot": boots,
           "ci": {g: tuple(np.percentile(boots[g], [2.5, 97.5])) for g in groups},
           "estimate": {g: float(np.mean(boots[g])) for g in groups}, "levels":
           "动物 → 细胞 → 事件" if use_animal else "细胞 → 事件"}
    if len(groups) == 2:
        p_gt = float(np.mean(boots[groups[1]] > boots[groups[0]]))
        out["p_greater"] = p_gt
        out["p_two_sided"] = float(min(1.0, 2 * min(p_gt, 1 - p_gt) + 1 / n_boot))
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
