"""论文级绘图（matplotlib）。图内文字用英文，矢量导出时字体可编辑（pdf.fonttype=42）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import stats as st
from . import temporal as tp
from .features import FEATURE_LABELS

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
RC = {"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
      "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
      "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
      "svg.fonttype": "none", "font.family": "sans-serif",
      "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "xtick.major.width": 0.6,
      "ytick.major.width": 0.6, "legend.frameon": False}


def new_figure(w=7.0, h=4.5) -> Figure:
    import matplotlib as mpl
    mpl.rcParams.update(RC)
    return Figure(figsize=(w, h), dpi=120, layout="constrained")


def _cdf(ax, v, **kw):
    v = np.sort(np.asarray(v, float)[np.isfinite(v)])
    if len(v):
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), **kw)


def _label(m):
    base = m.replace("_mean", "").replace("_median", "")
    return FEATURE_LABELS.get(base, m)


# ------------------------------------------------------------------ 单细胞
def cell_overview(cell) -> tuple[Figure, str]:
    df, w = cell.valid()
    fs, y = cell.rec.fs, cell.processed
    fig = new_figure(7.2, 5.2)
    gs = fig.add_gridspec(3, 4)
    ax = fig.add_subplot(gs[0, :])
    i0 = int(df["onset_idx"].iloc[len(df) // 2]) if len(df) else 0
    a, b = max(i0 - int(2 * fs), 0), min(i0 + int(3 * fs), len(y))
    t = np.arange(a, b) / fs
    ax.plot(t, y[a:b], color="0.2", lw=0.4)
    ev = df[(df["peak_idx"] >= a) & (df["peak_idx"] < b)]
    ax.plot(ev["peak_idx"] / fs, y[ev["peak_idx"].to_numpy(int)], ".", color=PALETTE[3], ms=3)
    ax.set(xlabel="Time (s)", ylabel=f"I ({cell.rec.unit})", title=cell.name)

    sign = -1 if cell.params["direction"] == "negative" else 1
    ax = fig.add_subplot(gs[1, 0])
    if len(w):
        sel = np.random.default_rng(0).choice(len(w), min(len(w), 60), replace=False)
        ax.plot(cell.t_wave, sign * w[sel].T, color="0.75", lw=0.3)
        avg = np.nanmean(w, axis=0)
        ax.plot(cell.t_wave, sign * avg, color=PALETTE[0], lw=1.2)
    ax.set(xlabel="Time (ms)", ylabel=f"I ({cell.rec.unit})", title=f"Average event (n = {len(df)})")

    for k, (m, c) in enumerate([("amplitude", 0), ("rise_10_90", 1), ("decay_tau", 2)]):
        ax = fig.add_subplot(gs[1, k + 1])
        v = pd.to_numeric(df[m], errors="coerce").dropna()
        if len(v):
            ax.hist(v.clip(upper=v.quantile(0.99)), bins=30, color=PALETTE[c], alpha=0.85)
        ax.set(xlabel=FEATURE_LABELS[m], ylabel="Events")

    ax = fig.add_subplot(gs[2, 0]); _cdf(ax, df["amplitude"], color=PALETTE[0])
    ax.set(xlabel=FEATURE_LABELS["amplitude"], ylabel="Cumulative fraction")
    ax = fig.add_subplot(gs[2, 1]); _cdf(ax, df["iei"], color=PALETTE[1])
    ax.set(xlabel=FEATURE_LABELS["iei"], ylabel="Cumulative fraction", xscale="log")
    stab = tp.stability(df["time_s"].to_numpy(), df["amplitude"].to_numpy(), cell.rec.duration)
    ax = fig.add_subplot(gs[2, 2]); ax.plot(stab["t"], stab["frequency"], "-o", ms=2.5, color=PALETTE[1])
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(0, None))
    ax = fig.add_subplot(gs[2, 3]); ax.plot(stab["t"], stab["value"], "-o", ms=2.5, color=PALETTE[0])
    ax.set(xlabel="Time (s)", ylabel="Median amplitude (pA)", ylim=(0, None))

    s = cell.summary()
    bi = st.biexp_decay_fit(np.nanmean(w, axis=0), fs) if len(w) else {}
    ie = tp.iei_stats(df["time_s"].to_numpy())
    txt = [f"细胞：{cell.name}", f"事件数 {s['n_events']}，频率 {s['frequency']:.3g} Hz，记录时长 {s['duration_s']:.1f} s",
           f"噪声 RMS（稳健估计）{s['noise_rms']:.3g} pA",
           f"幅度 中位数 {s['amplitude_median']:.3g} pA（均值 {s['amplitude_mean']:.3g}，CV {s['amplitude_cv']:.2f}）",
           f"10–90% 上升时间 中位数 {s['rise_10_90_median']:.3g} ms；衰减 τ 中位数 {s['decay_tau_median']:.3g} ms",
           f"半宽 {s['half_width_median']:.3g} ms；电荷 {s['charge_median']:.3g} fC"]
    if "tau_mono" in bi:
        txt.append(f"平均事件衰减：单指数 τ = {bi['tau_mono']:.3g} ms")
    if "tau_weighted" in bi:
        txt.append(f"  双指数 τfast = {bi['tau_fast']:.3g} ms（占 {bi['frac_fast']:.0%}），τslow = {bi['tau_slow']:.3g} ms，"
                   f"加权 τ = {bi['tau_weighted']:.3g} ms；{'双指数明显更优（ΔAIC=%.0f），提示混合成分' % bi['delta_aic'] if bi.get('biexp_preferred') else '单指数已足够'}")
    if "cv" in ie:
        txt.append(f"IEI：CV = {ie['cv']:.2f}，CV2 = {ie['cv2']:.2f}（泊松过程均≈1）；与指数分布的 KS p = {ie['ks_expon_p']:.3g}")
    if "value_drift_pct" in stab:
        warn = "；⚠ 漂移明显，建议检查串联电阻或截取平稳区段" if (abs(stab["value_drift_pct"]) > 20 or abs(stab["freq_drift_pct"]) > 30) else ""
        txt.append(f"稳定性：记录全程幅度漂移 {stab['value_drift_pct']:+.0f}%（p={stab['value_drift_p']:.2g}），"
                   f"频率漂移 {stab['freq_drift_pct']:+.0f}%（p={stab['freq_drift_p']:.2g}）{warn}")
    return fig, "\n".join(txt)


def detection_qc(cell, n_per_amp: int = 30) -> tuple[Figure, str]:
    from . import validation as va
    df, w = cell.valid()
    p, fs = cell.params, cell.rec.fs
    noise = cell.qc.get("noise_rms", np.nan)
    det = lambda y: cell.detector_fn()(y).onsets
    amps = list(np.round(noise * np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0]), 2))
    tr, td = p["tau_rise_ms"], p["tau_decay_ms"]
    if len(w) >= 20:
        from .deconv import fit_kernel_from_average
        tr, td = fit_kernel_from_average(np.nanmean(w, axis=0), fs)
    curve = va.injection_recovery(cell.processed, fs, det, df["onset_idx"].to_numpy(), amps, tr, td,
                                  p["direction"], n_per_amp=n_per_amp)
    lim = va.amplitude_at_recovery(curve, 0.5)
    fp = va.inverted_polarity_fp(cell.processed, fs, det, len(df))
    cell.qc.update(detection_limit_pA=lim, fp_rate_hz=fp["fp_rate_hz"], fp_fraction=fp["fp_fraction"])
    scan = st.threshold_scan(df, cell.rec.duration)

    fig = new_figure(7.2, 2.4)
    ax = fig.subplots(1, 3)
    ax[0].plot(curve["amplitude"], curve["recovery"], "-o", ms=3, color=PALETTE[0])
    ax[0].axhline(0.5, color="0.6", ls=":", lw=0.6)
    if np.isfinite(lim):
        ax[0].axvline(lim, color=PALETTE[3], ls="--", lw=0.7)
    ax[0].set(xlabel="Injected amplitude (pA)", ylabel="Recovery (sensitivity)", ylim=(0, 1.05),
              title="Synthetic event injection")
    v = df["amplitude"].dropna()
    ax[1].hist(v.clip(upper=v.quantile(0.99)), bins=40, color=PALETTE[0], alpha=0.85)
    if np.isfinite(lim):
        ax[1].axvline(lim, color=PALETTE[3], ls="--", lw=0.7, label="50% detection limit")
    ax[1].axvline(2.5 * noise, color="0.4", ls=":", lw=0.7, label="2.5 × noise RMS")
    ax[1].legend(); ax[1].set(xlabel="Amplitude (pA)", ylabel="Events", title="Amplitudes vs detection limit")
    ax[2].plot(scan["threshold"], scan["frequency"], "-o", ms=3, color=PALETTE[1])
    ax[2].set(xlabel="Score threshold", ylabel="Frequency (Hz)", title="Threshold robustness", ylim=(0, None))
    near = float(np.mean(v < 1.5 * lim)) if np.isfinite(lim) else np.nan
    txt = [f"检测器：{p['detector']}；噪声 RMS = {noise:.3g} pA",
           f"合成事件注入（模板 τrise {tr:.2g} ms / τdecay {td:.2g} ms，每个幅度 {n_per_amp} 个）：",
           *[f"  {r.amplitude:.3g} pA → 检出 {r.n_recovered}/{r.n_injected}（{r.recovery:.0%}）" for r in curve.itertuples()],
           f"50% 检出幅度（检测下限）≈ {lim:.3g} pA" if np.isfinite(lim) else "检测下限：在测试范围内未达到 50% 检出",
           f"反极性检测：翻转记录后检出 {fp['n_false_positive']} 个事件 → 假阳性率 ≈ {fp['fp_rate_hz'] * 60:.2g} 个/分钟，"
           f"约占已检出事件的 {fp['fp_fraction']:.1%}",
           f"有 {near:.0%} 的事件幅度低于 1.5 × 检测下限。"
           + ("⚠ 幅度分布紧贴检测下限：组间“频率”差异可能只是幅度变化把事件推过/推下阈值，需结合幅度分布解读。" if near > 0.3 else "幅度分布与检测下限分离良好。")]
    return fig, "\n".join(txt)


def cluster_figure(res, df, waves, t_wave, direction="negative") -> Figure:
    fig = new_figure(7.2, 4.6)
    gs = fig.add_gridspec(2, 3)
    sign = -1 if direction == "negative" else 1
    lab = res.labels
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(*res.embedding[lab < 0].T, s=3, color="0.8", lw=0)
    for u in range(res.k):
        ax.scatter(*res.embedding[lab == u].T, s=4, color=PALETTE[u % 6], lw=0, alpha=0.7, label=f"C{u + 1}")
    ax.set(xlabel="Dim 1", ylabel="Dim 2", title="Embedding"); ax.legend(markerscale=2)
    ax = fig.add_subplot(gs[0, 1])
    for u in range(res.k):
        m = lab == u
        sem = np.nanstd(waves[m], axis=0) / np.sqrt(m.sum())
        ax.fill_between(t_wave, sign * (res.mean_waves[u] - sem), sign * (res.mean_waves[u] + sem),
                        color=PALETTE[u % 6], alpha=0.25, lw=0)
        ax.plot(t_wave, sign * res.mean_waves[u], color=PALETTE[u % 6])
    ax.set(xlabel="Time (ms)", ylabel="I (pA)", title="Cluster averages")
    ax = fig.add_subplot(gs[0, 2])
    for u in range(res.k):
        ax.plot(t_wave, res.mean_waves_norm[u], color=PALETTE[u % 6])
    ax.set(xlabel="Time (ms)", ylabel="Normalized", title="Peak-scaled (kinetics only)")
    ax = fig.add_subplot(gs[1, 0])
    if res.bic:
        ks = sorted(res.bic); ax.plot(ks, [res.bic[k] for k in ks], "-o", ms=3, color="0.3")
        ax.set(xlabel="Number of clusters (GMM)", ylabel="BIC", title="Model selection", xticks=ks)
    ax = fig.add_subplot(gs[1, 1])
    for u in range(res.k):
        d = df[lab == u]
        ax.scatter(d["rise_10_90"], d["decay_tau"], s=4, color=PALETTE[u % 6], lw=0, alpha=0.6)
    ax.set(xlabel=FEATURE_LABELS["rise_10_90"], ylabel=FEATURE_LABELS["decay_tau"], xscale="log", yscale="log")
    ax = fig.add_subplot(gs[1, 2])
    for u in range(res.k):
        _cdf(ax, df.loc[lab == u, "amplitude"], color=PALETTE[u % 6])
    ax.set(xlabel=FEATURE_LABELS["amplitude"], ylabel="Cumulative fraction")
    return fig


def rise_amp_figure(cell) -> tuple[Figure, str]:
    df, _ = cell.valid()
    r = st.rise_amplitude_relation(df)
    fig = new_figure(4.8, 2.4); ax = fig.subplots(1, 2)
    ax[0].scatter(df["rise_10_90"], df["amplitude"], s=4, color=PALETTE[0], alpha=0.5, lw=0)
    ax[0].set(xlabel=FEATURE_LABELS["rise_10_90"], ylabel=FEATURE_LABELS["amplitude"])
    ax[1].scatter(df["rise_10_90"], df["decay_tau"], s=4, color=PALETTE[2], alpha=0.5, lw=0)
    ax[1].set(xlabel=FEATURE_LABELS["rise_10_90"], ylabel=FEATURE_LABELS["decay_tau"])
    if "rho_rise_amp" not in r:
        return fig, "事件数不足。"
    if r["p_rise_amp"] < 0.05 and r["rho_rise_amp"] < 0:
        interp = "上升越慢的事件幅度越小：符合树突电紧张滤波（远端输入又慢又小）。比较组间幅度时，建议只取快上升事件或把上升时间作为协变量。"
    elif r["p_rise_amp"] < 0.05:
        interp = "上升越慢幅度反而越大：不符合单纯的树突滤波，可能存在动力学不同的突触亚群（建议做动力学聚类）或重叠事件。"
    else:
        interp = "上升时间与幅度无显著相关：幅度差异不太可能由电紧张衰减造成。"
    txt = (f"上升时间–幅度：Spearman ρ = {r['rho_rise_amp']:.2f}，p = {r['p_rise_amp']:.3g}（n = {r['n']}）\n"
           f"上升时间–衰减 τ：ρ = {r['rho_rise_decay']:.2f}，p = {r['p_rise_decay']:.3g}"
           f"{'（正相关同样是树突滤波的特征）' if r['rho_rise_decay'] > 0 and r['p_rise_decay'] < 0.05 else ''}\n"
           f"最快 1/3 事件的幅度中位数 {r['amp_fast_third']:.3g} pA；最慢 1/3 为 {r['amp_slow_third']:.3g} pA\n{interp}")
    return fig, txt


def temporal_figure(cell) -> tuple[Figure, str]:
    df, _ = cell.valid()
    t, T = df["time_s"].to_numpy(), cell.rec.duration
    rate = len(t) / T
    binms = float(np.clip(1000 / max(rate, 0.1) / 10, 5, 100))
    ac = tp.autocorrelogram(t, T, bin_ms=binms, max_lag_ms=binms * 40)
    sc = tp.serial_correlation(t)
    ff = tp.fano_factor(t, T)
    hk = tp.hawkes_fit(t, T)
    ie = tp.iei_stats(t)
    ad = tp.amplitude_dependence(t, df["amplitude"].to_numpy())
    fig = new_figure(7.2, 2.4); ax = fig.subplots(1, 3)
    ax[0].fill_between(ac["lag_ms"], ac["lo"], ac["hi"], color="0.85", lw=0, label="95% surrogate")
    ax[0].plot(ac["lag_ms"], ac["ratio"], color=PALETTE[0]); ax[0].axhline(1, color="0.5", ls=":", lw=0.6)
    ax[0].set(xlabel="Lag after an event (ms)", ylabel="Rate / mean rate", title="Autocorrelogram"); ax[0].legend()
    ax[1].fill_between(sc["lags"], sc["lo"], sc["hi"], color="0.85", lw=0)
    ax[1].plot(sc["lags"], sc["rho"], "-o", ms=3, color=PALETTE[1]); ax[1].axhline(0, color="0.5", ls=":", lw=0.6)
    ax[1].set(xlabel="Lag (intervals)", ylabel="IEI serial correlation (ρ)", title="IEI serial correlation")
    ax[2].plot(ff["window_s"], ff["fano"], "-o", ms=3, color=PALETTE[2]); ax[2].axhline(1, color="0.5", ls=":", lw=0.6)
    ax[2].set(xlabel="Counting window (s)", ylabel="Fano factor", xscale="log", title="Fano factor")
    early = ac["ratio"][:3] > ac["hi"][:3]
    dip = ac["ratio"][:3] < ac["lo"][:3]
    txt = [f"事件数 {len(t)}，平均频率 {rate:.3g} Hz"]
    if "cv" in ie:
        txt.append(f"IEI：CV = {ie['cv']:.2f}，CV2 = {ie['cv2']:.2f}；与指数分布比较 KS p = {ie['ks_expon_p']:.3g}；"
                   f"相邻 IEI 相关 ρ = {ie['serial_rho']:.2f}（p = {ie['serial_p']:.3g}）")
    if "branching_ratio" in hk:
        txt.append(f"Hawkes 自激模型：分支比 n = {hk['branching_ratio']:.2f}（每个事件平均触发 {hk['branching_ratio']:.2f} 个后续事件），"
                   f"触发时间常数 {hk['trigger_tau_ms']:.3g} ms，背景频率 μ = {hk['mu_hz']:.3g} Hz；"
                   f"相对泊松的似然比检验 p = {hk['p_vs_poisson']:.3g}")
    txt.append(f"幅度时序依赖：相邻事件幅度相关 ρ = {ad['amp_serial_rho']:.2f}（p = {ad['amp_serial_p']:.3g}）；"
               f"幅度 vs 前一个 IEI ρ = {ad['amp_vs_preceding_iei_rho']:.2f}（p = {ad['amp_vs_preceding_iei_p']:.3g}）")
    sig = hk.get("p_vs_poisson", 1) < 0.01 and hk.get("branching_ratio", 0) > 0.05
    if early.any() and sig:
        txt.append("结论：短延迟处事件率显著高于替代数据，且 Hawkes 模型显著优于泊松 → 存在“一个事件之后更容易跟着另一个事件”的簇状发放。"
                   "mini 条件（TTX）下提示多囊泡/同步释放或钙依赖的释放耦合；s 条件下更可能反映突触前神经元的簇状放电。")
    elif dip.any():
        txt.append("结论：短延迟处事件率低于替代数据 → 存在不应期样的抑制（也可能是重叠事件漏检造成的假象，请结合去卷积视图检查）。")
    else:
        txt.append("结论：未发现显著偏离独立（泊松）发放的证据。")
    txt.append("提示：慢的频率漂移会抬高 CV、Fano 因子和分支比，请先在“细胞总览”中确认记录平稳。CV2 对慢漂移不敏感，可作对照。")
    return fig, "\n".join(txt)


def nsfa_figure(cell, cluster: int = -1, driving_force_mv: float = 70.0) -> tuple[Figure, str]:
    df, w = cell.valid()
    m = ~df["overlap"].to_numpy(bool)
    if cluster >= 0:
        m &= df["cluster"].to_numpy() == cluster
    r = st.peak_scaled_nsfa(w[m], cell.rec.fs, driving_force_mv=driving_force_mv)
    fig = new_figure(3.0, 2.4); ax = fig.subplots()
    if "I" in r:
        ax.plot(r["I"], r["var"], "o", ms=3, color=PALETTE[0])
    if "fit_I" in r:
        ax.plot(r["fit_I"], r["fit_var"], color=PALETTE[3])
    ax.set(xlabel="Mean current (pA)", ylabel="Variance (pA²)", title="Peak-scaled NSFA")
    if "i_pA" not in r:
        return fig, r.get("message", "拟合失败")
    txt = (f"峰值缩放非平稳噪声分析（{r['n_events']} 个无重叠事件{'，簇 %d' % (cluster + 1) if cluster >= 0 else ''}）\n"
           f"单通道电流 i = {r['i_pA']:.3g} ± {r['i_se']:.2g} pA；驱动力 {driving_force_mv:g} mV 下单通道电导 γ ≈ {r['gamma_pS']:.3g} pS\n"
           f"峰值时开放通道数 N ≈ {r['N']:.3g}（N 的估计在抛物线未明显弯曲时很不可靠，SE = {r['N_se']:.2g}）\n"
           "解读：组间幅度变化若伴随 γ 不变，说明受体数目改变（如 AMPAR 插入/内吞）；γ 改变则提示亚基组成或磷酸化状态改变。\n"
           "前提：动力学均一、对齐良好、空间钳位充分；低通滤波和树突滤波都会使 γ 被低估。")
    return fig, txt


def quantal_figure(cell) -> tuple[Figure, str]:
    from scipy import stats as ss
    df, _ = cell.valid()
    a = df["amplitude"].dropna().to_numpy()
    r = st.quantal_peaks(a)
    fig = new_figure(3.4, 2.4); ax = fig.subplots()
    ax.hist(a, bins=50, density=True, color="0.8")
    if "means" not in r:
        return fig, r.get("message", "")
    x = np.linspace(0, np.percentile(a, 99.5), 400)
    tot = np.zeros_like(x)
    for mu, sd, wt in zip(r["means"], r["sds"], r["weights"]):
        comp = wt * ss.norm.pdf(x, mu, sd); tot += comp
        ax.plot(x, comp, lw=0.7, color=PALETTE[0])
    ax.plot(x, tot, color=PALETTE[3]); ax.set(xlabel="Amplitude (pA)", ylabel="Density", xlim=(0, x[-1]))
    txt = [f"幅度分布：CV = {r['cv']:.2f}，偏度 = {r['skew']:.2f}；高斯混合 BIC 最优成分数 k = {r['k']}",
           "各峰均值（pA）：" + "，".join(f"{m:.3g}（{w_:.0%}）" for m, w_ in zip(r["means"], r["weights"]))]
    if r["k"] > 1:
        txt.append(f"峰位比（相对第一峰 q = {r['q']:.3g} pA）：" + "，".join(f"{x_:.2f}" for x_ in r["peak_ratios"])
                   + ("；近似整数倍 → 支持量子化/多囊泡释放" if r["equally_spaced"] else "；不是整数倍 → 不支持等间距量子峰"))
    txt.append("单个对数正态分布的 BIC 更低：右偏单峰（不同突触 q 不同）是更简约的解释，多峰可能只是拟合偏态的产物。"
               if r["lognormal_preferred"] else "高斯混合优于单个对数正态分布。")
    return fig, "\n".join(txt)


# ------------------------------------------------------------------ 组间
def _group_bar(axis, i, v, rng):
    axis.bar(i, np.mean(v) if len(v) else 0, width=0.6, color=PALETTE[i % 6], alpha=0.35)
    if len(v) > 1:
        axis.errorbar(i, np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)), color="k", capsize=2, lw=0.7)
    axis.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=PALETTE[i % 6], edgecolor="k", lw=0.3, zorder=3)


def group_comparison(project, metric: str = "amplitude_median", event_metric: str | None = None,
                     n_boot: int = 1000) -> tuple[Figure, str]:
    cells = project.cell_table()
    events = project.events_table()
    cells = cells[cells["group"].astype(str) != ""]
    res = st.compare_groups(cells, metric)
    groups = res["groups"]
    em = event_metric or metric.replace("_mean", "").replace("_median", "").replace("_cv", "")
    has_ev = em in events.columns
    fig = new_figure(7.2 if has_ev else 2.6, 2.6)
    ax = fig.subplots(1, 3 if has_ev else 1, width_ratios=[1, 1.4, 1.4] if has_ev else None)
    ax0 = ax[0] if has_ev else ax
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        v = pd.to_numeric(cells.loc[cells["group"] == g, metric], errors="coerce").dropna().to_numpy()
        _group_bar(ax0, i, v, rng)
    ax0.set(xticks=range(len(groups)), xticklabels=[f"{g}\n(n={n})" for g, n in zip(groups, res["n"])],
            ylabel=_label(metric), ylim=(0, None))
    if np.isfinite(res.get("p", np.nan)):
        ax0.set_title(f"p = {res['p']:.3g}")
    txt = ["【细胞水平：每个点是一个细胞】", st.format_comparison(res)]
    if has_ev and len(groups) >= 2:
        for i, g in enumerate(groups):
            for c, sub in events[events["group"] == g].groupby("cell"):
                _cdf(ax[1], sub[em], color=PALETTE[i % 6], alpha=0.35, lw=0.5)
        pooled = st.pooled_cdf(events, em)
        for i, g in enumerate(groups):
            if g in pooled:
                _cdf(ax[1], pooled[g], color=PALETTE[i % 6], lw=1.6, label=str(g))
        ax[1].set(xlabel=_label(em), ylabel="Cumulative fraction", title="Per-cell (thin) and pooled CDFs")
        ax[1].legend()
        try:
            hb = st.hierarchical_bootstrap(events, em, animal="animal", n_boot=n_boot)
            for i, g in enumerate(groups):
                ax[2].hist(hb["boot"][g], bins=40, color=PALETTE[i % 6], alpha=0.6, density=True, label=str(g))
            ax[2].set(xlabel=f"Bootstrapped {_label(em)}", ylabel="Density", title="Hierarchical bootstrap")
            txt += ["", f"【层级 bootstrap（{hb['levels']}，{n_boot} 次）】"]
            txt += [f"  {g}: {hb['estimate'][g]:.4g}，95% CI [{hb['ci'][g][0]:.4g}, {hb['ci'][g][1]:.4g}]" for g in groups]
            if "p_two_sided" in hb:
                txt.append(f"  P({groups[1]} > {groups[0]}) = {hb['p_greater']:.3f}，双侧 p ≈ {hb['p_two_sided']:.3g}")
        except Exception as e:
            txt.append(f"层级 bootstrap 失败：{e}")
        try:
            lm = st.event_level_lmm(events[events["group"].astype(str) != ""], em)
            txt += ["", f"【事件水平线性混合模型：log({em}) ~ group + (1|cell)】",
                    f"  {lm['n_events']} 个事件 / {lm['n_cells']} 个细胞；{lm['method']}，p = {lm['p']:.4g}"]
            txt += [f"  {k}: 倍数变化 = {v:.3f}" for k, v in lm["fold_change"].items()]
        except Exception as e:
            txt.append(f"事件水平混合模型失败：{e}")
        txt += ["", "说明：统计推断以细胞（有动物编号时以动物）为单位。把所有事件合并后直接做 KS 检验会因伪重复严重夸大显著性，"
                    "合并的累积分布图只用于展示分布形状。"]
    return fig, "\n".join(txt)


def scaling_figure(project, metric: str = "amplitude") -> tuple[Figure, str]:
    events = project.events_table()
    events = events[events["group"].astype(str) != ""]
    groups = list(pd.unique(events["group"]))
    if len(groups) < 2:
        return new_figure(3, 2), "需要两个组（第一组视为对照）。"
    pooled = st.pooled_cdf(events, metric, n_per_cell=100)
    c, t = pooled[groups[0]], pooled[groups[1]]
    r = st.scaling_analysis(c, t, n=min(len(c), len(t), 1000))
    fig = new_figure(7.2, 2.4); ax = fig.subplots(1, 3)
    ax[0].plot(r["ctrl_sorted"], r["treat_sorted"], ".", ms=2, color=PALETTE[0])
    lim = [0, max(r["ctrl_sorted"].max(), r["treat_sorted"].max())]
    ax[0].plot(lim, lim, color="0.6", ls=":", lw=0.6)
    xs = np.array(lim); ax[0].plot(xs, r["slope"] * xs + r["intercept"], color=PALETTE[3], lw=0.8)
    ax[0].set(xlabel=f"{groups[0]} (ranked, pA)", ylabel=f"{groups[1]} (ranked, pA)", title="Rank-order plot")
    ratio = r["treat_sorted"] / r["ctrl_sorted"]
    ax[1].plot(r["ctrl_sorted"], ratio, ".", ms=2, color=PALETTE[1]); ax[1].axhline(1, color="0.6", ls=":", lw=0.6)
    ax[1].set(xlabel=f"{groups[0]} amplitude (pA)", ylabel="Ratio (treated / control)", title="Ratio plot",
              ylim=(0, np.percentile(ratio, 99) * 1.2))
    _cdf(ax[2], c, color=PALETTE[0], label=str(groups[0])); _cdf(ax[2], t, color=PALETTE[1], label=str(groups[1]))
    _cdf(ax[2], t / r["iterative_factor"], color=PALETTE[1], ls="--", label=f"{groups[1]} ÷ {r['iterative_factor']:.2f}")
    ax[2].set(xlabel=FEATURE_LABELS.get(metric, metric), ylabel="Cumulative fraction"); ax[2].legend()
    lo, hi = np.percentile(ratio[: len(ratio) // 3], 50), np.percentile(ratio[-len(ratio) // 3:], 50)
    if r["multiplicative"]:
        concl = f"除以统一因子 {r['iterative_factor']:.2f} 后两组分布重合（KS p = {r['iterative_ks_p']:.3g}）→ 支持均一的乘性缩放（所有突触按同一比例改变）。"
    elif abs(hi - lo) / max(lo, hi) > 0.15:
        concl = (f"比例图不平坦（小事件比值 {lo:.2f}，大事件比值 {hi:.2f}）→ 非均一（发散性）缩放：不同强度的突触改变比例不同，"
                 "或只有一部分突触发生了改变。")
    else:
        concl = "单一乘性因子不能使分布重合；可能存在加性成分或检测阈值截断的影响。"
    txt = (f"秩序图线性回归：{groups[1]} = {r['slope']:.3f} × {groups[0]} + {r['intercept']:.3g}（R² = {r['r2']:.3f}）\n"
           f"迭代法最优乘性因子 = {r['iterative_factor']:.2f}（缩放后 KS p = {r['iterative_ks_p']:.3g}；未缩放 KS p = {r['ks_raw_p']:.3g}）\n"
           f"{concl}\n每个细胞等量抽取 100 个事件后合并。检测阈值会截掉小事件并扭曲截距，建议先做检测质控。")
    return fig, txt


def cluster_composition(project) -> tuple[Figure, str]:
    ct = project.cell_table()
    cols = [c for c in ct.columns if c.startswith("cluster") and c.endswith("_fraction")]
    if not cols:
        return new_figure(3, 2), "请先在“动力学聚类”页对“全部细胞合并”做聚类。"
    ct = ct[ct["group"].astype(str) != ""]
    groups = list(pd.unique(ct["group"]))
    fig = new_figure(1.6 + 1.4 * len(cols), 2.4); ax = fig.subplots()
    rng = np.random.default_rng(0)
    wd = 0.8 / max(len(groups), 1)
    txt = ["各簇在每个细胞中所占比例（每个点是一个细胞）："]
    for j, col in enumerate(cols):
        for i, g in enumerate(groups):
            v = ct.loc[ct["group"] == g, col].dropna().to_numpy()
            x = j + (i - (len(groups) - 1) / 2) * wd
            ax.bar(x, v.mean() if len(v) else 0, wd * 0.9, color=PALETTE[i % 6], alpha=0.35, label=str(g) if j == 0 else None)
            ax.scatter(x + rng.uniform(-wd / 4, wd / 4, len(v)), v, s=8, color=PALETTE[i % 6], edgecolor="k", lw=0.3, zorder=3)
        if len(groups) >= 2:
            r = st.compare_groups(ct, col)
            txt.append(f"  簇 {j + 1} 比例：" + "，".join(f"{g} {m:.2f}" for g, m in zip(r["groups"], r["mean"]))
                       + (f"；{r['test']} p = {r['p']:.3g}" if np.isfinite(r.get("p", np.nan)) else ""))
            fc = col.replace("_fraction", "_frequency")
            r2 = st.compare_groups(ct, fc)
            txt.append(f"  簇 {j + 1} 频率（Hz）：" + "，".join(f"{g} {m:.3g}" for g, m in zip(r2["groups"], r2["mean"]))
                       + (f"；p = {r2['p']:.3g}" if np.isfinite(r2.get("p", np.nan)) else ""))
    ax.set(xticks=range(len(cols)), xticklabels=[f"Cluster {j + 1}" for j in range(len(cols))],
           ylabel="Fraction of events", ylim=(0, 1)); ax.legend()
    txt.append("解读：某一簇的频率选择性改变，提示处理只影响了某一类突触输入（特定来源或特定树突区域），这是总体频率看不出来的。")
    return fig, "\n".join(txt)


def ei_figure(cell_e, cell_i) -> tuple[Figure, str]:
    de, _ = cell_e.valid(); di, _ = cell_i.valid()
    r = st.ei_balance(de, di, cell_e.rec.duration, cell_i.rec.duration)
    fig = new_figure(4.6, 2.4); ax = fig.subplots(1, 2)
    ax[0].bar([0, 1], [r["charge_rate_E"], r["charge_rate_I"]], color=[PALETTE[3], PALETTE[0]])
    ax[0].set(xticks=[0, 1], xticklabels=["E", "I"], ylabel="Synaptic charge rate (fC/s)")
    txt = [f"E：{cell_e.name}；I：{cell_i.name}",
           f"频率 E {r['freq_E']:.3g} Hz / I {r['freq_I']:.3g} Hz（比值 {r['ei_freq_ratio']:.2f}）；幅度比 {r['ei_amp_ratio']:.2f}",
           f"单位时间电荷 E {r['charge_rate_E']:.3g} / I {r['charge_rate_I']:.3g} fC/s → E/I 电荷比 {r['ei_charge_ratio']:.2f}，"
           f"兴奋占比 E/(E+I) = {r['e_fraction']:.2f}"]
    if abs(cell_e.rec.duration - cell_i.rec.duration) < 1e-6 and cell_e.rec.path == cell_i.rec.path:
        cc = tp.cross_correlogram(de["time_s"].to_numpy(), di["time_s"].to_numpy(), cell_e.rec.duration)
        ax[1].fill_between(cc["lag_ms"], cc["lo"], cc["hi"], color="0.85", lw=0)
        ax[1].plot(cc["lag_ms"], cc["ratio"], color=PALETTE[4])
        ax[1].set(xlabel="IPSC lag relative to EPSC (ms)", ylabel="Rate / chance", title="E–I cross-correlogram")
    else:
        ax[1].axis("off")
        txt.append("两段记录不是同时记录的，互相关不适用（互相关只对同一段记录中同时检测的 E 与 I 事件有意义）。")
    txt.append("注意：若 E 与 I 在不同钳制电位记录，电荷比受驱动力影响，应换算为电导后再比较。")
    return fig, "\n".join(txt)


def cumulative_figure(project, metrics=("amplitude", "iei"), mode: str = "cell_average",
                      n_per_cell: int = 100, log_iei: bool = True, design_aware=False) -> tuple[Figure, str]:
    """幅度与 IEI（及其他事件特征）的累积概率分布，多组叠加。

    mode：
    * "cell_average"（默认）：先求每个细胞自己的累积分布，再在公共网格上按细胞平均，阴影为 SEM。
      每个细胞权重相同，是发表时最推荐的画法；
    * "pooled_equal"：每个细胞等量抽取 n_per_cell 个事件后合并；
    * "pooled_all"：所有事件直接合并（事件多的细胞权重更大，仅供展示）。
    细细线为单个细胞。组间检验在细胞水平上做（每个细胞的中位数），而不是对合并事件做 KS。
    """
    events = project.events_table()
    events = events[events["group"].astype(str) != ""]
    groups = list(pd.unique(events["group"]))
    cells_tab = project.cell_table()
    cells_tab = cells_tab[cells_tab["group"].astype(str) != ""]
    fig = new_figure(2.6 * len(metrics) + 0.4, 2.6)
    axes = np.atleast_1d(fig.subplots(1, len(metrics)))
    txt = [{"cell_average": "画法：每个细胞的累积分布按细胞平均（粗线 ± SEM 阴影），细线为单个细胞。",
            "pooled_equal": f"画法：每个细胞等量抽取 {n_per_cell} 个事件后合并（粗线），细线为单个细胞。",
            "pooled_all": "画法：全部事件合并（粗线；事件多的细胞权重更大，仅供展示），细线为单个细胞。"}[mode]]
    rng = np.random.default_rng(0)
    for ax, m in zip(axes, metrics):
        vals_all = pd.to_numeric(events[m], errors="coerce")
        vals_all = vals_all[np.isfinite(vals_all) & (vals_all > 0 if m == "iei" else True)]
        if not len(vals_all):
            ax.axis("off"); continue
        lo, hi = np.nanpercentile(vals_all, [0.5, 99.5])
        logx = m == "iei" and log_iei and lo > 0
        grid = np.geomspace(lo, hi, 300) if logx else np.linspace(min(lo, 0) if m != "iei" else lo, hi, 300)
        curves = {}
        units = {}
        for gi, g in enumerate(groups):
            col = PALETTE[gi % 6]
            per_cell = []
            animal_ids = []
            pooled = []
            for _, sub in events[events["group"] == g].groupby("cell"):
                v = pd.to_numeric(sub[m], errors="coerce").to_numpy(float)
                v = np.sort(v[np.isfinite(v) & ((v > 0) if m == "iei" else True)])
                if len(v) < 5:
                    continue
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, alpha=0.25, lw=0.4)
                per_cell.append(np.searchsorted(v, grid, side="right") / len(v))
                animal_ids.append(str(sub["animal"].iloc[0]).strip())
                pooled.append(v if mode == "pooled_all" else rng.choice(v, n_per_cell, replace=len(v) < n_per_cell))
            if not per_cell:
                continue
            curves[g] = np.asarray(per_cell)
            units[g] = animal_ids
            if mode == "cell_average":
                P = np.array(per_cell)
                mu = P.mean(axis=0)
                if len(P) > 1:
                    se = P.std(axis=0, ddof=1) / np.sqrt(len(P))
                    ax.fill_between(grid, mu - se, mu + se, color=col, alpha=0.25, lw=0)
                ax.plot(grid, mu, color=col, lw=1.6, label=f"{g} (n={len(P)})")
            else:
                _cdf(ax, np.concatenate(pooled), color=col, lw=1.6, label=f"{g} (n={len(per_cell)})")
        ax.set(xlabel=FEATURE_LABELS.get(m, m), ylabel="Cumulative probability", ylim=(0, 1.02))
        if logx:
            ax.set_xscale("log")
        ax.legend(loc="best")
        from itertools import combinations
        from scipy.stats import permutation_test
        comparisons = []
        for first, second in combinations(curves, 2):
            left, right = curves[first], curves[second]
            permutation_type = "independent"
            if design_aware:
                ids = [units[first], units[second]]
                if any(not value for side in ids for value in side):
                    txt.append(f"{m}：{first} vs {second} 动物编号不全，仅展示。")
                    continue
                animals = [sorted(set(side)) for side in ids]
                if set(animals[0]) & set(animals[1]):
                    if animals[0] != animals[1]:
                        txt.append(f"{m}：{first} vs {second} 部分配对，建议混合模型；不报告置换 p。")
                        continue
                    permutation_type = "samples"
                left, right = [np.array([values[np.array(side)==animal].mean(axis=0) for animal in names])
                               for values, side, names in zip((left,right), ids, animals)]
            if min(len(left), len(right)) < 2:
                txt.append(f"{m}：{first} vs {second}，细胞数不足，未检验。")
                continue
            def distance(one, two):
                return np.max(np.abs(one.mean(axis=0) - two.mean(axis=0)))
            combined = np.concatenate([left, right])
            def statistic(one, two):
                return distance(combined[one], combined[two])
            test = permutation_test((np.arange(len(left)), np.arange(len(left), len(combined))),
                statistic, permutation_type=permutation_type, vectorized=False, alternative="greater", n_resamples=999, random_state=0)
            if design_aware:
                txt.append(f"{m}：{first} vs {second} 推荐并执行动物{'内配对交换' if permutation_type=='samples' else '间独立标签置换'}，n={len(left)}/{len(right)}。")
            comparisons.append((first, second, test.statistic, test.pvalue))
        for first, second, delta, probability in comparisons:
            txt.append(f"{m}：{first} vs {second}，平均细胞 CDF 最大差={delta:.3g}，"
                       f"置换 p（该指标内 Bonferroni 校正）={min(1., probability * len(comparisons)):.3g}。")
        cm = "iei_median" if m == "iei" else f"{m}_median"
        if not design_aware and len(groups) >= 2 and cm in cells_tab:
            r = st.compare_groups(cells_tab, cm)
            line = f"{FEATURE_LABELS.get(m, m)} 每个细胞的中位数：" + "，".join(
                f"{g} {mm:.3g} ± {se:.2g}" for g, mm, se in zip(r["groups"], r["mean"], r["sem"]))
            if np.isfinite(r.get("p", np.nan)):
                line += f"（{r['test']}，p = {r['p']:.3g}）"
                ax.set_title(f"p = {r['p']:.3g}")
            txt.append(line)
    if "iei" in metrics:
        txt.append("IEI 分布右移 = 频率降低；左移 = 频率升高。IEI 横轴默认取对数。")
    txt.append("展示曲线按选定画法；检验先在动物内平均细胞 CDF。跨组重复编号必须代表同一动物。" if design_aware else
               "CDF 置换以细胞为单位，假设细胞独立，不是动物水平检验。")
    return fig, "\n".join(txt)
