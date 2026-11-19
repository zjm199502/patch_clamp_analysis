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
